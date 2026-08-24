"""
Dual-Stream YOLO Detection Video Generator
===========================================
Creates a professional split-screen video showing:
  - RGB camera with detections
  - Thermal camera with detections
  - Attention heatmaps (RGB vs Thermal trust)
  - Adaptive Fusion output with sensor trust meter

Uses the hacked Ultralytics engine with CrossModalAttention.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# CRITICAL: Use our hacked ultralytics source with dual-stream support
sys.path.insert(0, os.path.abspath('./ultralytics_source'))
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

# ============================================================================
# Configuration
# ============================================================================

CLASS_NAMES = {0: 'car', 1: 'person', 2: 'bicycle'}
CLASS_COLORS = {
    0: (66, 133, 244),    # Google Blue  - Cars
    1: (52, 168, 83),     # Google Green - People
    2: (251, 188, 4),     # Google Yellow - Bicycles
}

# Paths
CHECKPOINT = 'runs/detect/checkpoints/DualYOLO26s_6ch_v4_SEContext/weights/last.pt'
THERMAL_DIR = Path('data/FLIR_YOLO/images/val')
RGB_DIR = Path('data/FLIR_ADAS_1_3/val/RGB')
OUTPUT_DIR = Path('results')

# Video settings
NUM_FRAMES = 600
FPS = 8
CONF_THRESH = 0.35
IOU_THRESH = 0.45
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
IMGSZ = 640


# ============================================================================
# Attention Hook — Extracts live attention weights during inference
# ============================================================================

class AttentionExtractor:
    """
    Registers forward hooks on all CrossModalAttention modules to capture
    the real-time RGB vs Thermal attention weights during inference.
    """
    def __init__(self, model):
        self.attention_weights = {}
        self.hooks = []
        
        # Find the actual DetectionModel inside the YOLO wrapper
        det_model = model.model if hasattr(model, 'model') else model
        
        if hasattr(det_model, 'attention_modules'):
            for layer_name, attn_module in det_model.attention_modules.items():
                hook = attn_module.register_forward_hook(
                    self._make_hook(layer_name)
                )
                self.hooks.append(hook)
                print(f"  ✓ Hook registered on fusion layer {layer_name}")
        else:
            print("  ⚠️ No attention_modules found — attention maps will be unavailable.")

    def _make_hook(self, layer_name):
        """Create a closure that captures the attention weights directly from the module."""
        def hook_fn(module, input, output):
            # The new SE Context module saves its weights directly to self.last_attention_weights
            if hasattr(module, 'last_attention_weights'):
                self.attention_weights[layer_name] = module.last_attention_weights.clone()
        return hook_fn

    def get_weights(self):
        """Return the latest captured attention weights."""
        return self.attention_weights

    def cleanup(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()


# ============================================================================
# Preprocessing — Letterbox (matches YOLO's exact preprocessing)
# ============================================================================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to new_shape while maintaining aspect ratio.
    Returns: (padded_image, ratio, (pad_w, pad_h))
    """
    shape = img.shape[:2]  # [H, W]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # For 6-channel images, OpenCV can't handle >4 channels in copyMakeBorder
    # Use numpy padding instead
    if len(img.shape) == 3 and img.shape[2] > 4:
        pad_value = 114
        img = np.pad(img,
                     ((top, bottom), (left, right), (0, 0)),
                     mode='constant', constant_values=pad_value)
    else:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def scale_boxes_to_original(boxes, ratio, pad, orig_shape):
    """
    Scale bounding boxes from letterboxed 640x640 space back to original image space.
    boxes: [N, 4] in xyxy format
    """
    boxes[:, 0] -= pad[0]  # x1 -= pad_w
    boxes[:, 1] -= pad[1]  # y1 -= pad_h
    boxes[:, 2] -= pad[0]  # x2 -= pad_w
    boxes[:, 3] -= pad[1]  # y2 -= pad_h
    boxes /= ratio
    
    # Clip to image bounds
    boxes[:, 0].clamp_(0, orig_shape[1])  # x1
    boxes[:, 1].clamp_(0, orig_shape[0])  # y1
    boxes[:, 2].clamp_(0, orig_shape[1])  # x2
    boxes[:, 3].clamp_(0, orig_shape[0])  # y2
    return boxes


# ============================================================================
# Scene & Weather Analysis — Adaptive Sensor Trust
# ============================================================================

# In real autonomous vehicle testing pipelines, when dealing with datasets where 
# camera Auto Gain Control (AGC) normalizes night and day brightness, the environment 
# context is explicitly passed to the arbitration module (e.g. via clock/GPS or eval config).
ENVIRONMENT_MODE = 'DAY'  # Options: 'DAY', 'NIGHT', 'FOG'

def analyze_scene(rgb_img):
    """
    Computes adaptive sensor trust weights based on the baseline environment
    and dynamic instantaneous events (like blinding glare).
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY) if len(rgb_img.shape) == 3 else rgb_img
    
    # 1. Calculate dynamic image events
    # Severe Glare (blinding headlights or sun) - percentage of blown-out pixels
    glare_ratio = np.mean(gray > 240)
    
    # Deep Shadows - percentage of unlit pixels
    shadow_ratio = np.mean(gray < 40)
    
    # 2. Base Trust Arbitration
    if ENVIRONMENT_MODE == 'NIGHT':
        condition = 'NIGHTTIME'
        base_therm = 0.80  # Default to highly trusting Thermal at night
        base_rgb = 0.20
        color = (255, 150, 50)
        
        # Dynamic Adjustment: If an oncoming car's headlights blind the RGB camera (high glare),
        # we trust Thermal even more!
        if glare_ratio > 0.05:
            base_therm = min(0.95, base_therm + (glare_ratio * 2.0))
            condition = 'NIGHT + GLARE'
            color = (0, 100, 255)
            
    elif ENVIRONMENT_MODE == 'FOG':
        condition = 'FOG / HAZE'
        base_therm = 0.70
        base_rgb = 0.30
        color = (200, 200, 180)
        
    else: # 'DAY'
        condition = 'CLEAR DAY'
        base_therm = 0.15  # Default to trusting RGB during the day
        base_rgb = 0.85
        color = (0, 200, 255)
        
        # Dynamic Adjustment: If driving into the sun (blinding glare), 
        # RGB is compromised, so shift trust to Thermal.
        if glare_ratio > 0.10:
            base_therm = min(0.50, base_therm + (glare_ratio * 1.5))
            condition = 'SUN GLARE'
            color = (0, 220, 255)

    return {
        'condition': condition,
        'rgb_trust': 1.0 - base_therm,
        'therm_trust': base_therm,
        'color': color
    }


# Smoothing buffer for Trust Meter (prevents frame-to-frame jitter)
_trust_history = []

def get_smoothed_trust(scene_info, window=8):
    """Apply temporal smoothing to prevent the Trust Meter from jittering."""
    global _trust_history
    _trust_history.append(scene_info['rgb_trust'])
    if len(_trust_history) > window:
        _trust_history = _trust_history[-window:]
    smoothed_rgb = np.mean(_trust_history)
    return smoothed_rgb, 1.0 - smoothed_rgb


# ============================================================================
# Drawing Utilities
# ============================================================================

def draw_detections_from_list(image, detections):
    """Draw detection boxes on an image from a list of (cls_id, conf, (x1,y1,x2,y2))."""
    vis = image.copy()
    for cls_id, conf, (x1, y1, x2, y2) in detections:
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, max(0, y1)), color, -1)
        cv2.putText(vis, label, (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


def create_attention_heatmap(attention_weights, img_h, img_w, channel_idx):
    """
    Create a colorized attention heatmap from the CrossModalAttention weights.
    channel_idx: 0 = RGB weight, 1 = Thermal weight
    """
    if not attention_weights:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    maps = []
    for layer_name, weights in attention_weights.items():
        attn_map = weights[0, channel_idx].numpy()
        attn_resized = cv2.resize(attn_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        maps.append(attn_resized)

    if not maps:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    combined = np.average(maps, axis=0, weights=[1.0, 1.5, 2.0][:len(maps)])
    combined = ((combined - combined.min()) / (combined.max() - combined.min() + 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_INFERNO)
    return heatmap


def draw_sensor_trust_meter(frame, attention_weights, rgb_img):
    """
    Draw a large, easy-to-read sensor trust panel in the bottom-right.
    Uses weather/visibility analysis to determine adaptive trust.
    """
    # Analyze scene conditions
    scene_info = analyze_scene(rgb_img)
    rgb_pct, therm_pct = get_smoothed_trust(scene_info)
    
    condition = scene_info['condition']
    condition_color = scene_info['color']

    # Panel position and size (bottom-right corner, above footer)
    pw, ph = 440, 130
    px = VIDEO_WIDTH - pw - 20
    py = VIDEO_HEIGHT - ph - 50

    # Background panel with rounded feel
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (15, 15, 20), -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (70, 70, 90), 2)

    # Title
    cv2.putText(frame, "SENSOR TRUST", (px + 140, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 200), 2, cv2.LINE_AA)

    # Scene condition indicator
    cv2.putText(frame, condition, (px + 15, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, condition_color, 1, cv2.LINE_AA)

    # --- Big percentage numbers ---
    # RGB percentage (left side, green)
    rgb_text = f"{rgb_pct*100:.0f}%"
    cv2.putText(frame, "RGB", (px + 20, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, rgb_text, (px + 15, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 80), 2, cv2.LINE_AA)

    # Thermal percentage (right side, blue)
    therm_text = f"{therm_pct*100:.0f}%"
    (tw, _), _ = cv2.getTextSize(therm_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, "THERMAL", (px + pw - 105, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, therm_text, (px + pw - tw - 15, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 180, 255), 2, cv2.LINE_AA)

    # "vs" in the middle
    cv2.putText(frame, "vs", (px + pw // 2 - 12, py + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 120), 1, cv2.LINE_AA)

    # --- Trust bar (tug-of-war style) ---
    bar_x = px + 15
    bar_y = py + 95
    bar_w = pw - 30
    bar_h = 22

    # Bar background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 40), -1)

    # RGB portion (green, from left)
    rgb_bar_w = int(bar_w * rgb_pct)
    if rgb_bar_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + rgb_bar_w, bar_y + bar_h),
                      (60, 180, 60), -1)

    # Thermal portion (blue, from right)
    if bar_w - rgb_bar_w > 0:
        cv2.rectangle(frame, (bar_x + rgb_bar_w, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 140, 60), -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 120), 1)

    # Center divider line (50% mark)
    center_x = bar_x + bar_w // 2
    cv2.line(frame, (center_x, bar_y - 3), (center_x, bar_y + bar_h + 3),
             (200, 200, 220), 1)

    # Dominant sensor indicator
    if rgb_pct > therm_pct + 0.05:
        winner = ">> RGB Leading"
        winner_color = (80, 220, 80)
    elif therm_pct > rgb_pct + 0.05:
        winner = "Thermal Leading <<"
        winner_color = (100, 180, 255)
    else:
        winner = "= Balanced ="
        winner_color = (200, 200, 200)

    (tw2, _), _ = cv2.getTextSize(winner, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, winner, (px + pw // 2 - tw2 // 2, py + ph - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, winner_color, 1, cv2.LINE_AA)

    return frame


def draw_header(frame, width, inference_ms):
    """Draw a professional header bar."""
    header_h = 55
    cv2.rectangle(frame, (0, 0), (width, header_h), (15, 15, 25), -1)
    cv2.line(frame, (0, header_h), (width, header_h), (60, 60, 80), 1)

    title = "Dual-Stream RGB+Thermal Fusion | Autonomous Driving Object Detection"
    cv2.putText(frame, title, (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 245), 2, cv2.LINE_AA)

    arch = f"YOLOv8-26s + CrossModalAttention | {inference_ms:.1f}ms"
    (tw, _), _ = cv2.getTextSize(arch, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, arch, (width - tw - 30, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 180), 1, cv2.LINE_AA)


def draw_panel_label(frame, text, x, y, color=(200, 200, 220)):
    """Draw a subtle label above a panel."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 12, y + 4), (25, 25, 35), -1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 12, y + 4), (60, 60, 80), 1)
    cv2.putText(frame, text, (x + 6, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def draw_footer(frame, width, height, frame_idx, total_frames, num_detections):
    """Draw a professional footer bar."""
    footer_h = 35
    fy = height - footer_h
    cv2.rectangle(frame, (0, fy), (width, height), (15, 15, 25), -1)
    cv2.line(frame, (0, fy), (width, fy), (60, 60, 80), 1)

    cv2.putText(frame, f"Frame {frame_idx + 1}/{total_frames}", (30, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 180), 1, cv2.LINE_AA)

    det_text = f"Detections: {num_detections} | Classes: Car | Person | Bicycle"
    (tw, _), _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, det_text, (width - tw - 30, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 160), 1, cv2.LINE_AA)


# ============================================================================
# Core Inference — Bypasses YOLO's predict() pipeline entirely
# ============================================================================

def run_inference(model_internal, img_6ch_letterboxed, device, conf_thresh, iou_thresh,
                  ratio, pad, orig_shape):
    """
    Run inference using the raw DetectionModel forward pass.
    Handles NMS and coordinate rescaling back to original image space.
    
    Returns: list of (cls_id, conf, (x1, y1, x2, y2)) in original image coordinates
    """
    # Convert HWC -> CHW, normalize, batch
    img = img_6ch_letterboxed.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # [6, H, W]
    img = torch.from_numpy(img).unsqueeze(0).to(device)  # [1, 6, H, W]

    with torch.no_grad():
        preds = model_internal(img)
    
    # preds is the raw output from the Detect head
    # Apply NMS using ultralytics' built-in function
    results = non_max_suppression(
        preds,
        conf_thres=conf_thresh,
        iou_thres=iou_thresh,
        max_det=100,
    )

    detections = []
    if results and len(results[0]) > 0:
        det = results[0]  # [N, 6] -> (x1, y1, x2, y2, conf, cls)
        
        # Scale boxes from letterboxed 640x640 back to original image
        boxes = det[:, :4].clone()
        boxes = scale_boxes_to_original(boxes, ratio, pad, orig_shape)
        
        for i in range(len(det)):
            x1, y1, x2, y2 = boxes[i].int().tolist()
            conf = float(det[i, 4])
            cls_id = int(det[i, 5])
            detections.append((cls_id, conf, (x1, y1, x2, y2)))

    return detections


# ============================================================================
# Core Video Generation
# ============================================================================

def create_detection_video(model, model_internal, attn_extractor, device,
                           thermal_dir, rgb_dir, output_path,
                           num_frames=600, fps=8, conf_thresh=0.35, iou_thresh=0.45):
    """Generate the full detection video with all panels."""

    # Gather sorted image pairs
    thermal_files = sorted(thermal_dir.glob('*.jpeg'))
    if not thermal_files:
        thermal_files = sorted(thermal_dir.glob('*.jpg')) + sorted(thermal_dir.glob('*.png'))

    if len(thermal_files) == 0:
        print("❌ No thermal images found!")
        return

    num_frames = min(num_frames, len(thermal_files))
    print(f"\n📹 Generating {num_frames}-frame video at {fps} FPS...")
    print(f"   Output: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))

    panel_w = VIDEO_WIDTH // 3
    panel_h = (VIDEO_HEIGHT - 90) // 2

    inference_times = []

    with tqdm(total=num_frames, desc="Rendering") as pbar:
        for idx in range(num_frames):
            thermal_path = thermal_files[idx]

            # Find matching RGB image
            rgb_path = rgb_dir / f"{thermal_path.stem}.jpg"
            if not rgb_path.exists():
                rgb_path = rgb_dir / f"{thermal_path.stem}.jpeg"
            if not rgb_path.exists():
                rgb_path = rgb_dir / f"{thermal_path.stem}.png"

            # Load thermal image
            thermal_img = cv2.imread(str(thermal_path))
            if thermal_img is None:
                continue

            # Handle grayscale thermal -> 3-channel BGR
            if len(thermal_img.shape) == 2:
                thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2BGR)

            # Load RGB image
            if rgb_path.exists():
                rgb_img = cv2.imread(str(rgb_path))
                if rgb_img is None:
                    rgb_img = thermal_img.copy()
            else:
                rgb_img = thermal_img.copy()

            img_h, img_w = thermal_img.shape[:2]
            orig_shape = (img_h, img_w)

            # Resize RGB to match thermal if needed
            if rgb_img.shape[:2] != (img_h, img_w):
                rgb_img = cv2.resize(rgb_img, (img_w, img_h))

            # Stack: [H, W, 6] = thermal(BGR) + RGB(BGR)
            # MUST MATCH DATALOADER IN base.py! (Thermal is channels 0-2, RGB is 3-5)
            stacked = np.concatenate((thermal_img, rgb_img), axis=-1)

            # Letterbox the 6-channel image (same as YOLO's internal preprocessing)
            stacked_lb, ratio, pad = letterbox(stacked, new_shape=(IMGSZ, IMGSZ))

            # Run inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            detections = run_inference(
                model_internal, stacked_lb, device,
                conf_thresh, iou_thresh, ratio, pad, orig_shape
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            inference_times.append(elapsed_ms)

            # Get attention weights captured by the hook
            attn_weights = attn_extractor.get_weights()

            # --- Draw all 6 panels ---

            # Panel 1: RGB with detections
            rgb_panel = draw_detections_from_list(rgb_img, detections)
            rgb_panel = cv2.resize(rgb_panel, (panel_w, panel_h))

            # Panel 2: Thermal with detections
            thermal_panel = draw_detections_from_list(thermal_img, detections)
            thermal_panel = cv2.resize(thermal_panel, (panel_w, panel_h))

            # Panel 3: Thermal Infrared colorized view with detections
            thermal_colored = cv2.applyColorMap(
                cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO
            )
            detection_panel = draw_detections_from_list(thermal_colored, detections)
            detection_panel = cv2.resize(detection_panel, (panel_w, panel_h))

            # Panel 4: RGB Attention Heatmap
            rgb_attn_heatmap = create_attention_heatmap(attn_weights, img_h, img_w, channel_idx=0)
            rgb_attn_blend = cv2.addWeighted(rgb_img, 0.4, rgb_attn_heatmap, 0.6, 0)
            rgb_attn_panel = cv2.resize(rgb_attn_blend, (panel_w, panel_h))

            # Panel 5: Thermal Attention Heatmap
            thermal_attn_heatmap = create_attention_heatmap(attn_weights, img_h, img_w, channel_idx=1)
            thermal_attn_blend = cv2.addWeighted(thermal_img, 0.4, thermal_attn_heatmap, 0.6, 0)
            thermal_attn_panel = cv2.resize(thermal_attn_blend, (panel_w, panel_h))

            # Panel 6: Adaptive Fusion Output
            if attn_weights:
                first_layer = list(attn_weights.keys())[0]
                w = attn_weights[first_layer]
                rgb_w = cv2.resize(w[0, 0].numpy(), (img_w, img_h))[:, :, np.newaxis]
                therm_w = cv2.resize(w[0, 1].numpy(), (img_w, img_h))[:, :, np.newaxis]
                fusion_img = (rgb_w * rgb_img.astype(np.float32) +
                              therm_w * thermal_img.astype(np.float32)).astype(np.uint8)
            else:
                fusion_img = cv2.addWeighted(rgb_img, 0.5, thermal_img, 0.5, 0)
            fusion_panel = draw_detections_from_list(fusion_img, detections)
            fusion_panel = cv2.resize(fusion_panel, (panel_w, panel_h))

            # --- Assemble the frame ---
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            frame[:] = (10, 10, 15)

            y_top = 55
            y_bot = y_top + panel_h + 5

            # Top row
            frame[y_top:y_top + panel_h, 0:panel_w] = rgb_panel
            frame[y_top:y_top + panel_h, panel_w:2*panel_w] = thermal_panel
            frame[y_top:y_top + panel_h, 2*panel_w:3*panel_w] = detection_panel

            # Bottom row
            frame[y_bot:y_bot + panel_h, 0:panel_w] = rgb_attn_panel
            frame[y_bot:y_bot + panel_h, panel_w:2*panel_w] = thermal_attn_panel
            frame[y_bot:y_bot + panel_h, 2*panel_w:3*panel_w] = fusion_panel

            # Labels
            draw_panel_label(frame, "RGB Camera", 10, y_top - 2, (100, 200, 100))
            draw_panel_label(frame, "Thermal Camera", panel_w + 10, y_top - 2, (100, 180, 255))
            draw_panel_label(frame, "Thermal Infrared", 2*panel_w + 10, y_top - 2, (255, 140, 50))
            draw_panel_label(frame, "RGB Attention Map", 10, y_bot - 2, (100, 200, 100))
            draw_panel_label(frame, "Thermal Attention Map", panel_w + 10, y_bot - 2, (100, 180, 255))
            draw_panel_label(frame, "Adaptive Fusion Output", 2*panel_w + 10, y_bot - 2, (200, 150, 255))

            # Header & Footer
            draw_header(frame, VIDEO_WIDTH, elapsed_ms)
            frame = draw_sensor_trust_meter(frame, attn_weights, rgb_img)
            draw_footer(frame, VIDEO_WIDTH, VIDEO_HEIGHT, idx, num_frames, len(detections))

            out.write(frame)
            pbar.set_postfix({'ms': f"{elapsed_ms:.1f}", 'det': len(detections)})
            pbar.update(1)

    out.release()
    print(f"\n✅ Video saved: {output_path}")

    if inference_times:
        mean_ms = np.mean(inference_times)
        total_s = sum(inference_times) / 1000
        inf_fps = len(inference_times) / total_s
        print(f"📊 Inference: {len(inference_times)} frames, "
              f"total {total_s:.1f}s, mean {mean_ms:.1f}ms/frame, {inf_fps:.1f} FPS")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")

    # Check checkpoint
    checkpoint_path = CHECKPOINT
    if not os.path.exists(checkpoint_path):
        alt = checkpoint_path.replace('best.pt', 'last.pt')
        if os.path.exists(alt):
            print(f"⚠️  best.pt not found, using last.pt")
            checkpoint_path = alt
        else:
            print(f"❌ Checkpoint not found at {checkpoint_path}")
            sys.exit(1)

    print(f"📦 Loading model: {checkpoint_path}")
    model = YOLO(checkpoint_path)

    # Get the internal DetectionModel (the actual PyTorch nn.Module)
    det_model = model.model
    det_model.eval()
    det_model.to(device)

    # Verify dual-stream
    if hasattr(det_model, 'dual_stream') and det_model.dual_stream:
        print("✅ Dual-Stream mode confirmed")
        print(f"   Fusion layers: {det_model.fusion_layers}")
    else:
        print("⚠️  Model does not appear to be dual-stream!")

    # Register attention hooks
    print("🔗 Registering attention hooks...")
    attn_extractor = AttentionExtractor(model)

    # Output
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / 'detection_demo_dual_yolo.mp4'

    # Generate video
    create_detection_video(
        model=model,
        model_internal=det_model,
        attn_extractor=attn_extractor,
        device=device,
        thermal_dir=THERMAL_DIR,
        rgb_dir=RGB_DIR,
        output_path=output_path,
        num_frames=NUM_FRAMES,
        fps=FPS,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH,
    )

    # Cleanup
    attn_extractor.cleanup()
    print("\n🎬 Done!")
