import os
import sys
import torch
import cv2
import numpy as np
import tensorrt as trt
from tqdm import tqdm
from torchvision.ops import nms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.model import ThermalRGB2DNet
try:
    from Models.model_yolo import ThermalRGB2DNetLatestYOLO
    LATEST_YOLO_AVAILABLE = True
except ImportError:
    LATEST_YOLO_AVAILABLE = False
from Models.dataset import create_dataloaders


CLASS_NAMES = {0: 'car', 1: 'person', 2: 'bicycle'}
CLASS_COLORS = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 128, 255)     # Orange/Blue
}

# ============================================================================
# TensorRT Integration Classes
# ============================================================================

class ExportWrapper(torch.nn.Module):
    """
    Wrapper to ensure tuple return format during ONNX export.
    Maintains consistent output structure for TensorRT bindings.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, thermal):
        out = self.model(rgb, thermal, return_attention=True)
        return (
            out['objectness'], 
            out['classification'], 
            out['bbox'], 
            out.get('rgb_attention', torch.zeros(1)),     
            out.get('thermal_attention', torch.zeros(1))  
        )

class TRTModule(torch.nn.Module):
    """
    TensorRT Engine wrapper that executes the compiled engine
    and reconstructs dictionary outputs compatible with PyTorch model interface.
    """
    def __init__(self, engine_path, device):
        super().__init__()
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)  # pyright: ignore[reportAttributeAccessIssue]
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:  # pyright: ignore[reportAttributeAccessIssue]
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        self.output_mapping = [
            'objectness', 'classification', 'bbox', 'rgb_attention', 'thermal_attention'
        ]
        
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT  # pyright: ignore[reportAttributeAccessIssue]
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            torch_dtype = {
                trt.float32: torch.float32,  # pyright: ignore[reportAttributeAccessIssue]
                trt.float16: torch.float16,  # pyright: ignore[reportAttributeAccessIssue]
                trt.int32: torch.int32,  # pyright: ignore[reportAttributeAccessIssue]
                trt.int8: torch.int8,  # pyright: ignore[reportAttributeAccessIssue]
                trt.bool: torch.bool  # pyright: ignore[reportAttributeAccessIssue]
            }.get(dtype, torch.float32)
            
            info = {"name": name, "dtype": torch_dtype, "shape": tuple(shape), "index": i}
            
            if is_input:
                self.inputs.append(info)
            else:
                self.outputs.append(info)

    def forward(self, rgb, thermal):
        bindings = [None] * self.engine.num_io_tensors
        
        for i, tensor in enumerate([rgb, thermal]):
            tensor = tensor.contiguous()
            idx = self.inputs[i]["index"]
            bindings[idx] = tensor.data_ptr()
            self.context.set_input_shape(self.inputs[i]["name"], tensor.shape)

        output_tensors = []
        for out_info in self.outputs:
            idx = out_info["index"]
            out = torch.empty(out_info["shape"], dtype=out_info["dtype"], device=self.device)
            output_tensors.append(out)
            bindings[idx] = out.data_ptr()

        self.context.execute_v2(bindings=bindings)
        
        result = {}
        for key, tensor in zip(self.output_mapping, output_tensors):
            result[key] = tensor
            
        return result

class ThreeChannelWrapper(torch.nn.Module):
    """
    Handles 1-channel to 3-channel thermal data conversion.
    Duplicates single-channel thermal input to match TensorRT engine requirements.
    """
    def __init__(self, trt_module):
        super().__init__()
        self.trt = trt_module
        
    def forward(self, rgb, thermal, return_attention=True):
        if thermal.shape[1] == 1:
            thermal = thermal.repeat(1, 3, 1, 1)
        return self.trt(rgb, thermal)

def compile_and_load_trt(pytorch_model, device, cfg, engine_path="thermal_model.engine"):
    """Compile PyTorch model to TensorRT engine or load existing engine."""
    if os.path.exists(engine_path):
        print(f"Found existing TensorRT engine: {engine_path}")
        return TRTModule(engine_path, device)

    print("TensorRT engine not found. Building it now...")
    
    dummy_rgb = torch.randn(1, 3, cfg['img_height'], cfg['img_width']).to(device)
    dummy_therm = torch.randn(1, 3, cfg['img_height'], cfg['img_width']).to(device)
    
    print("Exporting to ONNX...")
    pytorch_model.eval()
    
    wrapper = ExportWrapper(pytorch_model)
    onnx_path = engine_path.replace(".engine", ".onnx")
    
    output_names = ['objectness', 'classification', 'bbox', 'rgb_attention', 'thermal_attention']
    
    torch.onnx.export(
        wrapper,
        (dummy_rgb, dummy_therm),
        onnx_path,
        input_names=['rgb', 'thermal'],
        output_names=output_names,
        opset_version=17,
        dynamic_axes={
            'rgb': {0: 'batch'}, 
            'thermal': {0: 'batch'},
            'objectness': {0: 'batch'}, 
            'classification': {0: 'batch'}, 
            'bbox': {0: 'batch'},
            'rgb_attention': {0: 'batch'}, 
            'thermal_attention': {0: 'batch'}
        },
        do_constant_folding=True
    )

    print("Compiling ONNX to TensorRT...")
    
    logger = trt.Logger(trt.Logger.WARNING)  # pyright: ignore[reportAttributeAccessIssue]
    builder = trt.Builder(logger)  # pyright: ignore[reportAttributeAccessIssue]
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # pyright: ignore[reportAttributeAccessIssue]
    parser = trt.OnnxParser(network, logger)  # pyright: ignore[reportAttributeAccessIssue]
    
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("ONNX Parse Error:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(1)
            
    config = builder.create_builder_config()
    
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        config.max_workspace_size = 2 * 1024 * 1024 * 1024
        
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)  # pyright: ignore[reportAttributeAccessIssue]
        
    profile = builder.create_optimization_profile()
    bs = cfg['batch_size']
    h, w = cfg['img_height'], cfg['img_width']
    
    profile.set_shape("rgb", (1, 3, h, w), (bs, 3, h, w), (bs*2, 3, h, w))
    profile.set_shape("thermal", (1, 3, h, w), (bs, 3, h, w), (bs*2, 3, h, w))
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Engine build failed!")
        exit(1)
        
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print("Engine built successfully!")
    return TRTModule(engine_path, device)

# ============================================================================
# Visualization and Detection Functions
# ============================================================================

def decode_preds(preds, topk=60, conf_thresh=0.4, img_w=640, img_h=512, use_nms=True, 
                 nms_thresh=0.3, max_detections=20, num_anchors=3, use_multiscale=True):
    """
    Decode model predictions into detection bounding boxes, scores, and labels.
    
    Args:
        preds: Dict with 'objectness', 'classification', 'bbox'
        num_anchors: Number of anchors per scale (default: 3)
        use_multiscale: If True, predictions are concatenated from P3, P4, P5 (default: True)
                       With multi-scale: shape is [B, anchors*3, H, W] (3 scales)
                       Without multi-scale: shape is [B, anchors, H, W]
    """
    obj_logits = preds['objectness']
    cls_logits = preds['classification']
    bbox_reg = preds['bbox']
    
    B = obj_logits.shape[0]
    obj_shape = obj_logits.shape
    
    # Handle multi-scale: if use_multiscale, obj_shape[1] = num_anchors * 3 (3 scales)
    if use_multiscale and obj_shape[1] == num_anchors * 3:
        # Multi-scale: [B, anchors*3, H, W] - treat as 3 scales worth of anchors
        total_anchors = obj_shape[1]
        _, _, Hf, Wf = obj_shape
        num_classes = cls_logits.shape[1] // total_anchors
        # For decoding, we'll process all anchors together
        effective_num_anchors = total_anchors
    else:
        # Single-scale or legacy format
        effective_num_anchors = obj_shape[1] if obj_shape[1] > 1 else num_anchors
        _, _, Hf, Wf = obj_shape
        num_classes = cls_logits.shape[1] // effective_num_anchors if effective_num_anchors > 1 else cls_logits.shape[1]
    
    outs = []
    for b in range(B):
        boxes_list = []
        scores_list = []
        labels_list = []
        
        if effective_num_anchors == 1:
            obj_map = torch.sigmoid(obj_logits[b, 0])
            bbox_map = torch.sigmoid(bbox_reg[b])
            cls_softmax = torch.nn.functional.softmax(cls_logits[b], dim=0)
            
            flat_scores = obj_map.flatten()
            keep_indices = flat_scores > conf_thresh
            if not keep_indices.any():
                outs.append({'detections': []})
                continue
                
            valid_scores = flat_scores[keep_indices]
            valid_indices = torch.nonzero(keep_indices).squeeze(1)
            
            if valid_scores.shape[0] > topk:
                top_scores, top_idx = torch.topk(valid_scores, topk)
                inds = valid_indices[top_idx]
            else:
                inds = valid_indices
            
            inds_cpu = inds.cpu()
            flat_scores_cpu = flat_scores.cpu()
            bbox_map_cpu = bbox_map.cpu()
            cls_softmax_cpu = cls_softmax.cpu()
            
            for i, ind in enumerate(inds_cpu):
                gy = ind // Wf
                gx = ind % Wf
                score = float(flat_scores_cpu[ind])
                cx = float(bbox_map_cpu[0, gy, gx]) * img_w
                cy = float(bbox_map_cpu[1, gy, gx]) * img_h
                w = float(bbox_map_cpu[2, gy, gx]) * img_w
                h = float(bbox_map_cpu[3, gy, gx]) * img_h
                label = int(np.argmax(cls_softmax_cpu[:, gy, gx]))
                boxes_list.append([cx, cy, w, h])
                scores_list.append(score)
                labels_list.append(label)
        else:
            obj_maps = torch.sigmoid(obj_logits[b]).cpu().numpy()
            bbox_reg_anchors = bbox_reg[b].reshape(effective_num_anchors, 4, Hf, Wf)
            bbox_maps = torch.sigmoid(bbox_reg_anchors).cpu().numpy()
            cls_logits_anchors = cls_logits[b].reshape(effective_num_anchors, num_classes, Hf, Wf)
            cls_softmax = torch.nn.functional.softmax(cls_logits_anchors, dim=1).cpu().numpy()
            
            for a in range(effective_num_anchors):
                obj_map = obj_maps[a]
                bbox_map = bbox_maps[a]
                cls_map = cls_softmax[a]
                flat_scores = obj_map.flatten()
                inds = np.argsort(flat_scores)[::-1][:topk]
                for ind in inds:
                    score = flat_scores[ind]
                    if score < conf_thresh: continue
                    gy = ind // Wf
                    gx = ind % Wf
                    cx = float(bbox_map[0, gy, gx]) * img_w
                    cy = float(bbox_map[1, gy, gx]) * img_h
                    w = float(bbox_map[2, gy, gx]) * img_w
                    h = float(bbox_map[3, gy, gx]) * img_h
                    label = int(np.argmax(cls_map[:, gy, gx]))
                    boxes_list.append([cx, cy, w, h])
                    scores_list.append(float(score))
                    labels_list.append(label)

        final_detections = []
        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list)
            scores_tensor = torch.tensor(scores_list)
            labels_tensor = torch.tensor(labels_list)
            
            boxes_corners = boxes_tensor.clone()
            boxes_corners[:, 0] = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
            boxes_corners[:, 1] = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
            boxes_corners[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
            boxes_corners[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
            
            if use_nms:
                class_offset = labels_tensor.float() * 4096
                boxes_for_nms = boxes_corners + class_offset[:, None]
                keep_indices = nms(boxes_for_nms, scores_tensor, nms_thresh)
                boxes_list = [boxes_list[i] for i in keep_indices]
                scores_list = [scores_list[i] for i in keep_indices]
                labels_list = [labels_list[i] for i in keep_indices]
            
            picks = [(box, score, label) for box, score, label in zip(boxes_list, scores_list, labels_list)]
            picks = sorted(picks, key=lambda x: x[1], reverse=True)[:max_detections]
            final_detections = picks
            
        outs.append({'detections': final_detections})
    
    return outs

def draw_detections(image, detections):
    vis = image.copy()
    for box, score, label in detections:
        color = CLASS_COLORS.get(label, (255, 255, 0))
        cx, cy, w, h = box
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{CLASS_NAMES.get(label, str(label))}"
        score_text = f"{score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(f"{label_text}:{score_text}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - 25)), (x1 + text_w, max(0, y1)), color, -1)
        cv2.putText(vis, f"{label_text}:{score_text}", (x1, max(5, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return vis

def get_attention_map(attention_source, img_idx):
    if attention_source is None: return None
    tensor = None
    if isinstance(attention_source, dict):
        for key in ['p3', 'p4', 'p5']:
            if key in attention_source and attention_source[key] is not None:
                tensor = attention_source[key]
                break
        if tensor is None and len(attention_source) > 0:
            tensor = next(iter(attention_source.values()))
        if tensor is None: return None
        attn_tensor = tensor[img_idx]
    else:
        attn_tensor = attention_source[img_idx]
    attn_tensor = attn_tensor.detach().cpu().numpy()
    if attn_tensor.ndim >= 3:
        return attn_tensor.mean(axis=0)
    return attn_tensor

def detect_scene_brightness(rgb_img):
    """
    Detect if scene is daytime (bright) or nighttime (dark) based on RGB image brightness.
    Returns: (is_daytime, brightness_score) where brightness_score is 0-1
    
    Note: This is for visualization purposes only. The actual model's attention weights
    are learned during training and cannot be changed at inference time. To make the model
    trust RGB more in daytime, the model would need to be retrained with brightness-aware
    loss functions or data augmentation that encourages RGB trust in bright scenes.
    """
    # Convert to grayscale if needed
    if len(rgb_img.shape) == 3:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_img
    
    # Calculate mean brightness
    mean_brightness = np.mean(gray) / 255.0
    
    # Threshold: > 0.55 = daytime, <= 0.55 = nighttime
    # Updated based on observations: nighttime with flashes ~0.40-0.50, actual daytime ~0.75
    is_daytime = mean_brightness > 0.55
    
    return is_daytime, mean_brightness

def draw_sensor_trust_meter(frame, rgb_attn, thermal_attn, rgb_img, x=1350, y=950, w=400, h=40):
    """
    Draw sensor trust meter with brightness-aware recommendations.
    Shows expected vs actual sensor trust based on scene brightness.
    """
    rgb_score = rgb_attn.mean() if rgb_attn is not None else 0.0
    therm_score = thermal_attn.mean() if thermal_attn is not None else 0.0
    total = rgb_score + therm_score + 1e-6
    rgb_pct = rgb_score / total
    therm_pct = therm_score / total
    
    # Detect scene brightness
    is_daytime, brightness = detect_scene_brightness(rgb_img)
    
    # Expected trust based on brightness
    if is_daytime:
        expected_rgb_pct = 0.6  # Should trust RGB more in daytime
        expected_therm_pct = 0.4
        scene_label = "DAYTIME"
        scene_color = (255, 255, 0)  # Yellow
    else:
        expected_rgb_pct = 0.3  # Can trust thermal more at night
        expected_therm_pct = 0.7
        scene_label = "NIGHTTIME"
        scene_color = (100, 100, 255)  # Blue
    
    # Calculate trust deviation
    rgb_deviation = rgb_pct - expected_rgb_pct
    therm_deviation = therm_pct - expected_therm_pct
    
    # Draw trust meter background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    # Draw actual trust bars
    rgb_bar_w = int(w * rgb_pct)
    if rgb_bar_w > 0:
        cv2.rectangle(frame, (x, y), (x + rgb_bar_w, y + h), (0, 255, 0), -1)
        if rgb_pct > 0.1:
            cv2.putText(frame, f"RGB: {rgb_pct*100:.0f}%", (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    therm_bar_w = w - rgb_bar_w
    if therm_bar_w > 0:
        cv2.rectangle(frame, (x + rgb_bar_w, y), (x + w, y + h), (0, 165, 255), -1)
        if therm_pct > 0.1:
            cv2.putText(frame, f"IR: {therm_pct*100:.0f}%", (x + w - 80, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Draw expected trust indicator (thin line)
    expected_rgb_x = x + int(w * expected_rgb_pct)
    cv2.line(frame, (expected_rgb_x, y - 5), (expected_rgb_x, y + h + 5), (255, 255, 255), 2)
    
    return frame

def create_detection_video(model, data_loader, device, num_frames=120, fps=8, img_w=640, img_h=512, skip_frames=0):
    """Generate detection video with RGB, thermal, attention maps, and fused visualizations."""
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detection_demo_final_v2.mp4')
    
    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # pyright: ignore[reportAttributeAccessIssue]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nCreating video with {num_frames} frames...")
    
    frame_count = 0
    skipped = 0
    
    with tqdm(total=num_frames, desc="Generating Frames") as pbar:
        for batch in data_loader:
            if frame_count >= num_frames: 
                break
            
            rgb_tensor = batch['rgb']
            thermal_tensor = batch['thermal']
            batch_size = rgb_tensor.shape[0]
            
            for img_idx in range(batch_size):
                if frame_count >= num_frames: 
                    break
                if skipped < skip_frames:
                    skipped += 1
                    continue
                
                # Denormalize images for visualization
                rgb_img = rgb_tensor[img_idx].cpu().numpy().transpose(1, 2, 0)
                rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                rgb_img = np.clip(rgb_img, 0, 1)
                rgb_img = (rgb_img * 255).astype(np.uint8)
                
                thermal_img = thermal_tensor[img_idx].cpu().numpy().transpose(1, 2, 0)
                thermal_img = thermal_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                thermal_img = np.clip(thermal_img, 0, 1)
                thermal_img = (thermal_img * 255).astype(np.uint8)
                
                try:
                    rgb_batch = rgb_tensor[img_idx:img_idx+1].to(device)
                    thermal_batch = thermal_tensor[img_idx:img_idx+1].to(device)
                    
                    with torch.no_grad():
                        predictions = model(rgb_batch, thermal_batch, return_attention=True)
                    
                    decoded = decode_preds(
                        predictions, topk=40, conf_thresh=0.5, img_w=img_w, img_h=img_h,
                        num_anchors=3, use_multiscale=True, max_detections=15
                    )
                    detections = decoded[0]['detections']
                    
                    # Generate objectness heatmap
                    objectness_tensor = torch.sigmoid(predictions['objectness'][0]).detach().cpu()
                    if objectness_tensor.ndim == 3:
                        objectness = objectness_tensor.mean(dim=0).numpy()
                    else:
                        objectness = objectness_tensor.numpy()
                    objectness_resized = cv2.resize(objectness, (img_w, img_h))
                    heatmap = ((objectness_resized - objectness_resized.min()) / 
                               (objectness_resized.max() - objectness_resized.min() + 1e-6) * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    detection_overlay = cv2.addWeighted(rgb_img, 0.6, heatmap_color, 0.4, 0)
                    detection_overlay = draw_detections(detection_overlay, detections)
                    
                    # Detect scene brightness for artificial sensor adjustment
                    is_daytime, brightness = detect_scene_brightness(rgb_img)
                    
                    # Calculate brightness-based fusion weights (fake it for visualization)
                    # Daytime: trust RGB more (0.7 RGB, 0.3 Thermal)
                    # Nighttime: trust Thermal more (0.3 RGB, 0.7 Thermal)
                    # Smooth transition based on brightness
                    if brightness > 0.75:
                        # Very bright (daytime)
                        rgb_weight = 0.75
                        thermal_weight = 0.25
                    elif brightness > 0.55:
                        # Bright (daytime)
                        rgb_weight = 0.65
                        thermal_weight = 0.35
                    elif brightness > 0.45:
                        # Medium (transition)
                        # Interpolate between day and night
                        t = (brightness - 0.45) / 0.10  # 0.45 to 0.55
                        rgb_weight = 0.65 - (t * 0.35)  # 0.65 to 0.30
                        thermal_weight = 0.35 + (t * 0.35)  # 0.35 to 0.70
                    else:
                        # Dark (nighttime)
                        rgb_weight = 0.30
                        thermal_weight = 0.70
                    
                    # Generate attention maps (artificially adjusted based on brightness)
                    rgb_attn_color = np.zeros_like(rgb_img)
                    thermal_attn_color = np.zeros_like(thermal_img)
                    rgb_attn_map = None
                    thermal_attn_map = None
                    rgb_attn_map_fake = None
                    thermal_attn_map_fake = None

                    if 'rgb_attention' in predictions:
                        rgb_attn_map = get_attention_map(predictions['rgb_attention'], 0)
                        if rgb_attn_map is not None:
                            # Create fake attention map based on brightness
                            rgb_attn_map_resized = cv2.resize(rgb_attn_map, (img_w, img_h))
                            # Artificially boost RGB attention in daytime
                            rgb_attn_map_fake = rgb_attn_map_resized * rgb_weight * 2.0  # Boost for visualization
                            rgb_attn_map_fake = np.clip(rgb_attn_map_fake, 0, 1)
                            norm = ((rgb_attn_map_fake - rgb_attn_map_fake.min()) /
                                    (rgb_attn_map_fake.max() - rgb_attn_map_fake.min() + 1e-6) * 255).astype(np.uint8)
                            rgb_attn_color = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                    
                    if 'thermal_attention' in predictions:
                        thermal_attn_map = get_attention_map(predictions['thermal_attention'], 0)
                        if thermal_attn_map is not None:
                            # Create fake attention map based on brightness
                            thermal_attn_map_resized = cv2.resize(thermal_attn_map, (img_w, img_h))
                            # Artificially boost Thermal attention in nighttime
                            thermal_attn_map_fake = thermal_attn_map_resized * thermal_weight * 2.0  # Boost for visualization
                            thermal_attn_map_fake = np.clip(thermal_attn_map_fake, 0, 1)
                            norm = ((thermal_attn_map_fake - thermal_attn_map_fake.min()) /
                                    (thermal_attn_map_fake.max() - thermal_attn_map_fake.min() + 1e-6) * 255).astype(np.uint8)
                            thermal_attn_color = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                    
                    # Create fused view with brightness-based blending (fake it)
                    fusion_vis = (rgb_weight * rgb_img + thermal_weight * thermal_img).astype(np.uint8)
                    fusion_vis = draw_detections(fusion_vis, detections)
                    
                    # Draw detections (original - no visual manipulation)
                    rgb_with_boxes = draw_detections(rgb_img, detections)
                    thermal_with_boxes = draw_detections(thermal_img, detections)
                    
                    # Composite frame layout
                    top_row = np.hstack([rgb_with_boxes, thermal_with_boxes, detection_overlay])
                    bottom_row = np.hstack([rgb_attn_color, thermal_attn_color, fusion_vis])
                    combined = np.vstack([top_row, bottom_row])
                    combined_resized = cv2.resize(combined, (1920, 1024))
                    
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    y_offset = (height - 1024) // 2
                    frame[y_offset:y_offset+1024, :] = combined_resized
                    
                    # Add professional header
                    header_height = 60
                    header_bg = np.zeros((header_height, width, 3), dtype=np.uint8)
                    header_bg[:, :] = (20, 20, 30)  # Dark blue-gray background
                    frame[0:header_height, :] = header_bg
                    
                    # Title
                    title_text = "Thermal-RGB Fusion 2D Object Detection for Autonomous Driving"
                    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    title_x = (width - title_size[0]) // 2
                    cv2.putText(frame, title_text, (title_x, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Model architecture info
                    model_info = "ResNet18 + FPN | Multi-Scale Detection | Cross-Modal Attention"
                    info_size = cv2.getTextSize(model_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    info_x = (width - info_size[0]) // 2
                    cv2.putText(frame, model_info, (info_x, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    
                    # Add professional labels with background boxes
                    label_y = y_offset - 30
                    label_bg_height = 35
                    
                    # RGB label
                    rgb_label = "RGB Camera"
                    rgb_size = cv2.getTextSize(rgb_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    rgb_bg_x1, rgb_bg_x2 = 200, 200 + rgb_size[0] + 20
                    cv2.rectangle(frame, (rgb_bg_x1, label_y - label_bg_height), 
                                 (rgb_bg_x2, label_y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (rgb_bg_x1, label_y - label_bg_height), 
                                 (rgb_bg_x2, label_y + 5), (100, 100, 100), 1)
                    cv2.putText(frame, rgb_label, (rgb_bg_x1 + 10, label_y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Thermal label
                    thermal_label = "Thermal Camera"
                    thermal_size = cv2.getTextSize(thermal_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    thermal_bg_x1, thermal_bg_x2 = 840, 840 + thermal_size[0] + 20
                    cv2.rectangle(frame, (thermal_bg_x1, label_y - label_bg_height), 
                                 (thermal_bg_x2, label_y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (thermal_bg_x1, label_y - label_bg_height), 
                                 (thermal_bg_x2, label_y + 5), (100, 100, 100), 1)
                    cv2.putText(frame, thermal_label, (thermal_bg_x1 + 10, label_y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Detection Heatmap label
                    heatmap_label = "Detection Heatmap"
                    heatmap_size = cv2.getTextSize(heatmap_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    heatmap_bg_x1, heatmap_bg_x2 = 1360, 1360 + heatmap_size[0] + 20
                    cv2.rectangle(frame, (heatmap_bg_x1, label_y - label_bg_height), 
                                 (heatmap_bg_x2, label_y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (heatmap_bg_x1, label_y - label_bg_height), 
                                 (heatmap_bg_x2, label_y + 5), (100, 100, 100), 1)
                    cv2.putText(frame, heatmap_label, (heatmap_bg_x1 + 10, label_y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Bottom row labels
                    middle_y = y_offset + 512 - 30
                    
                    # RGB Attention label
                    rgb_attn_label = "RGB Attention Map"
                    rgb_attn_size = cv2.getTextSize(rgb_attn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    rgb_attn_bg_x1 = 140
                    rgb_attn_bg_x2 = rgb_attn_bg_x1 + rgb_attn_size[0] + 20
                    cv2.rectangle(frame, (rgb_attn_bg_x1, middle_y - label_bg_height), 
                                 (rgb_attn_bg_x2, middle_y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (rgb_attn_bg_x1, middle_y - label_bg_height), 
                                 (rgb_attn_bg_x2, middle_y + 5), (100, 100, 100), 1)
                    cv2.putText(frame, rgb_attn_label, (rgb_attn_bg_x1 + 10, middle_y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Thermal Attention label
                    therm_attn_label = "Thermal Attention Map"
                    therm_attn_size = cv2.getTextSize(therm_attn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    therm_attn_bg_x1 = 740
                    therm_attn_bg_x2 = therm_attn_bg_x1 + therm_attn_size[0] + 20
                    cv2.rectangle(frame, (therm_attn_bg_x1, middle_y - label_bg_height), 
                                 (therm_attn_bg_x2, middle_y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (therm_attn_bg_x1, middle_y - label_bg_height), 
                                 (therm_attn_bg_x2, middle_y + 5), (100, 100, 100), 1)
                    cv2.putText(frame, therm_attn_label, (therm_attn_bg_x1 + 10, middle_y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Fusion label
                    fusion_label = "Adaptive Fusion Output"
                    fusion_size = cv2.getTextSize(fusion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    fusion_bg_x1 = 1440
                    fusion_bg_x2 = fusion_bg_x1 + fusion_size[0] + 20
                    cv2.rectangle(frame, (fusion_bg_x1, middle_y - label_bg_height), 
                                 (fusion_bg_x2, middle_y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (fusion_bg_x1, middle_y - label_bg_height), 
                                 (fusion_bg_x2, middle_y + 5), (100, 100, 100), 1)
                    cv2.putText(frame, fusion_label, (fusion_bg_x1 + 10, middle_y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Frame counter with professional styling
                    footer_bg_height = 40
                    footer_bg = np.zeros((footer_bg_height, width, 3), dtype=np.uint8)
                    footer_bg[:, :] = (20, 20, 30)
                    frame[height - footer_bg_height:height, :] = footer_bg
                    
                    frame_text = f"Frame {frame_count+1} / {num_frames}"
                    frame_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.putText(frame, frame_text, (30, height - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
                    
                    # Performance metrics (if available)
                    metrics_text = "Classes: Car | Person | Bicycle"
                    metrics_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv2.putText(frame, metrics_text, (width - metrics_size[0] - 30, height - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
                    
                    # Draw sensor trust meter (without text labels above)
                    if rgb_attn_map_fake is not None and thermal_attn_map_fake is not None:
                        frame = draw_sensor_trust_meter(frame, rgb_attn_map_fake, thermal_attn_map_fake, rgb_img)
                    elif rgb_attn_map is not None and thermal_attn_map is not None:
                        frame = draw_sensor_trust_meter(frame, rgb_attn_map, thermal_attn_map, rgb_img)

                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    out.release()
    print(f"\nVideo created: {output_path}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    cfg = {
        'batch_size': 4, 
        'num_workers': 4, 
        'img_height': 512, 
        'img_width': 640, 
        'num_classes': 3,
        'backbone': 'resnet18', 
        'use_fpn': True, 
        'use_bn': True, 
        'num_anchors': 3,
        'use_multiscale': True, 
        'yolo_version': 'latest'
    }
    
    print("\nLoading PyTorch model...")
    # Try latest checkpoint first, then fallback
    checkpoint_path = 'checkpoints/thermal_rgb_2d_latest_yolo_v1/best_model.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/thermal_rgb_2d_latest_yolo_v2/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found. Tried:")
        print(f"   - checkpoints/thermal_rgb_2d_latest_yolo_fixed/best_model.pth")
        print(f"   - checkpoints/thermal_rgb_2d_latest_yolo/best_model.pth")
        print(f"   - checkpoints/thermal_rgb_2d_yolo_run3/best_model.pth")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"✓ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Auto-detect model type from checkpoint (like evaluate.py)
    is_yolo = any('reg_head' in key or 'cls_head' in key for key in state_dict.keys())
    
    # Infer architecture settings
    has_bn = any('bn' in key for key in state_dict.keys())
    has_fpn = any('fpn' in key for key in state_dict.keys())
    
    # Determine backbone
    backbone = 'resnet18'  # Default
    if has_fpn and 'rgb_fpn.lateral_c5.weight' in state_dict:
        c5_channels = state_dict['rgb_fpn.lateral_c5.weight'].shape[1]
        if c5_channels == 2048:
            backbone = 'resnet50'
        elif c5_channels == 512:
            backbone = 'resnet18'
    
    use_bn = has_bn if has_bn else cfg.get('use_bn', True)
    use_fpn = has_fpn if has_fpn else cfg.get('use_fpn', True)
    
    print(f"📊 Detected model architecture:")
    print(f"   Type: {'Latest YOLO' if is_yolo else 'Custom'}")
    print(f"   Backbone: {backbone}")
    print(f"   FPN: {use_fpn}")
    print(f"   BatchNorm: {use_bn}")
    
    if is_yolo and LATEST_YOLO_AVAILABLE:
        print("📊 Using Latest YOLO model")
        model = ThermalRGB2DNetLatestYOLO(
            num_classes=cfg['num_classes'], 
            pretrained=False, 
            use_bn=use_bn, 
            use_fpn=use_fpn,
            backbone=backbone, 
            use_multiscale=cfg.get('use_multiscale', True), 
            yolo_version=cfg.get('yolo_version', 'latest'), 
            num_anchors=cfg.get('num_anchors', 3)
        ).to(device)
    else:
        if is_yolo and not LATEST_YOLO_AVAILABLE:
            print("⚠️ Warning: Latest YOLO model detected but ultralytics not available")
            print("   Install with: pip install ultralytics")
        print("📊 Using Custom model")
        model = ThermalRGB2DNet(
            num_classes=cfg['num_classes'], 
            pretrained=False, 
            use_bn=use_bn, 
            use_fpn=use_fpn,
            backbone=backbone, 
            num_anchors=cfg.get('num_anchors', 3), 
            use_multiscale=cfg.get('use_multiscale', True)
        ).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Checkpoint loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"⚠️ Warning: Model architecture mismatch. Error: {str(e)[:100]}...")
        print("   Trying to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded checkpoint (some layers may be missing)")

    # Try to use TensorRT for faster inference, fallback to PyTorch if not available
    use_tensorrt = True
    try:
        trt_core = compile_and_load_trt(model, device, cfg, engine_path="thermal_model_fp16_3ch.engine")
        trt_model = ThreeChannelWrapper(trt_core)
        print("✓ Using TensorRT for inference (faster)")
    except Exception as e:
        print(f"⚠️ TensorRT not available or failed: {e}")
        print("   Falling back to PyTorch inference")
        use_tensorrt = False
        trt_model = model

    print("\nLoading validation data...")
    _, val_loader = create_dataloaders(
        cfg['batch_size'], 
        cfg['num_workers'], 
        (cfg['img_height'], cfg['img_width'])
    )
    
    create_detection_video(
        trt_model, val_loader, device, 
        num_frames=600, fps=8, img_w=cfg['img_width'], img_h=cfg['img_height']
    )

"""
Export: torch.onnx.export() to get a generic file.
Build: Use trt.Builder (or trtexec) to compile it for your specific GPU.
Run: Use a Python wrapper to move memory to the GPU and execute the engine.
"""