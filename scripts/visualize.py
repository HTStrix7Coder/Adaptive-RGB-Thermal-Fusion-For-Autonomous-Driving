import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from tqdm import tqdm
from torchvision.ops import nms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.model import ThermalRGB2DNet
try:
    from Models.model_yolo import ThermalRGB2DNetLatestYOLO
    LATEST_YOLO_AVAILABLE = True
except ImportError:
    LATEST_YOLO_AVAILABLE = False
    print("⚠️ Warning: Latest YOLO model not available. Install ultralytics: pip install ultralytics")
from Models.dataset import create_dataloaders, get_transforms
from utils.annotations import get_annotations_for_split

def decode_preds(preds, topk=100, conf_thresh=0.4, img_w=640, img_h=512, use_nms=True, 
                 nms_thresh=0.3, max_detections=50):
    """
    Convert network outputs to list of detections per image.
    FIXED: Uses torchvision.ops.nms for aggressive cleanup.
    """
    obj_logits = preds['objectness']
    cls_logits = preds['classification']
    bbox_reg = preds['bbox']

    B = obj_logits.shape[0]
    obj_shape = obj_logits.shape

    num_anchors = obj_shape[1]
    _, _, Hf, Wf = obj_shape

    if num_anchors == 1:
        num_classes = cls_logits.shape[1]
    else:
        num_classes = cls_logits.shape[1] // num_anchors

    outs = []
    
    # Process each image in batch
    for b in range(B):
        boxes_list = []
        scores_list = []
        labels_list = []

        # --- EXTRACT PREDICTIONS ---
        if num_anchors == 1:
            obj_map = torch.sigmoid(obj_logits[b, 0])
            bbox_map = torch.sigmoid(bbox_reg[b])
            cls_softmax = torch.nn.functional.softmax(cls_logits[b], dim=0)
            
            # Flatten to find top candidates
            flat_scores = obj_map.flatten()
            
            # Filter low confidence immediately to save compute
            keep_indices = flat_scores > conf_thresh
            if not keep_indices.any():
                outs.append({'detections': []})
                continue
                
            # Get indices of good scores
            # We use topk only on valid scores to keep it fast
            valid_scores = flat_scores[keep_indices]
            valid_indices = torch.nonzero(keep_indices).squeeze(1)
            
            # If too many, take topk
            if valid_scores.shape[0] > topk:
                top_scores, top_idx = torch.topk(valid_scores, topk)
                inds = valid_indices[top_idx]
            else:
                inds = valid_indices

            # Convert indices to grid coordinates
            inds = inds.cpu() # Move to CPU for loop processing if needed, or keep on GPU
            
            for ind in inds:
                gy = ind // Wf
                gx = ind % Wf
                
                score = float(flat_scores[ind])
                
                # Get Box (Normalized)
                cx_n = float(bbox_map[0, gy, gx])
                cy_n = float(bbox_map[1, gy, gx])
                w_n = float(bbox_map[2, gy, gx])
                h_n = float(bbox_map[3, gy, gx])
                
                # Convert to Pixels [cx, cy, w, h]
                cx = cx_n * img_w
                cy = cy_n * img_h
                w = w_n * img_w
                h = h_n * img_h
                
                label = int(torch.argmax(cls_softmax[:, gy, gx]))
                
                boxes_list.append([cx, cy, w, h])
                scores_list.append(score)
                labels_list.append(label)
                
        else:
            # Multi-anchor logic (keeping numpy implementation for simplicity in loop, but NMS will be Torch)
            obj_maps = torch.sigmoid(obj_logits[b]).cpu().numpy()
            bbox_reg_anchors = bbox_reg[b].reshape(num_anchors, 4, Hf, Wf)
            bbox_maps = torch.sigmoid(bbox_reg_anchors).cpu().numpy()
            cls_logits_anchors = cls_logits[b].reshape(num_anchors, num_classes, Hf, Wf)
            cls_softmax = torch.nn.functional.softmax(cls_logits_anchors, dim=1).cpu().numpy()

            for a in range(num_anchors):
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
                    scores_list.append(score)
                    labels_list.append(label)

        # --- AGGRESSIVE CLEANUP (NMS) ---
        final_detections = []
        if len(boxes_list) > 0:
            # Convert to Tensor for PyTorch NMS
            boxes_tensor = torch.tensor(boxes_list)
            scores_tensor = torch.tensor(scores_list)
            labels_tensor = torch.tensor(labels_list)
            
            # Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
            # NMS requires corners, not center-width
            boxes_corners = boxes_tensor.clone()
            boxes_corners[:, 0] = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2  # x1
            boxes_corners[:, 1] = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2  # y1
            boxes_corners[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2  # x2
            boxes_corners[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2  # y2

            if use_nms:
                # 1. Class-aware NMS? 
                # If we want to prevent a "Car" box suppressing a "Person" box, we add offset
                # This is a standard YOLO trick: add (label * huge_number) to coords
                class_offset = labels_tensor.float() * 4096 
                boxes_for_nms = boxes_corners + class_offset[:, None]
                
                # 2. Run NMS
                keep_indices = nms(boxes_for_nms, scores_tensor, nms_thresh)
                
                boxes_list = [boxes_list[i] for i in keep_indices]
                scores_list = [scores_list[i] for i in keep_indices]
                labels_list = [labels_list[i] for i in keep_indices]

            # 3. Limit max detections per image
            picks = [(box, score, label) for box, score, label in zip(boxes_list, scores_list, labels_list)]
            picks = sorted(picks, key=lambda x: x[1], reverse=True)[:max_detections]
            final_detections = picks
            
        outs.append({'detections': final_detections})

    return outs

print("="*60)
print("THERMAL-RGB 2D DETECTION - VISUALIZATION (OPTIMIZED)")
print("="*60)

# ... (The rest of your helper functions: get_attention_map, visualize_detections, create_comparison_grid)
# ... (KEEP THESE EXACTLY AS THEY WERE IN YOUR ORIGINAL FILE, NO CHANGES NEEDED BELOW THIS LINE UNTIL __main__)

def get_attention_map(attention_source, img_idx):
    """Return a 2D attention map for plotting."""
    if attention_source is None:
        return None
    tensor = None
    if isinstance(attention_source, dict):
        for key in ['p3', 'p4', 'p5']:
            if key in attention_source and attention_source[key] is not None:
                tensor = attention_source[key]
                break
        if tensor is None and len(attention_source) > 0:
            tensor = next(iter(attention_source.values()))
        if tensor is None:
            return None
        attn_tensor = tensor[img_idx]
    else:
        attn_tensor = attention_source[img_idx]
    attn_tensor = attn_tensor.detach().cpu().numpy()
    if attn_tensor.ndim >= 3:
        return attn_tensor.mean(axis=0)
    return attn_tensor

def visualize_detections(model, val_loader, annotations, device, num_samples=10, conf_thresh=0.6,
                          img_w=640, img_h=512):
    """Visualize detection results with bounding boxes"""
    
    model.eval()
    output_dir = 'results/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = {0: 'car', 1: 'person', 2: 'bicycle'}
    class_colors = {0: 'red', 1: 'green', 2: 'blue'}
    
    print(f"\n🎨 Creating visualizations for {num_samples} samples...")
    
    sample_count = 0
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= num_samples:
            break
        
        rgb_tensor = batch['rgb']
        thermal_tensor = batch['thermal']
        names = batch['name']
        
        # Get predictions
        with torch.no_grad():
            predictions = model(rgb_tensor.to(device), thermal_tensor.to(device), return_attention=True)
        
        # Decode predictions
        decoded_batch = decode_preds(
            predictions,
            topk=75,
            conf_thresh=conf_thresh,
            img_w=img_w,
            img_h=img_h,
            use_nms=True,
            nms_thresh=0.3, # <--- AGGRESSIVE NMS for Visualization
            max_detections=30
        )
        
        for img_idx in range(rgb_tensor.shape[0]):
            if sample_count >= num_samples:
                break
            
            name = names[img_idx]
            if '/' in name or '\\' in name:
                key = os.path.splitext(os.path.basename(name))[0]
            else:
                key = os.path.splitext(name)[0]
            
            gt = annotations.get(key, {})
            gt_boxes = gt.get('boxes', np.array([]))
            gt_labels = gt.get('labels', np.array([]))
            
            rgb_img = rgb_tensor[img_idx].cpu().numpy().transpose(1, 2, 0)
            rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = (rgb_img * 255).astype(np.uint8)
            
            thermal_img = thermal_tensor[img_idx].cpu().numpy().transpose(1, 2, 0)
            thermal_img = thermal_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            thermal_img = np.clip(thermal_img, 0, 1)
            thermal_img = (thermal_img * 255).astype(np.uint8)
            
            rgb_attn_map = None
            thermal_attn_map = None
            if 'rgb_attention' in predictions:
                rgb_attn_map = get_attention_map(predictions['rgb_attention'], img_idx)
            if 'thermal_attention' in predictions:
                thermal_attn_map = get_attention_map(predictions['thermal_attention'], img_idx)
            
            decoded = decoded_batch[img_idx]
            pred_detections = decoded['detections']
            
            fig = plt.figure(figsize=(20, 12))
            
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(rgb_img)
            ax1.set_title('RGB Input with Predictions', fontsize=14, fontweight='bold')
            for box, score, label in pred_detections:
                cx, cy, w, h = box
                x1 = cx - w/2
                y1 = cy - h/2
                rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=class_colors[label], 
                               facecolor='none', linestyle='--')
                ax1.add_patch(rect)
                ax1.text(x1, y1-5, f'{class_names[label]}: {score:.2f}', 
                        color=class_colors[label], fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            ax1.axis('off')
            
            ax2 = plt.subplot(2, 3, 2)
            ax2.imshow(rgb_img)
            ax2.set_title('RGB Input with Ground Truth', fontsize=14, fontweight='bold')
            if len(gt_boxes) > 0:
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    x1, y1, w, h = gt_box
                    rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=class_colors[int(gt_label)], 
                                   facecolor='none', linestyle='-')
                    ax2.add_patch(rect)
                    ax2.text(x1, y1-5, class_names[int(gt_label)], 
                            color=class_colors[int(gt_label)], fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            ax2.axis('off')
            
            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(thermal_img)
            ax3.set_title('Thermal Input with Predictions', fontsize=14, fontweight='bold')
            for box, score, label in pred_detections:
                cx, cy, w, h = box
                x1 = cx - w/2
                y1 = cy - h/2
                rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=class_colors[label],
                                 facecolor='none', linestyle='--')
                ax3.add_patch(rect)
                ax3.text(x1, y1-5, f'{class_names[label]}: {score:.2f}',
                         color=class_colors[label], fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            ax3.axis('off')
            
            ax4 = plt.subplot(2, 3, 4)
            objectness_tensor = torch.sigmoid(predictions['objectness'][img_idx]).detach().cpu()
            if objectness_tensor.ndim == 3:
                objectness_map = objectness_tensor.mean(dim=0).numpy()
            else:
                objectness_map = objectness_tensor.numpy()
            objectness_resized = cv2.resize(objectness_map, (img_w, img_h))
            im = ax4.imshow(objectness_resized, cmap='hot', alpha=0.7)
            ax4.imshow(rgb_img, alpha=0.3)
            ax4.set_title('Detection Heatmap', fontsize=14, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im, ax=ax4, fraction=0.046)
            
            if rgb_attn_map is not None:
                ax5 = plt.subplot(2, 3, 5)
                rgb_attn_resized = cv2.resize(rgb_attn_map, (640, 512))
                im = ax5.imshow(rgb_attn_resized, cmap='viridis', alpha=0.7)
                ax5.imshow(rgb_img, alpha=0.3)
                ax5.set_title('RGB Attention Map', fontsize=14, fontweight='bold')
                ax5.axis('off')
                plt.colorbar(im, ax=ax5, fraction=0.046)
            
            if thermal_attn_map is not None:
                ax6 = plt.subplot(2, 3, 6)
                thermal_attn_resized = cv2.resize(thermal_attn_map, (640, 512))
                im = ax6.imshow(thermal_attn_resized, cmap='viridis', alpha=0.7)
                ax6.imshow(thermal_img, alpha=0.3)
                ax6.set_title('Thermal Attention Map', fontsize=14, fontweight='bold')
                ax6.axis('off')
                plt.colorbar(im, ax=ax6, fraction=0.046)
            
            plt.suptitle(f'Sample: {key} | Pred: {len(pred_detections)} | GT: {len(gt_boxes)}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.97])   # pyright: ignore[reportArgumentType]
            
            save_path = os.path.join(output_dir, f'detection_{key}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"✓ Saved {sample_count} visualizations to {output_dir}/")

def create_comparison_grid(model, val_loader, device, output_dir='results/visualizations', num_samples=6,
                           img_w=640, img_h=512, conf_thresh=0.6):
    """Create side-by-side comparison grid"""
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = {0: 'car', 1: 'person', 2: 'bicycle'}
    class_colors = {0: 'red', 1: 'green', 2: 'blue'}
    
    print(f"\n📊 Creating comparison grid for {num_samples} samples...")
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= num_samples:
            break
        
        rgb_tensor = batch['rgb']
        thermal_tensor = batch['thermal']
        names = batch['name']
        
        with torch.no_grad():
            predictions = model(rgb_tensor.to(device), thermal_tensor.to(device), return_attention=False)
        
        decoded_batch = decode_preds(
            predictions,
            topk=75,
            conf_thresh=conf_thresh,
            img_w=img_w,
            img_h=img_h,
            use_nms=True,
            nms_thresh=0.3, # <--- AGGRESSIVE NMS
            max_detections=30
        )
        
        for img_idx in range(rgb_tensor.shape[0]):
            if sample_count >= num_samples:
                break
            
            # Denormalize images
            rgb_img = rgb_tensor[img_idx].cpu().numpy().transpose(1, 2, 0)
            rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = (rgb_img * 255).astype(np.uint8)
            
            thermal_img = thermal_tensor[img_idx].cpu().numpy().transpose(1, 2, 0)
            thermal_img = thermal_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            thermal_img = np.clip(thermal_img, 0, 1)
            thermal_img = (thermal_img * 255).astype(np.uint8)
            
            decoded = decoded_batch[img_idx]
            pred_detections = decoded['detections']
            
            # RGB with predictions
            axes[sample_count, 0].imshow(rgb_img)
            axes[sample_count, 0].set_title(f'Sample {sample_count+1}: RGB + Predictions ({len(pred_detections)})', fontsize=12)
            for box, score, label in pred_detections:
                cx, cy, w, h = box
                x1 = cx - w/2
                y1 = cy - h/2
                rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=class_colors[label], 
                               facecolor='none', linestyle='--')
                axes[sample_count, 0].add_patch(rect)
            axes[sample_count, 0].axis('off')
            
            # Thermal
            axes[sample_count, 1].imshow(thermal_img)
            axes[sample_count, 1].set_title(f'Sample {sample_count+1}: Thermal + Predictions', fontsize=12)
            for box, score, label in pred_detections:
                cx, cy, w, h = box
                x1 = cx - w/2
                y1 = cy - h/2
                rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=class_colors[label],
                                 facecolor='none', linestyle='--')
                axes[sample_count, 1].add_patch(rect)
                axes[sample_count, 1].text(x1, y1-5, f'{class_names[label]}: {score:.2f}',
                                         color=class_colors[label], fontsize=9, fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            axes[sample_count, 1].axis('off')
            
            # Objectness heatmap
            objectness_tensor = torch.sigmoid(predictions['objectness'][img_idx]).detach().cpu()
            if objectness_tensor.ndim == 3:
                objectness_map = objectness_tensor.mean(dim=0).numpy()
            else:
                objectness_map = objectness_tensor.numpy()
            objectness_resized = cv2.resize(objectness_map, (img_w, img_h))
            axes[sample_count, 2].imshow(objectness_resized, cmap='hot', alpha=0.7)
            axes[sample_count, 2].imshow(rgb_img, alpha=0.3)
            axes[sample_count, 2].set_title(f'Sample {sample_count+1}: Detection Heatmap', fontsize=12)
            axes[sample_count, 2].axis('off')
            
            sample_count += 1
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison grid to {save_path}")

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Configuration
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
    
    # Load model
    print("\n🏗️  Loading model...")
    checkpoint_path = 'checkpoints/thermal_rgb_2d_latest_yolo/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}!")
        print("   Please train a model first or update the checkpoint path.")
        exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    has_bn = any('bn' in key for key in state_dict.keys())
    has_fpn = any('fpn' in key for key in state_dict.keys())
    
    backbone = cfg['backbone']
    if has_fpn:
        if 'rgb_fpn.lateral_c5.weight' in state_dict:
            c5_channels = state_dict['rgb_fpn.lateral_c5.weight'].shape[1]
            if c5_channels == 2048:
                backbone = 'resnet50'
            elif c5_channels == 512:
                backbone = 'resnet18'
    
    use_bn = has_bn if has_bn else cfg['use_bn']
    use_fpn = has_fpn if has_fpn else cfg['use_fpn']
    num_anchors = cfg['num_anchors']
    use_multiscale = cfg['use_multiscale']
    
    is_yolo = any('reg_head' in key or 'cls_head' in key for key in state_dict.keys())
    
    if is_yolo and LATEST_YOLO_AVAILABLE:
        print("📊 Detected Latest YOLO model for visualization")
        model = ThermalRGB2DNetLatestYOLO(
            num_classes=cfg['num_classes'],
            pretrained=False,
            use_bn=use_bn,
            use_fpn=use_fpn,
            backbone=backbone,
            use_multiscale=use_multiscale,
            yolo_version=cfg['yolo_version'],
            num_anchors=num_anchors
        ).to(device)
    else:
        model = ThermalRGB2DNet(
            num_classes=cfg['num_classes'],
            pretrained=False,
            use_bn=use_bn,
            use_fpn=use_fpn,
            backbone=backbone,
            num_anchors=num_anchors,
            use_multiscale=use_multiscale
        ).to(device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    except RuntimeError as e:
        print(f"⚠️  Warning: Could not load checkpoint strictly: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    # Load validation data
    print("\n📦 Loading validation data...")
    _, val_loader = create_dataloaders(cfg['batch_size'], cfg['num_workers'], 
                                      (cfg['img_height'], cfg['img_width']))
    val_annotations = get_annotations_for_split('val')
    print(f"✓ Validation samples: {len(val_loader.dataset)}")   # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
    
    # --- HERE IS THE TUNING FOR YOUR RESULTS ---
    # conf_thresh: Raised to 0.45 to kill "ghost" boxes (the cloud of 30)
    # visualize_detections uses the default I set in decode_preds (nms=0.3)
    visualize_detections(
        model,
        val_loader,
        val_annotations,
        device,
        num_samples=10,
        conf_thresh=0.45,  # <--- TUNED HIGHER
        img_w=cfg['img_width'],
        img_h=cfg['img_height']
    )
    create_comparison_grid(
        model,
        val_loader,
        device,
        num_samples=6,
        img_w=cfg['img_width'],
        img_h=cfg['img_height'],
        conf_thresh=0.45   # <--- TUNED HIGHER
    )
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*60)