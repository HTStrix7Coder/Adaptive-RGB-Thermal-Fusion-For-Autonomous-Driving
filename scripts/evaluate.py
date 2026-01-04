import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.model import ThermalRGB2DNet
try:
    from Models.model_yolo import ThermalRGB2DNetLatestYOLO
    LATEST_YOLO_AVAILABLE = True
except ImportError:
    LATEST_YOLO_AVAILABLE = False
    print("⚠️ Warning: Latest YOLO model not available. Install ultralytics: pip install ultralytics")
from Models.dataset import create_dataloaders
from utils.annotations import get_annotations_for_split

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [cx, cy, w, h] format"""
    # Convert to [x1, y1, x2, y2]
    def to_corners(box):
        cx, cy, w, h = box
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]
    
    box1 = to_corners(box1)
    box2 = to_corners(box2)
    
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_map(results, annotations, iou_thresholds=[0.5, 0.75], conf_thresh=0.5):
    """
    Calculate mAP (mean Average Precision) - standard object detection metric
    Returns mAP@0.5 and mAP@0.75
    """
    class_names = {0: 'car', 1: 'person', 2: 'bicycle'}
    aps = {}
    
    for iou_thresh in iou_thresholds:
        aps_per_class = []
        for cls in range(3):
            # Get all predictions and ground truth for this class
            pred_scores = []
            pred_matches = []
            gt_count = 0
            
            for name, decoded in results:
                if '/' in name or '\\' in name:
                    key = os.path.splitext(os.path.basename(name))[0]
                else:
                    key = os.path.splitext(name)[0]
                
                gt = annotations.get(key, {})
                gt_boxes = gt.get('boxes', np.array([]))
                gt_labels = gt.get('labels', np.array([]))
                
                # Count GT for this class
                gt_count += np.sum(gt_labels == cls) if len(gt_labels) > 0 else 0
                
                # Get predictions for this class
                matched_gt = set()
                for pred_box, score, pred_label in decoded['detections']:
                    if pred_label != cls or score < conf_thresh:
                        continue
                    
                    # Find best matching GT
                    best_iou = 0.0
                    best_gt_idx = -1
                    for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if i in matched_gt or gt_label != cls:
                            continue
                        
                        gt_cx = gt_box[0] + gt_box[2]/2
                        gt_cy = gt_box[1] + gt_box[3]/2
                        gt_box_cx = [gt_cx, gt_cy, gt_box[2], gt_box[3]]
                        iou = calculate_iou(pred_box, gt_box_cx)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    pred_scores.append(score)
                    pred_matches.append(1 if best_iou >= iou_thresh else 0)
                    if best_iou >= iou_thresh:
                        matched_gt.add(best_gt_idx)
            
            # Calculate AP (Average Precision)
            if len(pred_scores) == 0:
                ap = 0.0
            else:
                # Sort by score descending
                sorted_indices = np.argsort(pred_scores)[::-1]
                sorted_matches = np.array(pred_matches)[sorted_indices]
                
                # Calculate precision-recall curve
                tp_cumsum = np.cumsum(sorted_matches)
                fp_cumsum = np.cumsum(1 - sorted_matches)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                recall = tp_cumsum / (gt_count + 1e-6)
                
                # Calculate AP using 11-point interpolation
                ap = 0.0
                for t in np.arange(0, 1.1, 0.1):
                    if np.sum(recall >= t) == 0:
                        p = 0
                    else:
                        p = np.max(precision[recall >= t])
                    ap += p / 11.0
            
            aps_per_class.append(ap)
        
        # mAP is mean of APs across all classes
        map_value = np.mean(aps_per_class)
        aps[f'mAP@{iou_thresh}'] = map_value
        aps[f'AP_per_class@{iou_thresh}'] = {class_names[i]: aps_per_class[i] for i in range(3)}
    
    return aps

def calculate_image_accuracy(results, annotations, iou_thresh=0.5, conf_thresh=0.5):
    """
    Calculate per-image accuracy: % of images where all objects are correctly detected
    """
    correct_images = 0
    total_images = 0
    
    for name, decoded in results:
        if '/' in name or '\\' in name:
            key = os.path.splitext(os.path.basename(name))[0]
        else:
            key = os.path.splitext(name)[0]
        
        gt = annotations.get(key, {})
        gt_boxes = gt.get('boxes', np.array([]))
        gt_labels = gt.get('labels', np.array([]))
        
        if len(gt_boxes) == 0:
            # Empty image - correct if no predictions
            if len(decoded['detections']) == 0:
                correct_images += 1
            total_images += 1
            continue
        
        total_images += 1
        matched_gt = set()
        
        # Check if all GT objects are detected
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            found = False
            for pred_box, score, pred_label in decoded['detections']:
                if score < conf_thresh or pred_label != gt_label or i in matched_gt:
                    continue
                
                gt_cx = gt_box[0] + gt_box[2]/2
                gt_cy = gt_box[1] + gt_box[3]/2
                gt_box_cx = [gt_cx, gt_cy, gt_box[2], gt_box[3]]
                iou = calculate_iou(pred_box, gt_box_cx)
                
                if iou >= iou_thresh:
                    matched_gt.add(i)
                    found = True
                    break
            
            if not found:
                break
        
        # Image is correct if all GT objects are matched
        if len(matched_gt) == len(gt_boxes):
            correct_images += 1
    
    return correct_images / total_images if total_images > 0 else 0.0

def evaluate_detections(results, annotations, iou_thresh=0.5, conf_thresh=0.5):
    """Calculate precision, recall, mAP, and accuracy metrics"""
    class_names = {0: 'car', 1: 'person', 2: 'bicycle'}
    
    # Per-class statistics
    stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for cls in range(3)}
    
    for name, decoded in results:
        # Get base name (remove path/extensions)
        if '/' in name or '\\' in name:
            key = os.path.splitext(os.path.basename(name))[0]
        else:
            key = os.path.splitext(name)[0]
        
        gt = annotations.get(key, {})
        gt_boxes = gt.get('boxes', np.array([]))
        gt_labels = gt.get('labels', np.array([]))
        
        preds = decoded['detections']
        
        # Count ground truth objects
        for cls in range(3):
            stats[cls]['gt'] += np.sum(gt_labels == cls) if len(gt_labels) > 0 else 0
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred_box, score, pred_label in preds:
            if score < conf_thresh:
                continue
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching GT box
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if i in matched_gt or gt_label != pred_label:
                    continue
                
                # Convert GT box from [x,y,w,h] to [cx,cy,w,h]
                gt_cx = gt_box[0] + gt_box[2]/2
                gt_cy = gt_box[1] + gt_box[3]/2
                gt_box_cx = [gt_cx, gt_cy, gt_box[2], gt_box[3]]
                
                iou = calculate_iou(pred_box, gt_box_cx)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Match if IoU > threshold
            if best_iou >= iou_thresh:
                stats[pred_label]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                stats[pred_label]['fp'] += 1
        
        # Count false negatives (unmatched GT)
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if i not in matched_gt:
                stats[gt_label]['fn'] += 1
    
    # Calculate metrics
    metrics = {}
    for cls in range(3):
        tp = stats[cls]['tp']
        fp = stats[cls]['fp']
        fn = stats[cls]['fn']
        gt = stats[cls]['gt']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_names[cls]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'gt': int(gt)
        }
    
    # Overall metrics
    total_tp = sum(stats[cls]['tp'] for cls in range(3))
    total_fp = sum(stats[cls]['fp'] for cls in range(3))
    total_fn = sum(stats[cls]['fn'] for cls in range(3))
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    metrics['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': int(total_tp),
        'fp': int(total_fp),
        'fn': int(total_fn)
    }
    
    # Calculate mAP (mean Average Precision) - standard object detection metric
    map_results = calculate_map(results, annotations, iou_thresholds=[0.5, 0.75], conf_thresh=conf_thresh)
    metrics['mAP'] = map_results
    
    # Calculate per-image accuracy
    img_accuracy = calculate_image_accuracy(results, annotations, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
    metrics['image_accuracy'] = img_accuracy
    
    return metrics

def nms(boxes, scores, labels, iou_threshold=0.5):
    """
    Non-Maximum Suppression to remove duplicate detections
    boxes: list of [cx, cy, w, h]
    scores: list of confidence scores
    labels: list of class labels
    """
    if len(boxes) == 0:
        return [], [], []
    
    # Convert to [x1, y1, x2, y2] for NMS
    boxes_corners = []
    for cx, cy, w, h in boxes:
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        boxes_corners.append([x1, y1, x2, y2])
    
    boxes_corners = np.array(boxes_corners)
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Sort by score (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Take highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes_corners[current]
        other_boxes = boxes_corners[indices[1:]]
        
        # Calculate IoU
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_others - intersection
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with IoU < threshold (and same class)
        same_class = labels[indices[1:]] == labels[current]
        keep_mask = (iou < iou_threshold) | (~same_class)
        indices = indices[1:][keep_mask]
    
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]

def decode_preds(preds, topk=100, conf_thresh=0.5, img_w=640, img_h=512, use_nms=True, nms_thresh=0.5):
    """
    Convert network outputs to list of detections per image.
    Handles both anchor-based [B, num_anchors, H, W] and non-anchor [B, 1, H, W] formats.
    Returns list of dicts: {'boxes': [N,4] in pixel coords cx,cy,w,h, 'scores':[N], 'labels':[N]}
    """
    obj_logits = preds['objectness']  # [B, num_anchors, H, W] or [B, 1, H, W]
    cls_logits = preds['classification']  # [B, num_anchors*C, H, W] or [B, C, H, W]
    bbox_reg = preds['bbox']  # [B, num_anchors*4, H, W] or [B, 4, H, W]

    B = obj_logits.shape[0]
    obj_shape = obj_logits.shape
    
    # Detect anchor format
    if len(obj_shape) == 4:
        if obj_shape[1] == 1:
            num_anchors = 1
            _, _, Hf, Wf = obj_shape
        else:
            num_anchors = obj_shape[1]
            _, _, Hf, Wf = obj_shape
    else:
        raise ValueError(f"Unexpected objectness shape: {obj_shape}")
    
    num_classes = cls_logits.shape[1] // num_anchors if num_anchors > 1 else cls_logits.shape[1]
    
    outs = []

    for b in range(B):
        boxes_list = []
        scores_list = []
        labels_list = []
        
        if num_anchors == 1:
            # Old format: no anchors
            obj_map = torch.sigmoid(obj_logits[b, 0]).cpu().numpy()  # Hf x Wf
            bbox_map = torch.sigmoid(bbox_reg[b]).cpu().numpy()    # 4 x Hf x Wf

            if isinstance(cls_logits, torch.Tensor):
                cls_softmax = torch.nn.functional.softmax(cls_logits[b], dim=0)
            else:
                cls_softmax = cls_logits[b]
            cls_map = cls_softmax.cpu().numpy() if isinstance(cls_softmax, torch.Tensor) else cls_softmax

            # Flatten and get topk
            flat_scores = obj_map.flatten()
            inds = np.argsort(flat_scores)[::-1][:topk]
                
            for ind in inds:
                score = flat_scores[ind]
                if score < conf_thresh:
                    continue
                gy = ind // Wf
                gx = ind % Wf
                # bbox normalized
                cx_n = bbox_map[0, gy, gx]
                cy_n = bbox_map[1, gy, gx]
                w_n = bbox_map[2, gy, gx]
                h_n = bbox_map[3, gy, gx]
                # map to pixels
                cx = float(cx_n) * img_w
                cy = float(cy_n) * img_h
                w = float(w_n) * img_w
                h = float(h_n) * img_h
                label = int(np.argmax(cls_map[:, gy, gx]))
                boxes_list.append([cx, cy, w, h])
                scores_list.append(float(score))
                labels_list.append(label)
        else:
            # Anchor format: process each anchor
            obj_maps = torch.sigmoid(obj_logits[b]).cpu().numpy()  # [num_anchors, Hf, Wf]
            bbox_reg_anchors = bbox_reg[b].reshape(num_anchors, 4, Hf, Wf)  # [num_anchors, 4, H, W]
            bbox_maps = torch.sigmoid(bbox_reg_anchors).cpu().numpy()  # [num_anchors, 4, H, W]
            
            cls_logits_anchors = cls_logits[b].reshape(num_anchors, num_classes, Hf, Wf)  # [num_anchors, C, H, W]
            cls_softmax = torch.nn.functional.softmax(cls_logits_anchors, dim=1)  # [num_anchors, C, H, W]
            cls_maps = cls_softmax.cpu().numpy()
            
            # Collect predictions from all anchors
            for a in range(num_anchors):
                obj_map = obj_maps[a]  # [H, W]
                bbox_map = bbox_maps[a]  # [4, H, W]
                cls_map = cls_maps[a]  # [C, H, W]
                
                flat_scores = obj_map.flatten()
                inds = np.argsort(flat_scores)[::-1][:topk]
                
                for ind in inds:
                    score = flat_scores[ind]
                    if score < conf_thresh:
                        continue
                    gy = ind // Wf
                    gx = ind % Wf
                    
                    cx_n = bbox_map[0, gy, gx]
                    cy_n = bbox_map[1, gy, gx]
                    w_n = bbox_map[2, gy, gx]
                    h_n = bbox_map[3, gy, gx]
                    
                    cx = float(cx_n) * img_w
                    cy = float(cy_n) * img_h
                    w = float(w_n) * img_w
                    h = float(h_n) * img_h
                    label = int(np.argmax(cls_map[:, gy, gx]))
                    
                    boxes_list.append([cx, cy, w, h])
                    scores_list.append(float(score))
                    labels_list.append(label)
        
        # Apply NMS to reduce duplicate detections
        if use_nms and len(boxes_list) > 0:
            boxes_list, scores_list, labels_list = nms(boxes_list, scores_list, labels_list, nms_thresh)
        
        picks = [(box, score, label) for box, score, label in zip(boxes_list, scores_list, labels_list)]
        outs.append({'detections': picks})
    return outs

if __name__ == "__main__":
    cfg = {
        'batch_size': 6,  # Match training
        'num_workers': 6, # Match training
        'img_height': 512,
        'img_width': 640,
        'num_classes': 3,
        'backbone': 'resnet18',
        'use_fpn': True,
        'use_bn': True,
        'num_anchors': 3,       # Updated to match training
        'use_multiscale': True, # Updated to match training
        'yolo_version': 'latest'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_dataloaders(cfg['batch_size'], cfg['num_workers'],
                                                  (cfg['img_height'], cfg['img_width']))
    val_annos = get_annotations_for_split('val')

    # Use the latest training checkpoint
    ckpt = 'checkpoints/thermal_rgb_2d_latest_yolo_fixed/best_model.pth'
    if not os.path.exists(ckpt):
        # Fallback to run3 if latest doesn't exist yet
        fallback_ckpt = 'checkpoints/thermal_rgb_2d_latest_yolo_run3/best_model.pth'
        if os.path.exists(fallback_ckpt):
            print(f"⚠️  Latest checkpoint not found, using fallback: {fallback_ckpt}")
            ckpt = fallback_ckpt
        else:
            print(f"❌ Checkpoint not found: {ckpt}")
            print(f"   Also checked: {fallback_ckpt}")
            print("   Please ensure the model has been trained.")
            raise RuntimeError(f"Checkpoint not found: {ckpt}")
    
    print(f"✓ Loading checkpoint from: {ckpt}")
    cp = torch.load(ckpt, map_location=device)
    
    # Try to infer architecture from checkpoint state_dict
    state_dict = cp['model_state_dict']
    has_bn = any('bn' in key for key in state_dict.keys())
    has_fpn = any('fpn' in key for key in state_dict.keys())
    
    # Determine backbone from channel sizes in state_dict
    backbone = 'resnet18'  # Default
    if has_fpn:
        # For FPN models, check lateral layer input channels (more reliable)
        if 'rgb_fpn.lateral_c5.weight' in state_dict:
            # lateral_c5 input channels: 512 for ResNet18, 2048 for ResNet50
            c5_channels = state_dict['rgb_fpn.lateral_c5.weight'].shape[1]
            if c5_channels == 2048:
                backbone = 'resnet50'
            elif c5_channels == 512:
                backbone = 'resnet18'
        elif 'rgb_fpn.lateral_c4.weight' in state_dict:
            # lateral_c4 input channels: 256 for ResNet18, 1024 for ResNet50
            c4_channels = state_dict['rgb_fpn.lateral_c4.weight'].shape[1]
            if c4_channels == 1024:
                backbone = 'resnet50'
            elif c4_channels == 256:
                backbone = 'resnet18'
    else:
        # For non-FPN models, check encoder layer4 channels
        if 'rgb_encoder.layer4.0.conv1.weight' in state_dict:
            # Check channel size to determine backbone
            weight_shape = state_dict['rgb_encoder.layer4.0.conv1.weight'].shape
            if len(weight_shape) == 4 and weight_shape[0] == 2048:
                backbone = 'resnet50'
            elif len(weight_shape) == 4 and weight_shape[0] == 512:
                backbone = 'resnet18'
    
    # Use inferred values or fall back to config
    use_bn = has_bn if has_bn else cfg.get('use_bn', True)
    use_fpn = has_fpn if has_fpn else cfg.get('use_fpn', True)
    
    print(f"📊 Inferred model architecture:")
    print(f"   Backbone: {backbone}")
    print(f"   FPN: {use_fpn}")
    print(f"   BatchNorm: {use_bn}")
    
    # Detect if this is a latest YOLO model or custom model
    # Latest YOLO heads use reg_head/cls_head naming (e.g., head_p3.reg_head.weight)
    is_yolo = any('reg_head' in key or 'cls_head' in key for key in state_dict.keys())
    
    # Load model with inferred/config settings
    num_anchors = cfg.get('num_anchors', 1)
    use_multiscale = cfg.get('use_multiscale', False)

    if is_yolo and LATEST_YOLO_AVAILABLE:
        print("📊 Detected Latest YOLO model architecture")
        model = ThermalRGB2DNetLatestYOLO(
            num_classes=cfg['num_classes'], 
            pretrained=False, 
            use_bn=use_bn, 
            use_fpn=use_fpn, 
            backbone=backbone,
            use_multiscale=use_multiscale,
            yolo_version=cfg.get('yolo_version', 'latest'),
            num_anchors=num_anchors
        ).to(device)
    else:
        if is_yolo and not LATEST_YOLO_AVAILABLE:
            print("⚠️ Warning: Latest YOLO model detected but ultralytics not available")
            print("   Install with: pip install ultralytics")
            print("   Falling back to custom model (may not load correctly)")
        print("📊 Using custom model architecture")
        model = ThermalRGB2DNet(
            num_classes=cfg['num_classes'], 
            pretrained=False, 
            use_bn=use_bn, 
            use_fpn=use_fpn, 
            backbone=backbone,
            num_anchors=num_anchors,
            use_multiscale=use_multiscale
        ).to(device)
    
    # Try loading with strict=True first
    try:
        model.load_state_dict(cp['model_state_dict'], strict=True)
        print("✓ Checkpoint loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"⚠️ Warning: Model architecture mismatch. Error: {str(e)[:100]}...")
        print("   Trying to load with strict=False...")
        model.load_state_dict(cp['model_state_dict'], strict=False)
        print("✓ Loaded checkpoint (some layers may be missing)")
    
    model.eval()

    results = []
    print("\n🔍 Running inference on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            rgb = batch['rgb'].to(device, non_blocking=True)
            thermal = batch['thermal'].to(device, non_blocking=True)
            names = batch['name']
            preds = model(rgb, thermal, return_attention=False)
            # Decode predictions with low confidence threshold for evaluation
            # We'll test different thresholds later
            decoded = decode_preds(preds, topk=100, conf_thresh=0.01, img_w=cfg['img_width'], img_h=cfg['img_height'], 
                                 use_nms=True, nms_thresh=0.5)
            results.extend(list(zip(names, decoded)))

    print(f"✓ Processed {len(results)} images")
    total_detections = sum(len(r[1]['detections']) for r in results)
    print(f"✓ Total detections: {total_detections} (avg {total_detections/len(results):.1f} per image)")

    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Calculate metrics with different thresholds
    # Extended range to find better precision/recall balance
    print("\n📊 Testing multiple confidence thresholds...")
    best_f1 = -1
    best_conf = 0.5
    best_metrics = None
    
    # Test wider range including higher thresholds to reduce false positives
    for conf_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        try:
            metrics = evaluate_detections(results, val_annos, iou_thresh=0.5, conf_thresh=conf_thresh)
            if metrics is not None and metrics.get('overall') is not None:
                f1 = metrics['overall'].get('f1', 0.0)
                prec = metrics['overall'].get('precision', 0.0)
                rec = metrics['overall'].get('recall', 0.0)
                print(f"  Conf={conf_thresh:.2f}: F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, TP={metrics['overall'].get('tp', 0)}, FP={metrics['overall'].get('fp', 0)}")
                if f1 > best_f1:
                    best_f1 = f1
                    best_conf = conf_thresh
                    best_metrics = metrics
            else:
                print(f"  Conf={conf_thresh:.2f}: No metrics available")
        except Exception as e:
            print(f"  Conf={conf_thresh:.2f}: Error - {e}")
            continue
    
    if best_metrics is None:
        print("\n❌ ERROR: Could not calculate metrics. Check if model is producing predictions.")
        print(f"   Total results: {len(results)}")
        if len(results) > 0:
            print(f"   Example: {results[0][0]} has {len(results[0][1]['detections'])} detections")
        exit(1)
    
    metrics = best_metrics
    print(f"\n✅ Best confidence threshold: {best_conf:.2f} (F1={best_f1:.3f})")
    if best_metrics and best_metrics.get('overall'):
        print(f"   Precision: {best_metrics['overall'].get('precision', 0.0):.3f}")
        print(f"   Recall: {best_metrics['overall'].get('recall', 0.0):.3f}")
        print(f"   TP: {best_metrics['overall'].get('tp', 0)}, FP: {best_metrics['overall'].get('fp', 0)}, FN: {best_metrics['overall'].get('fn', 0)}")
    
    # Print per-class metrics
    print(f"\n📊 Per-Class Metrics (IoU=0.5, Conf={best_conf:.2f}):")
    if metrics is not None:
        for class_name in metrics:
            if class_name not in ['overall', 'mAP', 'image_accuracy'] and metrics[class_name] is not None:
                m = metrics[class_name]
                print(f"\n{class_name.upper()}:")
                print(f"  Precision: {m.get('precision', 0.0):.3f}")
                print(f"  Recall:    {m.get('recall', 0.0):.3f}")
                print(f"  F1-Score:  {m.get('f1', 0.0):.3f}")
                print(f"  TP: {m.get('tp', 0)}, FP: {m.get('fp', 0)}, FN: {m.get('fn', 0)}, GT: {m.get('gt', 0)}")
    
    # Print overall metrics
    print(f"\n{'='*60}")
    print("OVERALL METRICS:")
    if metrics is not None and metrics.get('overall') is not None:
        print(f"  Precision: {metrics['overall']['precision']:.3f}")
        print(f"  Recall:    {metrics['overall']['recall']:.3f}")
        print(f"  F1-Score:  {metrics['overall']['f1']:.3f}")
        print(f"  TP: {metrics['overall']['tp']}, FP: {metrics['overall']['fp']}, FN: {metrics['overall']['fn']}")
        
        # Print mAP (standard object detection metric)
        if 'mAP' in metrics and metrics['mAP']:
            print(f"\n📈 mAP (mean Average Precision) - Standard Object Detection Metric:")
            for map_key, map_value in metrics['mAP'].items():
                if 'mAP@' in map_key:
                    print(f"  {map_key}: {map_value:.3f}")
                elif 'AP_per_class@' in map_key:
                    iou_val = map_key.replace('AP_per_class@', '')
                    print(f"\n  Per-Class AP (IoU={iou_val}):")
                    for cls_name, ap_val in map_value.items():
                        print(f"    {cls_name}: {ap_val:.3f}")
        
        # Print image accuracy
        if 'image_accuracy' in metrics:
            print(f"\n🎯 Per-Image Accuracy: {metrics['image_accuracy']:.3f}")
            print(f"   (% of images where all objects are correctly detected)")
    else:
        print("  No overall metrics available (metrics is None).")
    print("="*60)
    
    # Save metrics to file
    output_file = 'results/evaluation_metrics_latest_v2.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to: {output_file}")
    
    # Show example detections
    print(f"\n📸 Example detections (first 3 images):")
    for name, decoded in results[:3]:
        print(f"\n{name}: {len(decoded['detections'])} detections")
        for i, (box, score, label) in enumerate(decoded['detections'][:3]):  # Show first 3
            class_name = {0: 'car', 1: 'person', 2: 'bicycle'}[label]
            print(f"  {i+1}. {class_name}: conf={score:.3f}, box=[cx={box[0]:.1f}, cy={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}]")
