import torch
import torch.nn.functional as F
import numpy as np

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    Reduces the contribution of easy examples and focuses on hard examples
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)  # probability of correct class
    focal_weight = alpha * (1 - p_t) ** gamma
    return focal_weight * bce  # Return per-element loss (not mean)

def giou_loss(pred_boxes, target_boxes):
    """
    Generalized IoU Loss for better bbox regression
    More stable than L1/L2 loss for bounding boxes
    """
    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    def to_corners(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    pred_corners = to_corners(pred_boxes)
    target_corners = to_corners(target_boxes)
    
    # Calculate IoU
    inter_x1 = torch.max(pred_corners[:, 0], target_corners[:, 0])
    inter_y1 = torch.max(pred_corners[:, 1], target_corners[:, 1])
    inter_x2 = torch.min(pred_corners[:, 2], target_corners[:, 2])
    inter_y2 = torch.min(pred_corners[:, 3], target_corners[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
    target_area = (target_corners[:, 2] - target_corners[:, 0]) * (target_corners[:, 3] - target_corners[:, 1])
    union_area = pred_area + target_area - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    
    # Calculate GIoU
    enclose_x1 = torch.min(pred_corners[:, 0], target_corners[:, 0])
    enclose_y1 = torch.min(pred_corners[:, 1], target_corners[:, 1])
    enclose_x2 = torch.max(pred_corners[:, 2], target_corners[:, 2])
    enclose_y2 = torch.max(pred_corners[:, 3], target_corners[:, 3])
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    return (1 - giou).mean()

class YOLO2DLoss(torch.nn.Module):
    """
    YOLO-style normalized bbox loss.
    Assumptions:
      - Grid: head outputs have spatial size Hf x Wf
      - For a GT box, we assign it to the grid cell containing its center.
      - Predictions:
          objectness logits (raw) -> BCEWithLogits
          class logits -> CrossEntropy
          bbox_reg (raw) -> apply sigmoid for cx, cy, w, h in [0,1]
    """
    def __init__(self, num_classes=3, lambda_box=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0,
                 use_focal_loss=True, use_giou_loss=True, focal_alpha=0.25, focal_gamma=2.0,
                 class_weights=None, num_anchors=1, use_multiscale=False):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.use_focal_loss = use_focal_loss
        self.use_giou_loss = use_giou_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # Class-specific weights for handling class imbalance (e.g., bicycle is rare)
        # Default: [1.0, 1.0, 2.0] for [car, person, bicycle] - bicycle gets 2x weight
        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
            if num_classes >= 3:
                self.class_weights[2] = 2.0  # Bicycle (class 2) gets 2x weight
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        self.num_anchors = num_anchors  # Anchors per scale
        self.use_multiscale = use_multiscale  # If True, predictions are concatenated from multiple scales

    def forward(self, preds, targets):
        """
        preds: dict with 'objectness' [B,num_anchors,Hf,Wf] or [B,1,Hf,Wf], 
               'classification' [B,num_anchors*C,Hf,Wf] or [B,C,Hf,Wf], 
               'bbox' [B,num_anchors*4,Hf,Wf] or [B,4,Hf,Wf]
        targets: list of dicts len B, each {'boxes': np.array([N,4]) in [x,y,w,h] pixels, 'labels': np.array([N])}
                 IMPORTANT: boxes are pixel coords in original image size (W_img,H_img) — we'll normalize
        Returns: dict with loss components and total
        """
        device = preds['objectness'].device
        batch_size = preds['objectness'].shape[0]
        obj_shape = preds['objectness'].shape
        
        # Detect if using anchor boxes and multi-scale concatenation
        if len(obj_shape) == 4:
            if obj_shape[1] == 1:
                # Old format: [B, 1, H, W] - no anchors
                detected_anchors = 1
                _, _, Hf, Wf = obj_shape
            else:
                # New format: [B, num_anchors, H, W] - with anchors
                # If multi-scale concatenation: [B, anchors*num_scales, H, W]
                detected_anchors = obj_shape[1]
                _, _, Hf, Wf = obj_shape
                
                # If multi-scale concatenation is used, the detected anchors = anchors_per_scale * num_scales
                # We need to use the actual anchors_per_scale for target assignment
                if self.use_multiscale and detected_anchors > self.num_anchors:
                    # Multi-scale concatenation: detected_anchors = anchors_per_scale * num_scales (typically 3*3=9)
                    # We'll handle this by treating each scale-anchor combination as a separate prediction
                    # This is acceptable for anchor-free detection style
                    num_anchors = detected_anchors  # Use all concatenated predictions
                else:
                    num_anchors = detected_anchors
        else:
            raise ValueError(f"Unexpected objectness shape: {obj_shape}")
        
        # Debug: Print grid resolution on first call
        if not hasattr(self, '_grid_resolution_logged'):
            print(f"📊 Grid resolution: {Hf}x{Wf} = {Hf*Wf} cells (per image)")
            if self.use_multiscale and num_anchors > self.num_anchors:
                num_scales = num_anchors // self.num_anchors
                print(f"📊 Multi-scale detection: {num_scales} scales × {self.num_anchors} anchors = {num_anchors} predictions per cell")
                print(f"   Total predictions: {Hf*Wf*num_anchors} per image")
            elif num_anchors > 1:
                print(f"📊 Using {num_anchors} anchors per cell (total predictions: {Hf*Wf*num_anchors} per image)")
            if Hf*Wf < 2000:
                print(f"⚠️  WARNING: Low grid resolution ({Hf*Wf} cells). Consider using higher resolution FPN level.")
            self._grid_resolution_logged = True

        obj_logits = preds['objectness']  # [B, num_anchors, H, W] or [B, 1, H, W]
        cls_logits = preds['classification']  # [B, num_anchors*C, H, W] or [B, C, H, W]
        bbox_reg = preds['bbox']  # [B, num_anchors*4, H, W] or [B, 4, H, W]

        bce_obj = torch.nn.BCEWithLogitsLoss(reduction='none')
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        l1 = torch.nn.SmoothL1Loss(reduction='none')

        loss_obj = torch.tensor(0.0, device=device)
        loss_noobj = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        pos_count = 0
        # Track multiple objects per cell (YOLO limitation)
        cells_with_multiple_objects = 0
        cell_object_count = {}  # (b, grid_y, grid_x) -> count
        
        # For each image in batch: build target tensors
        # Handle both anchor and non-anchor cases
        if num_anchors == 1:
            target_obj = torch.zeros_like(obj_logits, device=device)  # [B,1,Hf,Wf]
            target_cls = torch.zeros((batch_size, self.num_classes, Hf, Wf), device=device)
            target_bbox = torch.zeros((batch_size,4,Hf,Wf), device=device)
        else:
            # With anchors: assign to best matching anchor
            target_obj = torch.zeros_like(obj_logits, device=device)  # [B,num_anchors,Hf,Wf]
            target_cls = torch.zeros((batch_size, num_anchors, self.num_classes, Hf, Wf), device=device)
            target_bbox = torch.zeros((batch_size, num_anchors, 4, Hf, Wf), device=device)

        for b in range(batch_size):
            t = targets[b]
            boxes = t['boxes']  # Nx4 in pixels [x,y,w,h]
            labels = t['labels']  # N
            if boxes is None or len(boxes)==0:
                continue
            H_img = t.get('img_h', 512)
            W_img = t.get('img_w', 640)

            for i,box in enumerate(boxes):
                x,y,w_box,h_box = box  # top-left x,y and width,height in pixels
                # Convert to Python floats (handles numpy types)
                x, y, w_box, h_box = float(x), float(y), float(w_box), float(h_box)
                cx = x + w_box/2.0
                cy = y + h_box/2.0
                # normalized center
                cx_n = cx / float(W_img)
                cy_n = cy / float(H_img)
                w_n = w_box / float(W_img)
                h_n = h_box / float(H_img)

                # assign to grid cell
                grid_x = int(np.clip((cx_n * Wf), 0, Wf-1))
                grid_y = int(np.clip((cy_n * Hf), 0, Hf-1))

                if num_anchors == 1:
                    # Old format: single anchor per cell
                    # Track multiple objects per cell (YOLO limitation - only last object is kept)
                    cell_key = (b, grid_y, grid_x)
                    if cell_key in cell_object_count:
                        cell_object_count[cell_key] += 1
                        if cell_object_count[cell_key] == 2:  # First time we see 2 objects
                            cells_with_multiple_objects += 1
                    else:
                        cell_object_count[cell_key] = 1

                    # create targets (ensure values are Python floats for tensor assignment)
                    # NOTE: If multiple objects share a cell, only the last one is kept (YOLO limitation)
                    target_obj[b,0,grid_y,grid_x] = 1.0
                    target_cls[b, labels[i], grid_y, grid_x] = 1.0
                    # store bbox in normalized coordinates
                    target_bbox[b,0,grid_y,grid_x] = float(cx_n)
                    target_bbox[b,1,grid_y,grid_x] = float(cy_n)
                    target_bbox[b,2,grid_y,grid_x] = float(w_n)
                    target_bbox[b,3,grid_y,grid_x] = float(h_n)
                else:
                    # New format: multiple anchors per cell - assign to first available anchor
                    # For anchor-free detection style (modern YOLO), this is acceptable
                    # Each anchor learns to predict different objects in the same cell
                    anchor_idx = 0
                    # Find first unused anchor in this cell
                    for a in range(num_anchors):
                        if target_obj[b,a,grid_y,grid_x] == 0.0:
                            anchor_idx = a
                            break
                    else:
                        # All anchors used - overwrite first one (track as multiple objects)
                        # This is expected when many objects share the same cell
                        cells_with_multiple_objects += 1
                        anchor_idx = 0
                    
                    target_obj[b,anchor_idx,grid_y,grid_x] = 1.0
                    target_cls[b,anchor_idx, labels[i], grid_y, grid_x] = 1.0
                    target_bbox[b,anchor_idx,0,grid_y,grid_x] = float(cx_n)
                    target_bbox[b,anchor_idx,1,grid_y,grid_x] = float(cy_n)
                    target_bbox[b,anchor_idx,2,grid_y,grid_x] = float(w_n)
                    target_bbox[b,anchor_idx,3,grid_y,grid_x] = float(h_n)
                
                pos_count += 1

        # Objectness loss: use focal loss if enabled, else standard BCE
        if self.use_focal_loss:
            # Focal loss for objectness (handles class imbalance better)
            focal_loss_map = focal_loss(obj_logits, target_obj, self.focal_alpha, self.focal_gamma)
            loss_obj = self.lambda_obj * (focal_loss_map * target_obj).sum()
            loss_noobj = self.lambda_noobj * (focal_loss_map * (1 - target_obj)).sum()
        else:
            # Standard BCE loss
            obj_loss_map = bce_obj(obj_logits, target_obj)
            loss_obj = self.lambda_obj * (obj_loss_map * target_obj).sum()
            loss_noobj = self.lambda_noobj * (obj_loss_map * (1 - target_obj)).sum()

        # Classification loss at positive cells only - VECTORIZED for speed
        if pos_count > 0:
            if num_anchors == 1:
                # Old format: [B, C, H, W] classification
                pos_mask = (target_obj > 0.5).float()  # [B, 1, Hf, Wf]
                target_labels = target_cls.argmax(dim=1)  # [B, Hf, Wf]
                
                # Classification loss - vectorized
                if self.use_focal_loss:
                    # Focal loss for classification: compute CE for all cells, then mask and apply focal weighting
                    # Reshape for cross_entropy: [B, C, Hf, Wf] -> [B*Hf*Wf, C] and [B, Hf, Wf] -> [B*Hf*Wf]
                    B, C, H, W = cls_logits.shape
                    cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
                    target_labels_flat = target_labels.reshape(-1)  # [B*H*W]
                    
                    # Compute CE loss for all cells
                    ce_loss_flat = F.cross_entropy(cls_logits_flat, target_labels_flat, reduction='none')  # [B*H*W]
                    ce_loss = ce_loss_flat.reshape(B, H, W)  # [B, H, W]
                    
                    # Apply focal weighting only to positive cells
                    p_t = torch.exp(-ce_loss)
                    focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
                    cls_loss_map = focal_weight * ce_loss  # [B, H, W]
                    
                    # Apply class-specific weights for handling class imbalance
                    # Get class weights for each positive cell
                    class_weights_tensor = self.class_weights.to(device)  # [num_classes]
                    # For each positive cell, get its class and apply weight
                    pos_mask_flat = pos_mask.squeeze(1).reshape(-1)  # [B*H*W]
                    target_labels_flat_for_weights = target_labels.reshape(-1)  # [B*H*W]
                    # Create weight map: for each cell, get weight of its target class
                    weight_map = class_weights_tensor[target_labels_flat_for_weights].reshape(B, H, W)  # [B, H, W]
                    
                    # Mask to only positive cells and apply class weights
                    cls_loss_val = (cls_loss_map * pos_mask.squeeze(1) * weight_map).sum()  # [B, H, W] * [B, H, W] * [B, H, W]
                else:
                    # Standard cross-entropy: compute for all cells, mask to positives
                    B, C, H, W = cls_logits.shape
                    cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
                    target_labels_flat = target_labels.reshape(-1)  # [B*H*W]
                    ce_loss_flat = F.cross_entropy(cls_logits_flat, target_labels_flat, reduction='none')  # [B*H*W]
                    ce_loss = ce_loss_flat.reshape(B, H, W)  # [B, H, W]
                    
                    # Apply class-specific weights for handling class imbalance
                    class_weights_tensor = self.class_weights.to(device)  # [num_classes]
                    target_labels_flat_for_weights = target_labels.reshape(-1)  # [B*H*W]
                    weight_map = class_weights_tensor[target_labels_flat_for_weights].reshape(B, H, W)  # [B, H, W]
                    
                    cls_loss_val = (ce_loss * pos_mask.squeeze(1) * weight_map).sum()
            else:
                # Anchor format: [B, num_anchors*C, H, W] classification, [B, num_anchors, H, W] objectness
                # Reshape to [B, num_anchors, C, H, W] for easier processing
                B, AC, H, W = cls_logits.shape
                C = self.num_classes
                assert AC == num_anchors * C, f"Expected {num_anchors * C} channels, got {AC}"
                
                cls_logits_anchors = cls_logits.view(B, num_anchors, C, H, W)  # [B, A, C, H, W]
                pos_mask_anchors = (target_obj > 0.5).float()  # [B, num_anchors, H, W]
                
                # Process each anchor separately
                cls_loss_per_anchor = []
                for a in range(num_anchors):
                    anchor_pos_mask = pos_mask_anchors[:, a, :, :]  # [B, H, W]
                    if anchor_pos_mask.sum() > 0:
                        anchor_cls_logits = cls_logits_anchors[:, a, :, :, :]  # [B, C, H, W]
                        anchor_target_labels = target_cls[:, a, :, :, :].argmax(dim=1)  # [B, H, W]
                        
                        # Compute loss for this anchor
                        anchor_cls_flat = anchor_cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
                        anchor_labels_flat = anchor_target_labels.reshape(-1)
                        ce_loss_flat = F.cross_entropy(anchor_cls_flat, anchor_labels_flat, reduction='none')
                        ce_loss = ce_loss_flat.reshape(B, H, W)
                        
                        # Apply class weights
                        class_weights_tensor = self.class_weights.to(device)
                        weight_map = class_weights_tensor[anchor_labels_flat].reshape(B, H, W)
                        
                        anchor_loss = (ce_loss * anchor_pos_mask * weight_map).sum()
                        cls_loss_per_anchor.append(anchor_loss)
                
                cls_loss_val = sum(cls_loss_per_anchor) if cls_loss_per_anchor else torch.tensor(0.0, device=device)
            
            # Bbox regression loss - handle anchors
            if num_anchors == 1:
                pred_bbox = torch.sigmoid(bbox_reg)  # [B, 4, Hf, Wf]
                gt_bbox = target_bbox  # [B, 4, Hf, Wf]
                pos_mask_bbox = pos_mask
            else:
                # Anchor format: [B, num_anchors*4, H, W]
                B, A4, H, W = bbox_reg.shape
                assert A4 == num_anchors * 4, f"Expected {num_anchors * 4} channels, got {A4}"
                bbox_reg_anchors = bbox_reg.view(B, num_anchors, 4, H, W)  # [B, A, 4, H, W]
                pred_bbox = torch.sigmoid(bbox_reg_anchors)  # [B, A, 4, H, W]
                gt_bbox = target_bbox  # [B, A, 4, H, W]
                pos_mask_bbox = (target_obj > 0.5).float()  # [B, A, H, W]
            
            if self.use_giou_loss:
                # GIoU loss: extract positive cells and compute vectorized
                if num_anchors == 1:
                    # Old format: [B, 4, H, W]
                    pred_bbox_perm = pred_bbox.permute(0, 2, 3, 1)  # [B, H, W, 4]
                    gt_bbox_perm = gt_bbox.permute(0, 2, 3, 1)  # [B, H, W, 4]
                    pos_mask_perm = pos_mask_bbox.squeeze(1) > 0.5  # [B, H, W]
                else:
                    # Anchor format: [B, A, 4, H, W]
                    pred_bbox_perm = pred_bbox.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 4]
                    gt_bbox_perm = gt_bbox.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 4]
                    pos_mask_perm = pos_mask_bbox > 0.5  # [B, A, H, W]
                    # Flatten anchor dimension
                    B, A, H, W, _ = pred_bbox_perm.shape
                    pred_bbox_perm = pred_bbox_perm.reshape(B * A, H, W, 4)
                    gt_bbox_perm = gt_bbox_perm.reshape(B * A, H, W, 4)
                    pos_mask_perm = pos_mask_perm.reshape(B * A, H, W)
                
                pred_bbox_pos = pred_bbox_perm[pos_mask_perm]  # [pos_count, 4]
                gt_bbox_pos = gt_bbox_perm[pos_mask_perm]  # [pos_count, 4]
                
                if len(pred_bbox_pos) > 0:
                    # Vectorized GIoU computation for batch of boxes
                    # Convert to corners
                    pred_cx, pred_cy, pred_w, pred_h = pred_bbox_pos[:, 0], pred_bbox_pos[:, 1], pred_bbox_pos[:, 2], pred_bbox_pos[:, 3]
                    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_pos[:, 0], gt_bbox_pos[:, 1], gt_bbox_pos[:, 2], gt_bbox_pos[:, 3]
                    
                    pred_x1 = pred_cx - pred_w/2
                    pred_y1 = pred_cy - pred_h/2
                    pred_x2 = pred_cx + pred_w/2
                    pred_y2 = pred_cy + pred_h/2
                    
                    gt_x1 = gt_cx - gt_w/2
                    gt_y1 = gt_cy - gt_h/2
                    gt_x2 = gt_cx + gt_w/2
                    gt_y2 = gt_cy + gt_h/2
                    
                    # Intersection
                    inter_x1 = torch.max(pred_x1, gt_x1)
                    inter_y1 = torch.max(pred_y1, gt_y1)
                    inter_x2 = torch.min(pred_x2, gt_x2)
                    inter_y2 = torch.min(pred_y2, gt_y2)
                    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
                    
                    # Union
                    pred_area = pred_w * pred_h
                    gt_area = gt_w * gt_h
                    union_area = pred_area + gt_area - inter_area
                    
                    # IoU
                    iou = inter_area / (union_area + 1e-7)
                    
                    # GIoU
                    enclose_x1 = torch.min(pred_x1, gt_x1)
                    enclose_y1 = torch.min(pred_y1, gt_y1)
                    enclose_x2 = torch.max(pred_x2, gt_x2)
                    enclose_y2 = torch.max(pred_y2, gt_y2)
                    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
                    
                    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
                    box_loss_val = (1 - giou).sum()
                else:
                    box_loss_val = torch.tensor(0.0, device=device)
            else:
                # L1 loss: vectorized
                if num_anchors == 1:
                    box_diff = torch.abs(pred_bbox - gt_bbox)  # [B, 4, H, W]
                    box_loss_map = box_diff.sum(dim=1)  # [B, H, W]
                    box_loss_val = (box_loss_map * pos_mask_bbox.squeeze(1)).sum()
                else:
                    # Anchor format: [B, A, 4, H, W]
                    box_diff = torch.abs(pred_bbox - gt_bbox)  # [B, A, 4, H, W]
                    box_loss_map = box_diff.sum(dim=2)  # [B, A, H, W]
                    box_loss_val = (box_loss_map * pos_mask_bbox).sum()
            
            loss_cls = self.lambda_cls * cls_loss_val
            loss_box = self.lambda_box * box_loss_val
        else:
            loss_cls = torch.tensor(0.0, device=device)
            loss_box = torch.tensor(0.0, device=device)

        # Normalize losses correctly:
        # - loss_obj, loss_box, loss_cls: by pos_count (only positive cells contribute)
        # - loss_noobj: by neg_count (negative cells should be normalized separately)
        total_cells = batch_size * Hf * Wf
        neg_count = total_cells - pos_count
        
        # Normalize each component correctly
        denom_pos = max(1.0, float(pos_count))
        denom_neg = max(1.0, float(neg_count))
        
        loss_obj_norm = loss_obj / denom_pos
        loss_noobj_norm = loss_noobj / denom_neg
        loss_box_norm = loss_box / denom_pos if pos_count > 0 else torch.tensor(0.0, device=device)
        loss_cls_norm = loss_cls / denom_pos if pos_count > 0 else torch.tensor(0.0, device=device)
        
        # Total loss: combine normalized components
        loss_total = loss_obj_norm + loss_noobj_norm + loss_box_norm + loss_cls_norm

        # Log warning if many objects share cells (only on first few batches to avoid spam)
        if not hasattr(self, '_multi_object_warning_logged'):
            if cells_with_multiple_objects > 0:
                pct = (cells_with_multiple_objects / max(1, pos_count)) * 100
                if pct > 5.0:  # More than 5% of objects share cells
                    print(f"⚠️  WARNING: {cells_with_multiple_objects}/{pos_count} objects ({pct:.1f}%) share grid cells with other objects.")
                    print(f"   This is a YOLO limitation - only the last object per cell is kept. Consider:")
                    print(f"   - Using higher resolution FPN level (P2 instead of P3)")
                    print(f"   - Increasing input resolution")
                    print(f"   - Using anchor boxes for multiple detections per cell")
            self._multi_object_warning_logged = True

        return {
            'total': loss_total,
            'objectness': loss_obj_norm,
            'no_objectness': loss_noobj_norm,
            'bbox': loss_box_norm,
            'classification': loss_cls_norm,
            'num_positive': pos_count,
            'cells_with_multiple_objects': cells_with_multiple_objects
        }
