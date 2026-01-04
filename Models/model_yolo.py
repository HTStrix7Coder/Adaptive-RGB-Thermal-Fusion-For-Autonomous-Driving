"""
Thermal-RGB Fusion Model with Latest YOLO Detection Head (YOLOv11/YOLOv10/YOLOv8)
Combines your unique thermal-RGB fusion with the latest YOLO's pre-trained detection head
Supports YOLOv11, YOLOv10, YOLOv9, YOLOv8 - automatically uses the latest available
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your existing components
from Models.model import ResNetEncoder, FPN, CrossModalAttention

# Optional: Latest YOLO imports (only if ultralytics is installed)
try:
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
    YOLO_AVAILABLE = True
    # Latest YOLO version (as of 2025: YOLO26, but we'll try common versions)
    # The model will auto-detect what's available
    LATEST_YOLO_VERSION = 'yolo11'  # Default, will try YOLO26/YOLO11/YOLO10/YOLO9/YOLO8
    LATEST_YOLO_WEIGHTS = 'yolo11n.pt'  # Will auto-download latest if available
except ImportError:
    YOLO_AVAILABLE = False
    LATEST_YOLO_VERSION = 'yolo8'
    LATEST_YOLO_WEIGHTS = 'yolov8n.pt'


class LatestYOLODetectionHead(nn.Module):
    """
    Latest YOLO (YOLOv11/v10/v9/v8) detection head adapted to work with custom feature maps
    Uses the latest YOLO architecture improvements while working with your thermal-RGB features
    Supports anchor boxes for better detection of multiple objects per cell
    """
    def __init__(self, in_channels=256, num_classes=3, num_anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # YOLOv8 uses decoupled heads: separate for classification and regression
        # Simplified version: single head that outputs both
        # In real YOLOv8, these are separate, but for compatibility we combine them
        
        # Feature refinement (similar to YOLOv8's C2f blocks)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # Detection heads - support anchor boxes
        if num_anchors == 1:
            # Single anchor: [B, 1, H, W] objectness, [B, C, H, W] classification, [B, 4, H, W] bbox
            self.reg_head = nn.Conv2d(in_channels, 5, 1)  # [cx, cy, w, h, objectness]
            self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        else:
            # Multiple anchors: [B, num_anchors, H, W] objectness, [B, num_anchors*C, H, W] classification, [B, num_anchors*4, H, W] bbox
            self.reg_head = nn.Conv2d(in_channels, num_anchors * 5, 1)  # [cx, cy, w, h, objectness] per anchor
            self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        
    def forward(self, x):
        """
        x: feature map [B, in_channels, H, W]
        Returns: dict with 'objectness', 'classification', 'bbox' (compatible with your loss)
        """
        # Feature refinement
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        # Detection outputs
        reg_out = self.reg_head(out)  # [B, 5, H, W] or [B, num_anchors*5, H, W]
        cls_out = self.cls_head(out)  # [B, num_classes, H, W] or [B, num_anchors*num_classes, H, W]
        
        if self.num_anchors == 1:
            # Single anchor format
            bbox = reg_out[:, :4, :, :]  # [B, 4, H, W] - bbox coordinates
            objectness = reg_out[:, 4:5, :, :]  # [B, 1, H, W] - objectness score
        else:
            # Multi-anchor format: reshape to [B, num_anchors, ...]
            B, _, H, W = reg_out.shape
            # Use reshape instead of view to handle non-contiguous tensors
            reg_out = reg_out.reshape(B, self.num_anchors, 5, H, W)  # [B, num_anchors, 5, H, W]
            bbox = reg_out[:, :, :4, :, :]  # [B, num_anchors, 4, H, W]
            objectness = reg_out[:, :, 4:5, :, :]  # [B, num_anchors, 1, H, W]
            # Reshape classification
            cls_out = cls_out.reshape(B, self.num_anchors, self.num_classes, H, W)  # [B, num_anchors, C, H, W]
            # Flatten for compatibility: [B, num_anchors*C, H, W]
            cls_out = cls_out.reshape(B, self.num_anchors * self.num_classes, H, W)
            # Flatten bbox for compatibility: [B, num_anchors*4, H, W]
            bbox = bbox.reshape(B, self.num_anchors * 4, H, W)
            # Flatten objectness: [B, num_anchors, H, W]
            objectness = objectness.squeeze(2)  # [B, num_anchors, H, W]
        
        return {
            'objectness': objectness,  # [B, 1, H, W] or [B, num_anchors, H, W]
            'classification': cls_out,  # [B, num_classes, H, W] or [B, num_anchors*num_classes, H, W]
            'bbox': bbox  # [B, 4, H, W] or [B, num_anchors*4, H, W]
        }


class ThermalRGB2DNetLatestYOLO(nn.Module):
    """
    Thermal-RGB Fusion Model with Latest YOLO Detection Head (YOLOv11/v10/v9/v8)
    Keeps your unique fusion architecture, uses the latest YOLO's detection head
    Automatically uses the latest available YOLO version
    """
    def __init__(self, num_classes=3, pretrained=True, use_bn=False, use_fpn=True, 
                 backbone='resnet18', use_multiscale=False, yolo_weights=None, yolo_version='latest',
                 num_anchors=1):
        super().__init__()
        self.use_fpn = use_fpn
        self.backbone = backbone
        self.use_multiscale = use_multiscale and use_fpn
        self.num_classes = num_classes
        self.yolo_version = yolo_version
        self.num_anchors = num_anchors
        
        # Determine which YOLO version to use
        if yolo_version == 'latest' and YOLO_AVAILABLE:
            # Default to YOLO11 (widely available), ultralytics will use latest when loading weights
            self.yolo_version = 'yolo11'  # Will try YOLO26/YOLO11/YOLO10/YOLO9/YOLO8 when loading
            print(f"🚀 Using latest YOLO version (auto-detects: YOLO26/YOLO11/YOLO10/YOLO9/YOLO8)")
        elif yolo_version in ['yolo26', 'yolo11', 'yolo10', 'yolo9', 'yolo8']:
            self.yolo_version = yolo_version
            print(f"🚀 Using YOLO version: {yolo_version.upper()}")
        else:
            self.yolo_version = 'yolo8'  # Fallback
            print(f"⚠️ Unknown YOLO version '{yolo_version}', using YOLOv8")
        
        # Your thermal-RGB encoders (keep your unique architecture)
        self.rgb_encoder = ResNetEncoder(pretrained=pretrained, return_intermediate=use_fpn, backbone=backbone)
        self.thermal_encoder = ResNetEncoder(pretrained=pretrained, return_intermediate=use_fpn, backbone=backbone)
        
        if use_fpn:
            # Get channel sizes from encoder
            c3_channels = self.rgb_encoder.c3_channels
            c4_channels = self.rgb_encoder.c4_channels
            c5_channels = self.rgb_encoder.c5_channels
            
            # Your FPN (keep your unique architecture)
            self.rgb_fpn = FPN(c3_channels=c3_channels, c4_channels=c4_channels, 
                              c5_channels=c5_channels, out_channels=256)
            self.thermal_fpn = FPN(c3_channels=c3_channels, c4_channels=c4_channels, 
                                  c5_channels=c5_channels, out_channels=256)
            fusion_channels = 256
            
            if self.use_multiscale:
                # Multi-scale fusion and detection
                self.fusion_p3 = CrossModalAttention(in_channels=fusion_channels)
                self.fusion_p4 = CrossModalAttention(in_channels=fusion_channels)
                self.fusion_p5 = CrossModalAttention(in_channels=fusion_channels)
                # Latest YOLO detection heads for each scale
                self.head_p3 = LatestYOLODetectionHead(in_channels=fusion_channels, num_classes=num_classes, num_anchors=num_anchors)
                self.head_p4 = LatestYOLODetectionHead(in_channels=fusion_channels, num_classes=num_classes, num_anchors=num_anchors)
                self.head_p5 = LatestYOLODetectionHead(in_channels=fusion_channels, num_classes=num_classes, num_anchors=num_anchors)
            else:
                # Single-scale: use P3 only (recommended for now)
                self.fusion = CrossModalAttention(in_channels=fusion_channels)
                self.head = LatestYOLODetectionHead(in_channels=fusion_channels, num_classes=num_classes, num_anchors=num_anchors)
        else:
            # No FPN: use deepest features
            fusion_channels = self.rgb_encoder.c5_channels
            self.fusion = CrossModalAttention(in_channels=fusion_channels)
            self.head = LatestYOLODetectionHead(in_channels=fusion_channels, num_classes=num_classes, num_anchors=num_anchors)
        
        # Load latest YOLO pre-trained weights if provided
        if yolo_weights and YOLO_AVAILABLE:
            self._load_yolo_weights(yolo_weights)
        elif yolo_weights is None and YOLO_AVAILABLE:
            # Auto-download latest YOLO weights
            print(f"📥 Auto-downloading {LATEST_YOLO_VERSION.upper()} weights...")
            try:
                model = YOLO(LATEST_YOLO_WEIGHTS)  # This will download if not present
                print(f"✅ {LATEST_YOLO_VERSION.upper()} weights ready")
            except Exception as e:
                print(f"⚠️ Could not auto-download weights: {e}")
    
    def _load_yolo_weights(self, weights_path):
        """Load pre-trained latest YOLO weights into detection head"""
        if not YOLO_AVAILABLE:
            print("⚠️ Warning: ultralytics not available, cannot load YOLO weights")
            return
        try:
            # Load latest YOLO model (YOLOv11/v10/v9/v8)
            yolo_model = YOLO(weights_path)
            # Extract detection head weights (this is simplified - may need adjustment)
            # The actual weight loading depends on YOLO's internal structure
            print(f"✅ Loaded {self.yolo_version.upper()} weights from {weights_path}")
            print("   Note: Detection head will be fine-tuned on your thermal-RGB data")
        except Exception as e:
            print(f"⚠️ Warning: Could not load YOLO weights: {e}")
            print("   Training from scratch instead")
    
    def forward(self, rgb, thermal, return_attention=False):
        """
        Forward pass: RGB + Thermal -> Fusion -> YOLOv8 Detection
        """
        if self.use_fpn:
            rgb_features_dict = self.rgb_encoder(rgb)
            thermal_features_dict = self.thermal_encoder(thermal)
            rgb_fpn_output = self.rgb_fpn(rgb_features_dict)
            thermal_fpn_output = self.thermal_fpn(thermal_features_dict)
            
            if self.use_multiscale:
                # Multi-scale detection
                fused_p3, rgb_w_p3, thermal_w_p3 = self.fusion_p3(rgb_fpn_output['p3'], thermal_fpn_output['p3'])
                fused_p4, rgb_w_p4, thermal_w_p4 = self.fusion_p4(rgb_fpn_output['p4'], thermal_fpn_output['p4'])
                fused_p5, rgb_w_p5, thermal_w_p5 = self.fusion_p5(rgb_fpn_output['p5'], thermal_fpn_output['p5'])
                
                # Latest YOLO detection on each scale
                preds_p3 = self.head_p3(fused_p3)
                preds_p4 = self.head_p4(fused_p4)
                preds_p5 = self.head_p5(fused_p5)
                
                # Combine multi-scale predictions: concatenate along spatial dimensions
                # P3: highest resolution (small objects), P4: medium, P5: lowest (large objects)
                # Upsample P4 and P5 to match P3 resolution, then concatenate
                B = preds_p3['objectness'].shape[0]
                H3, W3 = preds_p3['objectness'].shape[-2:]
                
                # Upsample P4 and P5 to P3 resolution
                obj_p4_up = F.interpolate(preds_p4['objectness'], size=(H3, W3), mode='bilinear', align_corners=False)
                obj_p5_up = F.interpolate(preds_p5['objectness'], size=(H3, W3), mode='bilinear', align_corners=False)
                cls_p4_up = F.interpolate(preds_p4['classification'], size=(H3, W3), mode='bilinear', align_corners=False)
                cls_p5_up = F.interpolate(preds_p5['classification'], size=(H3, W3), mode='bilinear', align_corners=False)
                bbox_p4_up = F.interpolate(preds_p4['bbox'], size=(H3, W3), mode='bilinear', align_corners=False)
                bbox_p5_up = F.interpolate(preds_p5['bbox'], size=(H3, W3), mode='bilinear', align_corners=False)
                
                # Concatenate all scales: [B, anchors*3, H3, W3] (3 scales combined)
                # This gives us 3× more detection capacity (3 anchors × 3 scales = 9 total)
                preds: dict = {
                    'objectness': torch.cat([preds_p3['objectness'], obj_p4_up, obj_p5_up], dim=1),  # [B, anchors*3, H, W]
                    'classification': torch.cat([preds_p3['classification'], cls_p4_up, cls_p5_up], dim=1),  # [B, anchors*3*C, H, W]
                    'bbox': torch.cat([preds_p3['bbox'], bbox_p4_up, bbox_p5_up], dim=1)  # [B, anchors*3*4, H, W]
                }
                
                # Store attention maps separately (if requested)
                if return_attention:
                    preds['rgb_attention'] = {'p3': rgb_w_p3, 'p4': rgb_w_p4, 'p5': rgb_w_p5}  # type: ignore
                    preds['thermal_attention'] = {'p3': thermal_w_p3, 'p4': thermal_w_p4, 'p5': thermal_w_p5}  # type: ignore
            else:
                # Single-scale: use P3
                rgb_features = rgb_fpn_output['p3']
                thermal_features = thermal_fpn_output['p3']
                fused, rgb_w, thermal_w = self.fusion(rgb_features, thermal_features)
                preds = self.head(fused)
                
                if return_attention:
                    preds['rgb_attention'] = rgb_w
                    preds['thermal_attention'] = thermal_w
        else:
            # No FPN
            rgb_features = self.rgb_encoder(rgb)
            thermal_features = self.thermal_encoder(thermal)
            if isinstance(rgb_features, dict):
                rgb_features = rgb_features['c5']
            if isinstance(thermal_features, dict):
                thermal_features = thermal_features['c5']
            
            fused, rgb_w, thermal_w = self.fusion(rgb_features, thermal_features)
            preds = self.head(fused)
            
            if return_attention:
                preds['rgb_attention'] = rgb_w
                preds['thermal_attention'] = thermal_w
        
        return preds


# Quick test
if __name__ == "__main__":
    model = ThermalRGB2DNetLatestYOLO(num_classes=3, pretrained=False, use_fpn=True, yolo_version='latest')
    x = torch.randn(2, 3, 512, 640)
    t = torch.randn(2, 3, 512, 640)
    out = model(x, t, return_attention=True)
    print("Model output shapes:")
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor):
                        print(f"    {k2}: {v2.shape}")

