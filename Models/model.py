import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----- ResNetEncoder with multi-scale outputs for FPN -----
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, return_intermediate=False, backbone='resnet18'):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.backbone = backbone
        
        # Load ResNet backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.c3_channels = 128   # layer2 output
            self.c4_channels = 256   # layer3 output
            self.c5_channels = 512   # layer4 output
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.c3_channels = 512   # layer2 output
            self.c4_channels = 1024  # layer3 output
            self.c5_channels = 2048  # layer4 output
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet50'")
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)   # 1/4 resolution
        c3 = self.layer2(c2)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution (deepest)
        
        if self.return_intermediate:
            return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}
        return c5  # backward compatible

# ----- Feature Pyramid Network (FPN) for multi-scale detection -----
class FPN(nn.Module):
    """
    Feature Pyramid Network - combines multi-scale features for better detection
    of objects at different sizes
    Supports both ResNet18 and ResNet50
    """
    def __init__(self, c3_channels=512, c4_channels=1024, c5_channels=2048, out_channels=256):
        super().__init__()
        # Lateral connections (1x1 convs to reduce channels)
        # Automatically adapts to ResNet18 or ResNet50 based on channel sizes
        self.lateral_c5 = nn.Conv2d(c5_channels, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4_channels, out_channels, 1)  # ResNet layer3 output
        self.lateral_c3 = nn.Conv2d(c3_channels, out_channels, 1)   # ResNet layer2 output
        
        # Top-down pathway (upsampling + fusion)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, features):
        """
        features: dict with 'c3', 'c4', 'c5' from ResNetEncoder
        Returns: fused multi-scale features
        """
        c3, c4, c5 = features['c3'], features['c4'], features['c5']
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p5 = self.smooth_p5(p5)
        
        # Upsample and add (use bilinear for better quality)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.smooth_p4(p4)
        
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.smooth_p3(p3)
        
        # Return all pyramid levels for multi-scale detection
        # P3: highest resolution (small objects), P4: medium, P5: lowest resolution (large objects)
        return {'p3': p3, 'p4': p4, 'p5': p5}

# ----- CrossModalAttention (reuse minimal fusion) -----
class CrossModalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_rgb = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_thermal = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_fusion = nn.Conv2d(in_channels // 4, 2, 1)

    def forward(self, rgb_features, thermal_features):
        rgb_attn = self.conv_rgb(rgb_features)
        thermal_attn = self.conv_thermal(thermal_features)
        concat = torch.cat([rgb_attn, thermal_attn], dim=1)
        attention_weights = self.conv_fusion(concat)
        attention_weights = torch.softmax(attention_weights, dim=1)
        rgb_weight = attention_weights[:, 0:1, :, :]
        thermal_weight = attention_weights[:, 1:2, :, :]
        fused = rgb_weight * rgb_features + thermal_weight * thermal_features
        return fused, rgb_weight, thermal_weight

# ----- Anchor-based 2D detection head (YOLOv5-style with multiple anchors per cell) -----
class Detection2DHead(nn.Module):
    """
    Improved detection head with anchor boxes for better learning.
    Output per anchor:
      - objectness logits: [B, num_anchors, Hf, Wf] (raw logits)
      - classification logits: [B, num_anchors * num_classes, Hf, Wf]
      - bbox_reg: [B, num_anchors * 4, Hf, Wf] -> (cx_offset, cy_offset, w_scale, h_scale)
    
    With anchors, each cell can predict multiple objects of different sizes.
    """
    def __init__(self, in_channels=256, num_classes=3, num_anchors=3, dropout_rate=0.3, use_bn=True):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.use_bn = use_bn
        
        # Improved backbone with CSP-like structure (better feature extraction)
        # First block
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout2d(p=dropout_rate) if use_bn else None
        
        # Second block
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1)
        if use_bn:
            self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        if use_bn:
            self.bn4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(p=dropout_rate) if use_bn else None

        # Output heads - one per anchor
        # Objectness: [B, num_anchors, H, W]
        self.objectness = nn.Conv2d(128, num_anchors, 1)
        # Classification: [B, num_anchors * num_classes, H, W]
        self.classification = nn.Conv2d(128, num_anchors * num_classes, 1)
        # Bbox: [B, num_anchors * 4, H, W] - predicts offset/scale relative to anchor
        self.bbox = nn.Conv2d(128, num_anchors * 4, 1)
        
        # Initialize outputs for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize detection head outputs for better convergence"""
        # Initialize objectness to predict low confidence initially
        if self.objectness.bias is not None:
            nn.init.constant_(self.objectness.bias, -2.0)  # sigmoid(-2.0) ≈ 0.12
        
        # Initialize bbox to predict small offsets
        nn.init.normal_(self.bbox.weight, mean=0.0, std=0.01)
        if self.bbox.bias is not None:
            nn.init.constant_(self.bbox.bias, 0.0)

    def forward(self, x):
        # First block with residual-like structure
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = F.relu(out)
        if self.use_bn and self.dropout1 is not None:
            out = self.dropout1(out)
        
        # Second block
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        if self.use_bn:
            out = self.bn4(out)
        out = F.relu(out)
        if self.use_bn and self.dropout2 is not None:
            out = self.dropout2(out)

        # Generate predictions
        obj_logits = self.objectness(out)  # [B, num_anchors, H, W]
        cls_logits = self.classification(out)  # [B, num_anchors * num_classes, H, W]
        bbox_reg = self.bbox(out)  # [B, num_anchors * 4, H, W]

        return {
            'objectness': obj_logits,
            'classification': cls_logits,
            'bbox': bbox_reg
        }

# ----- Full model combining encoders + FPN + fusion + detection head -----
class ThermalRGB2DNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, use_bn=False, use_fpn=True, backbone='resnet18', 
                 num_anchors=3, use_multiscale=True):
        super().__init__()
        self.use_fpn = use_fpn
        self.backbone = backbone
        self.num_anchors = num_anchors
        self.use_multiscale = use_multiscale and use_fpn  # Multi-scale only works with FPN
        
        # Create encoders with specified backbone
        self.rgb_encoder = ResNetEncoder(pretrained=pretrained, return_intermediate=use_fpn, backbone=backbone)
        self.thermal_encoder = ResNetEncoder(pretrained=pretrained, return_intermediate=use_fpn, backbone=backbone)
        
        if use_fpn:
            # Get channel sizes from encoder (they're the same for both encoders)
            c3_channels = self.rgb_encoder.c3_channels
            c4_channels = self.rgb_encoder.c4_channels
            c5_channels = self.rgb_encoder.c5_channels
            
            self.rgb_fpn = FPN(c3_channels=c3_channels, c4_channels=c4_channels, c5_channels=c5_channels, out_channels=256)
            self.thermal_fpn = FPN(c3_channels=c3_channels, c4_channels=c4_channels, c5_channels=c5_channels, out_channels=256)
            fusion_channels = 256
            
            if self.use_multiscale:
                # Multi-scale detection: separate heads for P3, P4, P5
                self.fusion_p3 = CrossModalAttention(in_channels=fusion_channels)
                self.fusion_p4 = CrossModalAttention(in_channels=fusion_channels)
                self.fusion_p5 = CrossModalAttention(in_channels=fusion_channels)
                self.head_p3 = Detection2DHead(in_channels=fusion_channels, num_classes=num_classes, 
                                              num_anchors=num_anchors, dropout_rate=0.3, use_bn=use_bn)
                self.head_p4 = Detection2DHead(in_channels=fusion_channels, num_classes=num_classes, 
                                              num_anchors=num_anchors, dropout_rate=0.3, use_bn=use_bn)
                self.head_p5 = Detection2DHead(in_channels=fusion_channels, num_classes=num_classes, 
                                              num_anchors=num_anchors, dropout_rate=0.3, use_bn=use_bn)
            else:
                # Single-scale: use P3 only (backward compatible)
                self.fusion = CrossModalAttention(in_channels=fusion_channels)
                self.head = Detection2DHead(in_channels=fusion_channels, num_classes=num_classes, 
                                           num_anchors=num_anchors, dropout_rate=0.3, use_bn=use_bn)
        else:
            # Use deepest feature channels (no FPN)
            fusion_channels = self.rgb_encoder.c5_channels
            self.fusion = CrossModalAttention(in_channels=fusion_channels)
            self.head = Detection2DHead(in_channels=fusion_channels, num_classes=num_classes, 
                                      num_anchors=num_anchors, dropout_rate=0.3, use_bn=use_bn)

    def forward(self, rgb, thermal, return_attention=False):
        if self.use_fpn:
            rgb_features_dict = self.rgb_encoder(rgb)
            thermal_features_dict = self.thermal_encoder(thermal)
            rgb_fpn_output = self.rgb_fpn(rgb_features_dict)
            thermal_fpn_output = self.thermal_fpn(thermal_features_dict)
            
            if self.use_multiscale:
                # Multi-scale detection: process P3, P4, P5 separately
                preds_p3 = self._forward_single_scale(rgb_fpn_output['p3'], thermal_fpn_output['p3'], 
                                                      self.fusion_p3, self.head_p3, 'p3')
                preds_p4 = self._forward_single_scale(rgb_fpn_output['p4'], thermal_fpn_output['p4'], 
                                                      self.fusion_p4, self.head_p4, 'p4')
                preds_p5 = self._forward_single_scale(rgb_fpn_output['p5'], thermal_fpn_output['p5'], 
                                                      self.fusion_p5, self.head_p5, 'p5')
                
                # Combine multi-scale predictions: concatenate along channel dimension (same as YOLO model)
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
                preds = {
                    'objectness': torch.cat([preds_p3['objectness'], obj_p4_up, obj_p5_up], dim=1),  # [B, anchors*3, H, W]
                    'classification': torch.cat([preds_p3['classification'], cls_p4_up, cls_p5_up], dim=1),  # [B, anchors*3*C, H, W]
                    'bbox': torch.cat([preds_p3['bbox'], bbox_p4_up, bbox_p5_up], dim=1)  # [B, anchors*3*4, H, W]
                }
                
                if return_attention:
                    preds['rgb_attention'] = {'p3': preds_p3.get('rgb_attention'), 'p4': preds_p4.get('rgb_attention'), 'p5': preds_p5.get('rgb_attention')}  # pyright: ignore[reportArgumentType]
                    preds['thermal_attention'] = {'p3': preds_p3.get('thermal_attention'), 'p4': preds_p4.get('thermal_attention'), 'p5': preds_p5.get('thermal_attention')}  # pyright: ignore[reportArgumentType]
            else:
                # Single-scale: use P3 only (backward compatible)
                rgb_features = rgb_fpn_output if isinstance(rgb_fpn_output, torch.Tensor) else rgb_fpn_output['p3']
                thermal_features = thermal_fpn_output if isinstance(thermal_fpn_output, torch.Tensor) else thermal_fpn_output['p3']
                fused, rgb_w, thermal_w = self.fusion(rgb_features, thermal_features)
                preds = self.head(fused)
                if return_attention:
                    preds['rgb_attention'] = rgb_w
                    preds['thermal_attention'] = thermal_w
        else:
            rgb_features = self.rgb_encoder(rgb)
            thermal_features = self.thermal_encoder(thermal)
            # Ensure same shape for fusion
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
    
    def _forward_single_scale(self, rgb_feat, thermal_feat, fusion_module, head_module, scale_name):
        """Helper to process a single FPN scale"""
        fused, rgb_w, thermal_w = fusion_module(rgb_feat, thermal_feat)
        preds = head_module(fused)
        preds['rgb_attention'] = rgb_w
        preds['thermal_attention'] = thermal_w
        preds['scale'] = scale_name
        return preds

# quick test when run as script
if __name__ == "__main__":
    model = ThermalRGB2DNet(num_classes=3, pretrained=False)
    x = torch.randn(2,3,512,640)
    t = torch.randn(2,3,512,640)
    out = model(x,t, return_attention=True)
    print({k: v.shape for k,v in out.items() if isinstance(v, torch.Tensor)})
