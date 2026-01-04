import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import numpy as np
from torch.cuda.amp import GradScaler, autocast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.model import ThermalRGB2DNet
from Models.dataset import create_dataloaders
from utils.annotations import get_annotations_for_split
from utils.loss_2d import YOLO2DLoss
from Models.model_yolo import ThermalRGB2DNetLatestYOLO
LATEST_YOLO_AVAILABLE = True

def prepare_batch_targets(names, annotations, img_h=512, img_w=640):
    """
    build list of target dicts for each image in batch:
      each dict: {'boxes': np.array([[x,y,w,h],...]), 'labels': np.array([...]), 'img_h':..., 'img_w':...}
    """
    batch_targets = []
    for name in names:
        # normalize filename to key used by annotations.py
        if '/' in name or '\\' in name:
            key = os.path.splitext(os.path.basename(name))[0]
        else:
            key = os.path.splitext(name)[0]
        ann = annotations.get(key, None)
        if ann is None or len(ann.get('boxes', [])) == 0:
            batch_targets.append({'boxes': np.empty((0,4)), 'labels': np.empty((0,)), 'img_h': img_h, 'img_w': img_w})
        else:
            batch_targets.append({'boxes': ann['boxes'], 'labels': ann['labels'], 'img_h': img_h, 'img_w': img_w})
    return batch_targets

def calculate_brightness(rgb_tensor):
    """
    Calculate brightness from RGB tensor (normalized 0-1 range).
    Returns brightness score in [0, 1] range.
    Based on observations: nighttime with flashes ~0.40-0.50, actual daytime ~0.75
    """
    # Denormalize if needed (assuming ImageNet normalization)
    # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
    if rgb_tensor.shape[1] == 3:  # [B, C, H, W]
        # Weighted grayscale conversion
        gray = 0.299 * rgb_tensor[:, 0:1, :, :] + 0.587 * rgb_tensor[:, 1:2, :, :] + 0.114 * rgb_tensor[:, 2:3, :, :]
    else:
        gray = rgb_tensor
    
    # Calculate mean brightness per image in batch
    # Average over spatial dimensions
    brightness = gray.mean(dim=[2, 3]).squeeze(1)  # [B]
    
    # Denormalize from ImageNet normalization if needed
    # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
    # Approximate denormalization: (normalized * std) + mean
    # For grayscale approximation: use average of means and stds
    brightness = brightness * 0.226 + 0.449  # Approximate denormalization
    
    # Clamp to [0, 1] range
    brightness = torch.clamp(brightness, 0.0, 1.0)
    
    return brightness

def brightness_aware_attention_loss(rgb_attention, thermal_attention, brightness, 
                                     brightness_threshold=0.55, lambda_brightness=0.1):
    """
    Regularization loss that encourages RGB trust in bright scenes and thermal trust in dark scenes.
    
    Args:
        rgb_attention: RGB attention map [B, 1, H, W] or dict with 'p3', 'p4', 'p5'
        thermal_attention: Thermal attention map [B, 1, H, W] or dict with 'p3', 'p4', 'p5'
        brightness: Brightness scores [B] in [0, 1] range
        brightness_threshold: Threshold above which scene is considered bright (default 0.55)
        lambda_brightness: Weight for brightness regularization (default 0.1)
    
    Returns:
        Regularization loss scalar
    """
    # Extract attention maps (handle multi-scale dict format)
    if isinstance(rgb_attention, dict):
        # Use P3 level for regularization (highest resolution)
        rgb_attn = rgb_attention.get('p3', list(rgb_attention.values())[0])
        thermal_attn = thermal_attention.get('p3', list(thermal_attention.values())[0])
    else:
        rgb_attn = rgb_attention
        thermal_attn = thermal_attention
    
    # Get mean attention per image in batch [B]
    rgb_mean = rgb_attn.mean(dim=[1, 2, 3])  # [B]
    thermal_mean = thermal_attn.mean(dim=[1, 2, 3])  # [B]
    
    # Normalize to get percentages
    total = rgb_mean + thermal_mean + 1e-6
    rgb_pct = rgb_mean / total  # [B]
    thermal_pct = thermal_mean / total  # [B]
    
    # Determine which images are bright (daytime)
    is_bright = brightness > brightness_threshold  # [B]
    
    # Expected RGB percentage: higher in bright scenes
    # Bright scenes (>0.55): expect ~60% RGB, 40% Thermal
    # Dark scenes (<=0.55): expect ~30% RGB, 70% Thermal
    expected_rgb_bright = 0.6
    expected_rgb_dark = 0.3
    
    # Calculate deviation from expected trust
    expected_rgb = torch.where(is_bright, 
                               torch.tensor(expected_rgb_bright, device=brightness.device),
                               torch.tensor(expected_rgb_dark, device=brightness.device))
    
    # Penalize deviation from expected RGB trust
    rgb_deviation = torch.abs(rgb_pct - expected_rgb)
    
    # Weight by brightness: stronger penalty in bright scenes
    # Bright scenes get higher weight to encourage RGB trust
    brightness_weight = torch.where(is_bright, 
                                   torch.tensor(1.5, device=brightness.device),  # Stronger in bright
                                   torch.tensor(0.5, device=brightness.device))   # Weaker in dark
    
    # Calculate weighted loss
    loss = (rgb_deviation * brightness_weight).mean()
    
    return lambda_brightness * loss

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_loader, val_loader = create_dataloaders(config['batch_size'], config['num_workers'],
                                                  (config['img_height'], config['img_width']))
    
    if train_loader is None or val_loader is None:
        print("❌ Failed to create dataloaders")
        return
    
    # Add dataset size verification
    train_size = len(train_loader.dataset)  # type: ignore
    val_size = len(val_loader.dataset)  # type: ignore
    train_batches = len(train_loader)
    val_batches = len(val_loader)

    print(f"\n📊 Dataset Info:")
    print(f"   Training: {train_size:,} samples, {train_batches:,} batches (batch_size={config['batch_size']})")
    print(f"   Validation: {val_size:,} samples, {val_batches:,} batches")
    print(f"   Expected time per epoch: ~{train_batches * 0.25 / 60:.1f}-{train_batches * 0.30 / 60:.1f} minutes")
    
    train_annos = get_annotations_for_split('train')
    val_annos = get_annotations_for_split('val')

    # Choose model architecture: 'custom' or 'yolo' (uses latest YOLO version)
    model_type = config.get('model_type', 'yolo')  # Default to latest YOLO for better performance
    
    use_bn = config.get('use_bn', True)  # Enable BatchNorm by default for new training
    use_fpn = config.get('use_fpn', True)  # Enable FPN for multi-scale detection
    backbone = config.get('backbone', 'resnet18')  # 'resnet18' or 'resnet50' (default: resnet18 for better data efficiency)
    use_multiscale = config.get('use_multiscale', False)  # Multi-scale detection (P3, P4, P5) - enabled in config
    yolo_weights = config.get('yolo_weights', None)  # Path to YOLO pre-trained weights (optional, auto-downloads latest if None)
    yolo_version = config.get('yolo_version', 'latest')  # 'latest', 'yolo11', 'yolo10', 'yolo9', 'yolo8'
    
    if model_type == 'yolo' and LATEST_YOLO_AVAILABLE:
        print("🚀 Using Latest YOLO Detection Head with Thermal-RGB Fusion")
        num_anchors = config.get('num_anchors', 1)  # Support anchors for YOLO model too
        model = ThermalRGB2DNetLatestYOLO(
            num_classes=config['num_classes'], 
            pretrained=config['pretrained'], 
            use_bn=use_bn,
            use_fpn=use_fpn,
            backbone=backbone,
            use_multiscale=use_multiscale,
            yolo_weights=yolo_weights,
            yolo_version=yolo_version,
            num_anchors=num_anchors
        ).to(device)
    else:
        if model_type == 'yolo' and not LATEST_YOLO_AVAILABLE:
            print("⚠️ Warning: Latest YOLO requested but ultralytics not available. Using custom model.")
            print("   Install with: pip install ultralytics")
        print("🔧 Using Custom Detection Head with Thermal-RGB Fusion")
        num_anchors = config.get('num_anchors', 1)  # Number of anchor boxes per cell
        model = ThermalRGB2DNet(
            num_classes=config['num_classes'], 
            pretrained=config['pretrained'], 
            use_bn=use_bn,
            use_fpn=use_fpn,
            backbone=backbone,
            num_anchors=num_anchors,
            use_multiscale=use_multiscale
        ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model: {backbone.upper()} backbone with FPN={use_fpn}, BatchNorm={use_bn}")
    if model_type == 'yolo':
        yolo_ver = config.get('yolo_version', 'latest')
        print(f"   Architecture: Latest YOLO Detection Head + Thermal-RGB Fusion")
        if hasattr(model, 'yolo_version'):
            yolo_ver_str = str(model.yolo_version)
            print(f"   YOLO Version: {yolo_ver_str.upper()}")
    else:
        num_anchors = config.get('num_anchors', 1)
        print(f"   Architecture: Custom Detection Head + Thermal-RGB Fusion (Anchors={num_anchors})")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Parameters per training sample: {trainable_params / 8347:.1f} (lower is better for small datasets)")
    if use_bn:
        print("✅ BatchNorm and Dropout enabled in detection head (prevents overfitting)")
    if use_fpn:
        print("✅ Feature Pyramid Network (FPN) enabled (multi-scale detection for better performance)")
    if model_type == 'custom' and config.get('num_anchors', 1) > 1:
        print(f"✅ Anchor boxes enabled ({config.get('num_anchors', 1)} anchors per cell) - allows multiple objects per cell")
    if use_multiscale:
        print("✅ Multi-scale detection enabled (P3, P4, P5) - better for objects of different sizes")
    if model_type == 'yolo':
        num_anchors = config.get('num_anchors', 1)
        if num_anchors > 1:
            print(f"✅ Latest YOLO detection head with {num_anchors} anchor boxes per cell (better multi-object detection)")
        else:
            print("✅ Latest YOLO detection head (pre-trained architecture, better performance expected)")
    
    # Enable mixed precision training (FP16) for RTX 4060 Ti - ~2x speedup
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print("✅ Mixed precision (FP16) training enabled for faster training")
    
    # Optional: torch.compile for PyTorch 2.0+ (additional ~20-30% speedup)
    if config.get('use_compile', False) and hasattr(torch, 'compile'):
        print("✅ torch.compile enabled for additional speedup")
        model = torch.compile(model, mode='reduce-overhead')

    # Improved loss with Focal Loss and GIoU
    use_focal = config.get('use_focal_loss', True)
    use_giou = config.get('use_giou_loss', True)
    # Class weights: [car, person, bicycle] - bicycle gets 2x weight due to class imbalance
    class_weights = config.get('class_weights', [1.0, 1.0, 2.0])
    # Loss weights: Need to balance objectness vs background suppression
    # lambda_noobj must be MUCH higher to prevent overconfidence
    lambda_obj = config.get('lambda_obj', 0.5)  # Reduced from 1.0
    lambda_noobj = config.get('lambda_noobj', 2.0)  # Increased from 0.5 to 2.0 (4x increase)
    lambda_box = config.get('lambda_box', 5.0)
    lambda_cls = config.get('lambda_cls', 1.0)
    
    num_anchors = config.get('num_anchors', 1)
    use_multiscale = config.get('use_multiscale', False)
    criterion = YOLO2DLoss(
        num_classes=config['num_classes'],
        lambda_obj=lambda_obj,
        lambda_noobj=lambda_noobj,
        lambda_box=lambda_box,
        lambda_cls=lambda_cls,
        use_focal_loss=use_focal,
        use_giou_loss=use_giou,
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        class_weights=class_weights,
        num_anchors=num_anchors,  # Pass num_anchors to loss function
        use_multiscale=use_multiscale  # Inform loss function about multi-scale concatenation
    )
    print(f"✅ Loss weights: lambda_obj={lambda_obj}, lambda_noobj={lambda_noobj}, lambda_box={lambda_box}, lambda_cls={lambda_cls}")
    if use_focal:
        print("✅ Focal Loss enabled (better handling of class imbalance)")
    if use_giou:
        print("✅ GIoU Loss enabled (better bbox localization)")
    if class_weights != [1.0, 1.0, 1.0]:
        print(f"✅ Class weights enabled: {class_weights} (bicycle gets {class_weights[2]}x weight for class imbalance)")
    
    # Brightness-aware regularization
    use_brightness_reg = config.get('use_brightness_regularization', True)
    if use_brightness_reg:
        brightness_threshold = config.get('brightness_threshold', 0.55)
        lambda_brightness = config.get('lambda_brightness', 0.1)
        print(f"✅ Brightness-aware attention regularization enabled")
        print(f"   Threshold: {brightness_threshold} (brightness > {brightness_threshold} = daytime)")
        print(f"   Weight: {lambda_brightness} (encourages RGB trust in bright scenes, thermal in dark)")
        print(f"   Expected: Bright scenes → 60% RGB, Dark scenes → 30% RGB")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Better scheduler: warmup + cosine annealing
    warmup_epochs = config.get('warmup_epochs', 3)
    total_epochs = config['num_epochs']
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"✅ Learning rate warmup ({warmup_epochs} epochs) + cosine annealing enabled")

    checkpoint_dir = f"checkpoints/{config['experiment_name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val = float('inf')
    patience = config.get('early_stopping_patience', 15)  # Increased for new architecture (was 10)
    min_epochs = config.get('min_epochs', 20)  # Minimum epochs before early stopping can trigger
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        train_loss = 0.0
        brightness_reg_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", ncols=120)
        for batch in pbar:
            # Non-blocking transfer for async GPU upload (faster)
            rgb = batch['rgb'].to(device, non_blocking=True)
            thermal = batch['thermal'].to(device, non_blocking=True)
            names = batch['name']

            targets = prepare_batch_targets(names, train_annos, config['img_height'], config['img_width'])

            optimizer.zero_grad()
            
            # Calculate brightness for brightness-aware regularization
            brightness = calculate_brightness(rgb)
            
            # Mixed precision forward pass
            # Get attention maps for brightness-aware regularization
            use_brightness_reg = config.get('use_brightness_regularization', True)
            return_attention = use_brightness_reg
            
            with autocast(enabled=use_amp):
                preds = model(rgb, thermal, return_attention=return_attention)
                loss_dict = criterion(preds, targets)
                loss = loss_dict['total']
                
                # Add brightness-aware attention regularization
                if use_brightness_reg and 'rgb_attention' in preds and 'thermal_attention' in preds:
                    brightness_loss = brightness_aware_attention_loss(
                        preds['rgb_attention'],
                        preds['thermal_attention'],
                        brightness,
                        brightness_threshold=config.get('brightness_threshold', 0.55),
                        lambda_brightness=config.get('lambda_brightness', 0.1)
                    )
                    loss = loss + brightness_loss
                    brightness_reg_loss += float(brightness_loss.item())
            
            # Skip backward if loss is invalid
            if not torch.isfinite(loss):
                steps += 1
                pbar.set_postfix({'loss': f'{(train_loss/max(1,steps)):.4f}', 'status': 'skipped'})
                continue
            
            # Mixed precision backward pass
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += float(loss.item())
            steps += 1
            pbar.set_postfix({'loss': f'{(train_loss/steps):.4f}'})

        train_loss /= max(1, steps)
        brightness_reg_loss /= max(1, steps)

        # validation
        model.eval()
        val_loss = 0.0
        val_brightness_reg_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", ncols=120)
            for batch in pbar:
                # Non-blocking transfer for async GPU upload
                rgb = batch['rgb'].to(device, non_blocking=True)
                thermal = batch['thermal'].to(device, non_blocking=True)
                names = batch['name']
                targets = prepare_batch_targets(names, val_annos, config['img_height'], config['img_width'])
                
                # Mixed precision inference (faster validation)
                # Calculate brightness for monitoring
                brightness = calculate_brightness(rgb)
                use_brightness_reg = config.get('use_brightness_regularization', True)
                
                with autocast(enabled=use_amp):
                    preds = model(rgb, thermal, return_attention=use_brightness_reg)
                    loss_dict = criterion(preds, targets)
                    
                    # Add brightness-aware attention regularization for validation monitoring
                    if use_brightness_reg and 'rgb_attention' in preds and 'thermal_attention' in preds:
                        brightness_loss = brightness_aware_attention_loss(
                            preds['rgb_attention'],
                            preds['thermal_attention'],
                            brightness,
                            brightness_threshold=config.get('brightness_threshold', 0.55),
                            lambda_brightness=config.get('lambda_brightness', 0.1)
                        )
                        val_brightness_reg_loss += float(brightness_loss.item())
                
                val_loss += float(loss_dict['total'].item())
                vsteps += 1
                pbar.set_postfix({'val_loss': f'{(val_loss/max(1,vsteps)):.4f}'})
        val_loss /= max(1, vsteps)
        val_brightness_reg_loss /= max(1, vsteps)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print training progress
        reg_info = ""
        if use_brightness_reg:
            reg_info = f", brightness_reg={brightness_reg_loss:.4f}"
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}, lr={current_lr:.6f}{reg_info}")

        # save best and early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss},
                       os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            # Only trigger early stopping after minimum epochs AND patience exceeded
            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n⚠️ Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation loss: {best_val:.4f} at epoch {best_epoch}")
                print(f"Training stopped at epoch {epoch}")
                break
            elif epoch < min_epochs:
                # Still in minimum training period
                print(f"  (Early stopping disabled until epoch {min_epochs})")

    print("Training done. Best val:", best_val)

if __name__ == "__main__":
    cfg = {
        'experiment_name': 'thermal_rgb_2d_latest_yolo_fixed',  # Updated for Latest YOLO
        'run_name': f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'num_epochs': 50,  # Increased from 40 - multi-scale + anchors need more epochs
        # Optimized batch size for RTX 4060 Ti 8GB with FP16 + multi-scale
        'batch_size': 6,  # Reduced from 8 - multi-scale (P3+P4+P5) + 3 anchors = 9x more predictions, needs more memory
        'learning_rate': 1.5e-4,  # Increased from 6e-5 - better for pre-trained backbone + multi-scale
        'weight_decay': 5e-5,  # Keep same - good regularization
        'num_workers': 6,  # Increased for better CPU-GPU parallelism (try 4-8)
        'img_height': 512,
        'img_width': 640,
        'num_classes': 3,
        'pretrained': True,
        # Performance optimizations
        'use_amp': True,  # Enable mixed precision (FP16) - ~2x speedup on RTX 4060 Ti
        'use_compile': False,  # Set True if PyTorch 2.0+ for additional ~20-30% speedup
        # Architecture improvements (for 65-75% mAP)
        'backbone': 'resnet18',  # 'resnet18' (recommended) or 'resnet50' - ResNet18 is better for small datasets
        'use_fpn': True,  # Feature Pyramid Network for multi-scale detection
        'use_bn': True,  # Enable BatchNorm + Dropout in detection head
        # Model architecture choice
        'model_type': 'yolo',  # 'custom' or 'yolo' - Latest YOLO uses pre-trained detection head
        'yolo_version': 'latest',  # 'latest' (auto), 'yolo11', 'yolo10', 'yolo9', 'yolo8'
        'yolo_weights': None,  # Path to YOLO weights file (optional, auto-downloads latest if None)
        'num_anchors': 3,  # Number of anchor boxes per cell (3 = YOLO standard, enables multi-object detection per cell)
        'use_multiscale': True,  # Use P3, P4, P5 for multi-scale detection (better for objects of different sizes)
        # Loss function improvements
        'use_focal_loss': True,  # Focal Loss for better class imbalance handling
        'use_giou_loss': True,  # GIoU Loss for better bbox localization
        'focal_alpha': 0.25,  # Focal loss alpha parameter
        'focal_gamma': 2.0,  # Focal loss gamma parameter
        'class_weights': [1.0, 1.0, 2.0],  # [car, person, bicycle] - bicycle gets 2x weight due to class imbalance
        # Loss weights: Adjusted for anchor boxes + multi-scale (3 anchors × 3 scales = 9x predictions)
        # With more predictions, we need to scale down loss weights to prevent explosion
        'lambda_obj': 0.4,      # Slightly reduced - with 3 anchors, we have 3x more positive predictions
        'lambda_noobj': 1.5,    # Reduced from 2.0 - with 3 anchors × 3 scales, we have 9x more negative predictions (already penalized more)
        'lambda_box': 5.0,      # Keep same - bbox loss is critical
        'lambda_cls': 1.0,      # Keep same - classification loss is balanced
        # Training improvements (for 50-70% mAP with anchors + multi-scale)
        'warmup_epochs': 5,  # Increased from 3 - multi-scale + anchors need longer warmup for stability
        # Overfitting prevention
        'min_epochs': 20,  # Increased from 15 - ensure minimum training before early stopping
        'early_stopping_patience': 15,  # Reduced from 20 - more aggressive if no improvement
        # Brightness-aware attention regularization
        'use_brightness_regularization': True,  # Enable brightness-aware training to encourage RGB trust in daytime
        'brightness_threshold': 0.55,  # Threshold for bright scenes (nighttime with flashes: 0.40-0.50, daytime: ~0.75)
        'lambda_brightness': 0.1,  # Weight for brightness regularization loss (start with 0.1, can increase to 0.2-0.3)
    }
    train(cfg)

"""
RGB Image ──┐
            ├─> Your Dual Encoders → Your FPN → Your Cross-Modal Attention → YOLO Head → Detections
Thermal ────┘
"""