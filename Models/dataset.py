import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ThermalRGBDataset(Dataset):
    """Dataset for Thermal-RGB paired images"""
    
    def __init__(self, pairs, transform=None, image_size=(512, 640)):
        """
        Args:
            pairs: List of {'thermal': path, 'rgb': path, 'name': id}
            transform: Albumentations transform
            image_size: (height, width) - default 512x640
        """
        self.pairs = pairs
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load thermal image
        thermal_img = cv2.imread(pair['thermal'])
        thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)  # pyright: ignore[reportCallIssue, reportArgumentType]
        
        # Load RGB image
        rgb_img = cv2.imread(pair['rgb'])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # pyright: ignore[reportCallIssue, reportArgumentType]
        
        # Resize to consistent size
        thermal_img = cv2.resize(thermal_img, (self.image_size[1], self.image_size[0]))
        rgb_img = cv2.resize(rgb_img, (self.image_size[1], self.image_size[0]))
        
        # Apply transforms (same transform to both images for consistency)
        if self.transform:
            # Apply same augmentation to both images
            transformed = self.transform(image=rgb_img, thermal=thermal_img)
            rgb_img = transformed['image']
            thermal_img = transformed['thermal']
        else:
            # Just convert to tensor
            rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
            thermal_img = torch.from_numpy(thermal_img).permute(2, 0, 1).float() / 255.0
        
        return {
            'rgb': rgb_img,
            'thermal': thermal_img,
            'name': pair['name']
        }


def get_transforms(split='train', image_size=(512, 640)):
    """Get augmentation transforms"""
    
    if split == 'train':
        # Training augmentations - stronger to prevent overfitting
        transform = A.Compose([  # pyright: ignore[reportArgumentType]
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.3),  # Geometric augmentation
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # Increased from 0.2
                contrast_limit=0.3,    # Increased from 0.2
                p=0.6
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),  # Color augmentation
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),  # Fixed: var_limit should be tuple
            A.GaussianBlur(blur_limit=3, p=0.2),  # Add blur
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),  # Cutout-like augmentation
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # pyright: ignore[reportArgumentType]
                std=[0.229, 0.224, 0.225]  # pyright: ignore[reportArgumentType]
            ),
            ToTensorV2()
        ], additional_targets={'thermal': 'image'})
    else:
        # Validation: only normalize
        transform = A.Compose([  # pyright: ignore[reportArgumentType]
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # pyright: ignore[reportArgumentType]
                std=[0.229, 0.224, 0.225]  # pyright: ignore[reportArgumentType]
            ),
            ToTensorV2()
        ], additional_targets={'thermal': 'image'})
    
    return transform


def create_dataloaders(batch_size=8, num_workers=4, image_size=(512, 640)):
    """Create train and validation dataloaders (SILENT)"""
    
    # Get project root directory (go up from Models/ folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'Config', 'clean_pairs.json')
    
    # Load clean pairs
    if not os.path.exists(config_path):
        print(f"❌ clean_pairs.json not found at: {config_path}")
        return None, None
    
    with open(config_path, 'r') as f:
        clean_data = json.load(f)
    
    train_pairs = clean_data['train']
    val_pairs = clean_data['val']
    
    # Create datasets
    train_dataset = ThermalRGBDataset(
        pairs=train_pairs,
        transform=get_transforms('train', image_size),
        image_size=image_size
    )
    
    val_dataset = ThermalRGBDataset(
        pairs=val_pairs,
        transform=get_transforms('val', image_size),
        image_size=image_size
    )
    
    # Create dataloaders with optimizations for RTX 4060 Ti
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch 2 batches per worker (reduces CPU-GPU wait)
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        drop_last=False  # Don't drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING DATASET")
    print("="*60)
    
    # Create dataloaders
    print("\n📦 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
        image_size=(512, 640)
    )
    
    if train_loader is None:
        print("❌ Failed to create dataloaders")
        exit(1)
    
    print(f"✓ Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")  # pyright: ignore[reportArgumentType]
    print(f"✓ Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
    
    # Test loading one batch
    print("\n🧪 Testing batch loading...")
    try:
        batch = next(iter(train_loader))
        
        print(f"✓ RGB shape: {batch['rgb'].shape}")
        print(f"✓ Thermal shape: {batch['thermal'].shape}")
        print(f"✓ Data ranges: RGB [{batch['rgb'].min():.2f}, {batch['rgb'].max():.2f}]")
        
        print("\n" + "="*60)
        print("✅ DATASET READY")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()