import sys
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ultralytics_source')))
from ultralytics import YOLO

def load_6ch_image(thermal_path):
    # Load thermal (PIL handles grayscale automatically, but we convert to RGB so it's 3-channel)
    img_t = Image.open(thermal_path).convert('RGB')
    
    # Construct RGB path directly
    p = Path(thermal_path)
    split_name = p.parent.name # e.g. 'val'
    f_rgb_path = Path(f'data/FLIR_ADAS_1_3/{split_name}/RGB/{p.stem}.jpg')
    
    if not f_rgb_path.exists():
        print(f"Warning: RGB image not found at {f_rgb_path}. Using thermal twice.")
        img_rgb = img_t
    else:
        img_rgb = Image.open(str(f_rgb_path)).convert('RGB')
        
    img_t_np = np.array(img_t)
    img_rgb_np = np.array(img_rgb)
    
    # Resize RGB to match thermal if needed
    if img_rgb_np.shape[:2] != img_t_np.shape[:2]:
        import cv2
        img_rgb_np = cv2.resize(img_rgb_np, (img_t_np.shape[1], img_t_np.shape[0]))
        
    # Stack along channels (Thermal + RGB)
    stacked = np.concatenate((img_t_np, img_rgb_np), axis=-1)
    
    # Convert to PyTorch format: [C, H, W]
    stacked = stacked.transpose((2, 0, 1))
    
    # Add batch dim, convert to float, and normalize to 0-1
    tensor = torch.from_numpy(stacked).float().unsqueeze(0) / 255.0
    
    # Ensure dimensions are multiples of 32 (stride) for YOLO
    h, w = tensor.shape[2], tensor.shape[3]
    h_new, w_new = (h // 32) * 32, (w // 32) * 32
    return tensor[:, :, :h_new, :w_new]

def run_test():
    checkpoint = 'runs/detect/checkpoints/DualYOLO26s_6ch_v4_SEContext/weights/last.pt'
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found at {checkpoint}")
        return
        
    print(f"Loading checkpoint: {checkpoint}")
    model = YOLO(checkpoint)
    model.model.eval()
    device = next(model.model.parameters()).device
    
    val_dir = Path('data/FLIR_YOLO/images/val')
    images = list(val_dir.glob('*.jpeg'))
    
    # Pick a few specific images or random ones
    import random
    random.seed(42) # For reproducibility
    test_images = random.sample(images, min(5, len(images)))
    
    print("\n===============================")
    print("Testing Multiple Images")
    print("===============================\n")
    
    for img_path in test_images:
        tensor = load_6ch_image(str(img_path)).to(device)
        
        with torch.no_grad():
            _ = model.model(tensor)
            
        print(f"Image: {img_path.name}")
        # Extract Trust Meter for all fusion layers
        for layer in ["16", "19", "22"]:
            attn = model.model.attention_modules[layer].last_attention_weights
            rgb_trust = attn[:, 0, :, :].mean().item() * 100
            therm_trust = attn[:, 1, :, :].mean().item() * 100
            print(f"  Layer {layer} -> RGB: {rgb_trust:>5.2f}% | Thermal: {therm_trust:>5.2f}%")
        print("-" * 40)
        
if __name__ == '__main__':
    run_test()
