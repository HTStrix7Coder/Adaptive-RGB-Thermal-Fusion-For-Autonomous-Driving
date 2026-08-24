import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.model_yolo import ThermalRGB2DNetLatestYOLO
from Models.dataset import create_dataloaders

def calculate_brightness(rgb_tensor):
    # Grayscale formula to determine average scene brightness
    gray = 0.2989 * rgb_tensor[:, 0, :, :] + 0.5870 * rgb_tensor[:, 1, :, :] + 0.1140 * rgb_tensor[:, 2, :, :]
    return gray.mean(dim=(1,2))

def plot_trust():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading DataLoader...")
    _, val_loader = create_dataloaders(16, 4, (512, 640))
    
    print("Loading Model...")
    checkpoint_path = 'checkpoints/thermal_rgb_2d_convnext_tiny/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = checkpoint.get('config', {})
    
    # Initialize Latest YOLO custom head with ConvNeXt-Tiny
    model = ThermalRGB2DNetLatestYOLO(
        num_classes=3,
        pretrained=False,
        use_bn=True,
        use_fpn=True,
        backbone='convnext_tiny',
        use_multiscale=True,
        yolo_version='latest',
        num_anchors=3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    brightness_list = []
    rgb_trust_list = []
    thermal_trust_list = []

    print(f"Analyzing {len(val_loader.dataset)} Validation Images...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            
            b = calculate_brightness(rgb).cpu().numpy()
            
            preds = model(rgb, thermal, return_attention=True)
            
            # Extract mean attention weight from the P4 FPN layer
            if isinstance(preds['rgb_attention'], dict):
                r_a = preds['rgb_attention']['p4'].mean(dim=(1,2,3)).cpu().numpy()
                t_a = preds['thermal_attention']['p4'].mean(dim=(1,2,3)).cpu().numpy()
            else:
                r_a = preds['rgb_attention'].mean(dim=(1,2,3)).cpu().numpy()
                t_a = preds['thermal_attention'].mean(dim=(1,2,3)).cpu().numpy()
                
            brightness_list.extend(b)
            rgb_trust_list.extend(r_a)
            thermal_trust_list.extend(t_a)

    brightness_list = np.array(brightness_list)
    rgb_trust_list = np.array(rgb_trust_list)
    thermal_trust_list = np.array(thermal_trust_list)

    # Sort arrays by brightness for smooth plotting
    sort_idx = np.argsort(brightness_list)
    brightness_list = brightness_list[sort_idx]
    rgb_trust_list = rgb_trust_list[sort_idx]
    thermal_trust_list = thermal_trust_list[sort_idx]

    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Scatter plot of actual image samples
    plt.scatter(brightness_list, rgb_trust_list, color='#2ecc71', alpha=0.5, edgecolor='black', s=50, label='RGB Sensor Trust')
    plt.scatter(brightness_list, thermal_trust_list, color='#9b59b6', alpha=0.5, edgecolor='black', s=50, label='Thermal Sensor Trust')
    
    # Quadratic Trendlines
    z_rgb = np.polyfit(brightness_list, rgb_trust_list, 2)
    p_rgb = np.poly1d(z_rgb)
    plt.plot(brightness_list, p_rgb(brightness_list), color='#27ae60', linewidth=4, linestyle='-')
    
    z_thm = np.polyfit(brightness_list, thermal_trust_list, 2)
    p_thm = np.poly1d(z_thm)
    plt.plot(brightness_list, p_thm(brightness_list), color='#8e44ad', linewidth=4, linestyle='-')
    
    # Formatting
    plt.title('Self-Driving Sensor Trust vs. Environmental Brightness', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Environmental Brightness (0.0 = Pitch Black Night, 1.0 = Bright Daylight)', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Mean Attention Trust (%)', fontsize=14, fontweight='bold', labelpad=15)
    
    # The Threshold Line
    plt.axvline(x=0.55, color='#e74c3c', linestyle='--', linewidth=2, label='Day/Night Activation Threshold (0.55)')
    
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.legend(fontsize=12, loc='center right')
    
    os.makedirs('results/visualizations', exist_ok=True)
    save_path = 'results/visualizations/dynamic_trust_curve.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Success! Saved academic curve to {save_path}")

if __name__ == '__main__':
    plot_trust()
