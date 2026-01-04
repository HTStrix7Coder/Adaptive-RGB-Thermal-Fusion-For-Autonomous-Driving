import os
import json
import cv2
import numpy as np
from tqdm import tqdm

print("="*60)
print("DATASET PAIR VERIFICATION V2")
print("="*60)

# Load dataset info
with open('dataset_info.json', 'r') as f:
    dataset_info = json.load(f)

def is_valid_image(img_path):
    """Check if image is valid"""
    if not os.path.exists(img_path):
        return False
    
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    # Check if image is mostly white/blank
    mean_val = np.mean(img)  # pyright: ignore[reportCallIssue, reportArgumentType]
    if mean_val > 250:  # Mostly white
        return False
    
    # Check if image has variance
    std_val = np.std(img)  # pyright: ignore[reportCallIssue, reportArgumentType]
    if std_val < 5:  # Too uniform
        return False
    
    return True

def find_matching_pairs(thermal_folder, rgb_folder):
    """Find matching thermal-RGB pairs with flexible matching"""
    print(f"\n🔍 Pairing:")
    print(f"  Thermal: {os.path.basename(thermal_folder)}")
    print(f"  RGB: {os.path.basename(rgb_folder)}")
    
    # Get all files
    thermal_files = sorted([f for f in os.listdir(thermal_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"  Thermal images: {len(thermal_files)}")
    print(f"  RGB images: {len(rgb_files)}")
    
    # Create RGB lookup (handle .jpg vs .jpeg)
    rgb_dict = {}
    for rgb_file in rgb_files:
        # Get base name without extension
        base = os.path.splitext(rgb_file)[0]
        rgb_dict[base] = rgb_file
    
    valid_pairs = []
    missing_rgb = 0
    invalid_images = 0
    
    for thermal_file in tqdm(thermal_files, desc="  Matching"):
        thermal_path = os.path.join(thermal_folder, thermal_file)
        
        # Get base name (remove extension)
        thermal_base = os.path.splitext(thermal_file)[0]
        
        # Look for matching RGB
        if thermal_base in rgb_dict:
            rgb_path = os.path.join(rgb_folder, rgb_dict[thermal_base])
            
            # Verify both are valid
            if is_valid_image(thermal_path) and is_valid_image(rgb_path):
                valid_pairs.append({
                    'thermal': thermal_path,
                    'rgb': rgb_path,
                    'name': thermal_base
                })
            else:
                invalid_images += 1
        else:
            missing_rgb += 1
    
    print(f"  ✓ Valid pairs: {len(valid_pairs)}")
    if missing_rgb > 0:
        print(f"  ⚠ Missing RGB: {missing_rgb}")
    if invalid_images > 0:
        print(f"  ⚠ Invalid images: {invalid_images}")
    
    return valid_pairs

# Process train and val splits
print("\n📊 Finding valid pairs...")

train_pairs = []
val_pairs = []

# Train pairs
train_thermal = "data\\FLIR_ADAS_1_3\\train\\thermal_8_bit"
train_rgb = "data\\FLIR_ADAS_1_3\\train\\RGB"
if os.path.exists(train_thermal) and os.path.exists(train_rgb):
    train_pairs = find_matching_pairs(train_thermal, train_rgb)

# Val pairs  
val_thermal = "data\\FLIR_ADAS_1_3\\val\\thermal_8_bit"
val_rgb = "data\\FLIR_ADAS_1_3\\val\\RGB"
if os.path.exists(val_thermal) and os.path.exists(val_rgb):
    val_pairs = find_matching_pairs(val_thermal, val_rgb)

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"\n✅ Training pairs: {len(train_pairs)}")
print(f"✅ Validation pairs: {len(val_pairs)}")
print(f"✅ Total valid pairs: {len(train_pairs) + len(val_pairs)}")

# Calculate what we have
total_thermal_train = len([f for f in os.listdir(train_thermal) if f.endswith(('.jpg', '.jpeg'))])
total_rgb_train = len([f for f in os.listdir(train_rgb) if f.endswith(('.jpg', '.jpeg'))])
total_thermal_val = len([f for f in os.listdir(val_thermal) if f.endswith(('.jpg', '.jpeg'))])
total_rgb_val = len([f for f in os.listdir(val_rgb) if f.endswith(('.jpg', '.jpeg'))])

print(f"\n📊 Dataset coverage:")
print(f"  Train: {len(train_pairs)}/{min(total_thermal_train, total_rgb_train)} ({len(train_pairs)/min(total_thermal_train, total_rgb_train)*100:.1f}%)")
print(f"  Val: {len(val_pairs)}/{min(total_thermal_val, total_rgb_val)} ({len(val_pairs)/min(total_thermal_val, total_rgb_val)*100:.1f}%)")

# Save clean pairs
if train_pairs or val_pairs:
    clean_dataset = {
        'train': train_pairs,
        'val': val_pairs,
        'total_train': len(train_pairs),
        'total_val': len(val_pairs),
        'total_pairs': len(train_pairs) + len(val_pairs)
    }
    
    with open('clean_pairs.json', 'w') as f:
        json.dump(clean_dataset, f, indent=2)
    
    print(f"\n💾 Saved to: clean_pairs.json")
    
    print("\n" + "="*60)
    print("✅ DATASET READY FOR TRAINING!")
    print("="*60)
    print(f"\n🎯 You have {len(train_pairs)} training pairs")
    print(f"🎯 You have {len(val_pairs)} validation pairs")
    print("\n✅ This is plenty of data for the project!")
    print("\n✅ Proceed to Step 4: Data Preprocessing")
else:
    print("\n❌ No valid pairs found")

print("="*60)