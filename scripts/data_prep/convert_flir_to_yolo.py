"""
Convert FLIR ADAS COCO-format annotations to YOLO format.

YOLO expects:
  - Images in:  .../images/train/  and  .../images/val/
  - Labels in:  .../labels/train/  and  .../labels/val/
  - Each label file: one line per object: class x_center y_center width height (normalized 0-1)

FLIR ADAS has:
  - Thermal images in:  data/FLIR_ADAS_1_3/{split}/thermal_8_bit/FLIR_XXXXX.jpeg
  - RGB images in:      data/FLIR_ADAS_1_3/{split}/RGB/FLIR_XXXXX.jpg
  - Annotations in:     data/FLIR_ADAS_1_3/{split}/thermal_annotations.json (COCO format)

This script:
  1. Parses COCO annotations for train/val
  2. Symlinks (or copies) thermal images into YOLO-expected directory structure
  3. Creates YOLO-format .txt label files
  4. Only includes images that have BOTH thermal AND RGB pairs
"""

import json
import os
import shutil
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "FLIR_ADAS_1_3"
# Output: YOLO-format dataset alongside the original
YOLO_ROOT = PROJECT_ROOT / "data" / "FLIR_YOLO"

# FLIR category IDs → our YOLO class IDs
# We only care about: car, person, bicycle
CATEGORY_MAP = {
    3: 0,    # car → class 0
    1: 1,    # person → class 1
    2: 2,    # bicycle → class 2
    71: 2,   # bike → class 2 (FLIR alternate name)
    73: -1,  # motor → skip
    74: 1,   # rider → person (class 1)
}

CLASS_NAMES = {0: 'car', 1: 'person', 2: 'bicycle'}


def convert_split(split: str):
    """Convert one split (train or val) from COCO to YOLO format."""
    
    anno_file = DATA_ROOT / split / "thermal_annotations.json"
    if not anno_file.exists():
        print(f"  ❌ Annotation file not found: {anno_file}")
        return
    
    with open(anno_file, 'r') as f:
        coco = json.load(f)
    
    # Build image lookup: id → {file_name, width, height}
    images = {}
    for img in coco['images']:
        images[img['id']] = {
            'file_name': img['file_name'],  # e.g. "thermal_8_bit/FLIR_00001.jpeg"
            'width': img['width'],
            'height': img['height'],
        }
    
    # Build annotations per image
    annos_per_image = {}
    for anno in coco['annotations']:
        img_id = anno['image_id']
        if img_id not in images:
            continue
        cat_id = anno['category_id']
        yolo_cls = CATEGORY_MAP.get(cat_id, -1)
        if yolo_cls < 0:
            continue  # skip non-target categories
        
        if img_id not in annos_per_image:
            annos_per_image[img_id] = []
        annos_per_image[img_id].append({
            'class': yolo_cls,
            'bbox': anno['bbox'],  # COCO: [x_min, y_min, width, height] in pixels
        })
    
    # Create output directories
    images_dir = YOLO_ROOT / "images" / split
    labels_dir = YOLO_ROOT / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    total = 0
    skipped_no_rgb = 0
    skipped_no_thermal = 0
    empty_labels = 0
    total_objects = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for img_id, img_info in images.items():
        # Thermal image path
        thermal_rel = img_info['file_name']  # "thermal_8_bit/FLIR_00001.jpeg"
        thermal_path = DATA_ROOT / split / thermal_rel
        
        if not thermal_path.exists():
            skipped_no_thermal += 1
            continue
        
        # Extract base name: FLIR_00001
        base_name = thermal_path.stem  # FLIR_00001
        
        # Find RGB pair: same base name but .jpg extension in RGB/ folder
        rgb_path = DATA_ROOT / split / "RGB" / f"{base_name}.jpg"
        if not rgb_path.exists():
            skipped_no_rgb += 1
            continue
        
        # Symlink/copy thermal image to YOLO images dir
        # Use thermal as the "primary" image since annotations are aligned to thermal
        dst_img = images_dir / f"{base_name}.jpeg"
        if not dst_img.exists():
            # Use symlink on Windows (requires admin) or copy
            try:
                os.symlink(thermal_path, dst_img)
            except OSError:
                shutil.copy2(thermal_path, dst_img)
        
        # Create YOLO label file
        img_w = img_info['width']
        img_h = img_info['height']
        
        label_path = labels_dir / f"{base_name}.txt"
        annos = annos_per_image.get(img_id, [])
        
        with open(label_path, 'w') as f:
            for anno in annos:
                cls = anno['class']
                # COCO bbox: [x_min, y_min, w, h] → YOLO: [x_center, y_center, w, h] normalized
                x_min, y_min, bw, bh = anno['bbox']
                x_center = (x_min + bw / 2) / img_w
                y_center = (y_min + bh / 2) / img_h
                bw_norm = bw / img_w
                bh_norm = bh / img_h
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bw_norm = max(0, min(1, bw_norm))
                bh_norm = max(0, min(1, bh_norm))
                
                # Skip degenerate boxes
                if bw_norm < 0.001 or bh_norm < 0.001:
                    continue
                
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
                total_objects += 1
                class_counts[cls] += 1
        
        if not annos:
            empty_labels += 1
        
        total += 1
    
    print(f"\n  📊 {split.upper()} Results:")
    print(f"     Total images processed: {total}")
    print(f"     Skipped (no RGB pair): {skipped_no_rgb}")
    print(f"     Skipped (no thermal): {skipped_no_thermal}")
    print(f"     Empty labels (no target objects): {empty_labels}")
    print(f"     Total objects: {total_objects}")
    print(f"     Class distribution:")
    for cls_id, count in class_counts.items():
        print(f"       {CLASS_NAMES[cls_id]}: {count}")
    
    return total


def main():
    print("=" * 60)
    print("FLIR ADAS → YOLO Format Converter")
    print("=" * 60)
    print(f"\nSource: {DATA_ROOT}")
    print(f"Output: {YOLO_ROOT}")
    
    # Clean output directory
    if YOLO_ROOT.exists():
        print(f"\n⚠️  Removing existing output: {YOLO_ROOT}")
        shutil.rmtree(YOLO_ROOT)
    
    for split in ['train', 'val', 'video']:
        print(f"\n{'─' * 40}")
        print(f"Converting {split}...")
        convert_split(split)
    
    # Verify
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")
    for split in ['train', 'val', 'video']:
        img_dir = YOLO_ROOT / "images" / split
        lbl_dir = YOLO_ROOT / "labels" / split
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")
        
        # Check a sample label
        if n_lbls > 0:
            sample = list(lbl_dir.glob("*.txt"))[0]
            with open(sample, 'r') as f:
                lines = f.readlines()
            print(f"    Sample label ({sample.name}):")
            for line in lines[:3]:
                print(f"      {line.strip()}")
            if len(lines) > 3:
                print(f"      ... ({len(lines)} total objects)")
    
    print(f"\n✅ Done! Dataset ready at: {YOLO_ROOT}")
    print(f"   Use 'Config/dataset_dual.yaml' to point YOLO at this dataset.")


if __name__ == "__main__":
    main()
