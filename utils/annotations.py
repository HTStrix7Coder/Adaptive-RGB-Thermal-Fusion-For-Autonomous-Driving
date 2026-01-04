import json
import os
import numpy as np

def load_coco_annotations(annotation_file):
    """Load COCO format annotations (thermal_annotations.json)"""

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Image ID → filename lookup
    images = {img['id']: img['file_name'] for img in data['images']}

    # Category ID → category name mapping
    categories = {cat['id']: cat['name'].lower() for cat in data['categories']}

    # Annotation dict
    annotations_by_image = {}

    # Class mapping for your 3 classes
    class_map = {
        'car': 0,
        'person': 1,
        'pedestrian': 1,      # FLIR sometimes uses this
        'bicycle': 2,
        'bike': 2,
        'cyclist': 2
    }

    for anno in data['annotations']:
        image_id = anno['image_id']

        # Skip if no image entry
        if image_id not in images:
            continue

        filename = images[image_id]

        # ❗ FIX: Clean filename properly
        base_name = os.path.splitext(os.path.basename(filename))[0]

        if base_name not in annotations_by_image:
            annotations_by_image[base_name] = {'boxes': [], 'labels': []}

        # COCO bbox: [x, y, width, height]
        bbox = anno['bbox']

        # Map category → class_id
        category_name = categories.get(anno['category_id'], 'unknown')
        class_id = class_map.get(category_name, -1)

        if class_id >= 0:
            annotations_by_image[base_name]['boxes'].append(bbox)
            annotations_by_image[base_name]['labels'].append(class_id)

    # Convert lists to arrays
    for base_name in annotations_by_image:
        annotations_by_image[base_name]['boxes'] = np.array(
            annotations_by_image[base_name]['boxes'], dtype=np.float32)
        annotations_by_image[base_name]['labels'] = np.array(
            annotations_by_image[base_name]['labels'], dtype=np.int64)

    return annotations_by_image


def get_annotations_for_split(split='train'):
    """Load annotations for train/val split"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    anno_file = os.path.join(
        project_root,
        'data',
        'FLIR_ADAS_1_3',
        split,
        'thermal_annotations.json'
    )

    if not os.path.exists(anno_file):
        print(f"⚠️ Annotation file not found: {anno_file}")
        return {}

    print(f"✓ Loading annotations from: {anno_file}")
    return load_coco_annotations(anno_file)


if __name__ == "__main__":
    print("="*60)
    print("TESTING ANNOTATION LOADING")
    print("="*60)

    print("\n📋 Loading train annotations...")
    train_annos = get_annotations_for_split('train')

    print("\n📋 Loading val annotations...")
    val_annos = get_annotations_for_split('val')

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n✓ Train annotations: {len(train_annos)} images")
    print(f"✓ Val annotations: {len(val_annos)} images")

    if train_annos:
        sample_id = list(train_annos.keys())[0]
        sample = train_annos[sample_id]
        print(f"\n📸 Example: {sample_id}")
        print(f"   Objects: {len(sample['boxes'])}")
        print(f"   Labels: {sample['labels']}")
        if len(sample['boxes']) > 0:
            print(f"   First box: {sample['boxes'][0]}")

    # Object distribution
    if train_annos:
        class_counts = {0: 0, 1: 0, 2: 0}
        for anno in train_annos.values():
            for label in anno['labels']:
                class_counts[label] += 1

        print(f"\n📊 Distribution:")
        print(f"   Cars (0): {class_counts[0]}")
        print(f"   Persons (1): {class_counts[1]}")
        print(f"   Bikes (2): {class_counts[2]}")
        print(f"   Total objects: {sum(class_counts.values())}")

    print("\n" + "="*60)
