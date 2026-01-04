# FLIR Dataset Structure - SHORT Version

## **The 2 Folders We Actually Use:**

FLIR_ADAS_1_3/
├── train/
│   ├── thermal_8_bit/  ← 8,862 thermal images (heat signatures)
│   └── RGB/            ← 8,363 normal camera images
└── val/
    ├── thermal_8_bit/  ← 1,366 thermal images
    └── RGB/            ← 1,257 normal camera images

## **What We Ignore:**

❌ annotated_thermal_8_bit/  - Pre-drawn boxes (don't need)
❌ thermal_16_bit/           - Empty folder
❌ video/                    - Not using
❌ PDFs/txt files            - Just docs

## **How It Works:**

```
thermal_8_bit/FLIR_00001.jpeg  +  RGB/FLIR_00001.jpg
     (Heat view)                    (Normal view)
          ↓                              ↓
        Same scene, same time, different cameras
```
## **What Each Shows:**

Hot (30°C) = White
Medium (25°C) = Gray
Cold (15°C) = Black

## **Bottom Line:**

- **We use:** `thermal_8_bit/` + `RGB/` (paired images)
- **We ignore:** Everything else
- **Total useful pairs:** ~9,600 (8,347 train + 1,256 val)

"I selected the FLIR ADAS v1.3 dataset (10,228 RGB-Thermal pairs) for several key reasons:
1. Hardware Constraints: The dataset size is optimized for single-GPU training on consumer hardware (RTX 4060 Ti, 8GB). Larger datasets like KAIST (95k samples) or nuScenes (1.4M samples) require 16-32GB GPUs or multi-GPU setups, which are infeasible for this project.
2. Training Efficiency: FLIR enables reasonable training time (~5 hours for 30 epochs) compared to 40+ hours for KAIST or 200+ hours for nuScenes, making iterative experimentation practical within project timelines.
3. Quality Over Quantity: FLIR provides professional-grade, perfectly aligned RGB-Thermal pairs captured with actual FLIR thermal cameras in real driving scenarios, with comprehensive COCO-format annotations for 3 object classes (car, person, bicycle).
4. Domain Relevance: The dataset specifically targets autonomous driving scenarios with urban/suburban road-level perspectives, directly matching the intended application domain.
5. Established Benchmark: FLIR is widely used in thermal-RGB fusion research (100+ citations), enabling direct comparison with existing methods and validation of results.
Alternative datasets were considered but rejected:

KAIST (95k): Requires 16GB+ GPU, 40+ hour training, 50GB download
nuScenes (1.4M): Requires multi-GPU setup, 350GB storage, enterprise-level compute
M3FD (4k): Too small for robust deep learning training
CVC-14 (7.5k): Limited annotations, less established benchmark

The FLIR dataset provides the optimal balance of size, quality, and computational feasibility for demonstrating RGB-Thermal fusion effectiveness on consumer hardware."