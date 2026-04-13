# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

FatigueSense is a real-time fatigue detection system using Vision AI. It detects driver/operator mental exhaustion via a 3-phase pipeline:

1. **Phase A — YOLO11 Detection:** Detects 4 ROIs (Head, Torso, Mouth, Eyes) from video frames
2. **Phase B — CNN Classification:** Lightweight CNNs (MobileNetV3/GhostNet) classify behavioral states per ROI crop
3. **Phase C — Temporal Modeling:** BiLSTM + ACAMF fusion across sliding windows outputs a continuous focus score (0.0–1.0)

Dataset hosted on Hugging Face Hub at `Jlords32/FatigueSense`. YOLO format labels. Classes: `0=Head, 1=Torso, 2=Mouth, 3=Eyes`.

## Commands

No build system. All entry points are Python scripts or Jupyter notebooks.

```bash
# Extract frames from videos (every Nth frame)
python scripts/frame_extraction.py

# Merge per-video datasets into train/val split under data/
python scripts/merge_datasets.py

# Fix class ID ordering (vid_001-specific issue)
python scripts/remap_labels.py

# Upload merged dataset to Hugging Face
python scripts/upload_dataset.py

# Crop detected regions from frames
python scripts/image_cropping.py
```

Training runs in `notebooks/YOLOv11_train.ipynb` (also contains webcam inference demo).

## Architecture & Data Flow

```
videos/ → frame_extraction.py → dataset/vid_XXX/{images,labels}/
                                        ↓
                               merge_datasets.py
                                        ↓
                               data/{train,val}/{images,labels}/
                                        ↓
                               YOLOv11_train.ipynb → runs/detect/
```

Per-video datasets live in `dataset/vid_001/` … `dataset/vid_008/`, each with `images/` and `labels/` subdirectories in YOLO format.

Merged data lands in `data/train/` and `data/val/`. Model checkpoints output to `runs/`.

## Key Files

| File | Purpose |
|------|---------|
| `docs/implementation_plan.md` | Architecture decisions for CNN and temporal model |
| `docs/class_labels.md` | Canonical class definitions and behavioral states |
| `DOCUMENTATION.md` | End-to-end pipeline walkthrough (annotation → training) |
| `SETUP.md` | Hugging Face auth methods |
| `.env` | ClearML + HF credentials — never commit new secrets here |
| `yolo11n.pt` / `best.pt` | Model weights tracked via Git LFS |

## Dependencies

Core: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `pandas`, `yaml`, `huggingface_hub`, `clearml`, `python-dotenv`, `tqdm`.

Load `.env` before running scripts that touch ClearML or Hugging Face.
