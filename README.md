# Dental X-Ray Detection & Diagnosis

A two-stage deep learning pipeline for automated dental analysis from panoramic X-ray images. The system detects jaw quadrants, enumerates individual teeth, and predicts dental diseases.

## Overview

| Stage | Model | Task |
|-------|-------|------|
| 1 | YOLOv8s | Quadrant detection (Q1–Q4) |
| 2 | EfficientNetB3 (dual-head) | Tooth enumeration + disease classification |

**Pipeline**: X-ray → CLAHE enhancement → YOLOv8 quadrant detection → crop & resize → EfficientNetB3 diagnosis → annotated output

## Project Structure

```
├── src/
│   └── Code_dental_detection.ipynb   # Main notebook (full pipeline)
├── best models/
│   ├── best.pt                       # YOLOv8s Stage 1 weights
│   └── stage2_efficientnetb3_best.keras  # EfficientNetB3 Stage 2 weights
└── docs/
    └── Project Final Report.pdf
```

## Requirements

- Python 3.8+
- `ultralytics` (YOLOv8)
- `tensorflow` / `keras` (EfficientNetB3)
- `opencv-python`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `Pillow`

## Usage

Please Open and run `src/Code_dental_detection.ipynb` end to end. Update `BASE_DIR` in the configuration cell to point to the local project root before running and the best weights path too. 

## Notes

- CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to all X-rays before inference to improve contrast on low-quality scans.
- Stage 1 crops are saved to `stage1_crops/` and used as input to Stage 2.
- Training uses an 75/15/10 train/val/test split.
