# FatigueSense — Implementation Plan

## Architecture Overview

3-phase pipeline:

1. **Phase A — Pose-Based Spatial Extraction:** YOLO11n-pose detects ROIs + extracts keypoints
2. **Phase B — CNN Classification:** MobileNetV3-Small classifies Eyes and Mouth crops (binary)
3. **Phase C — Temporal Modeling:** BiGRU across sliding windows → continuous Focus Score (0.0–1.0)

Target hardware: average laptop CPU. Target RTF < 1 (processing at 15fps via frame skipping).

### End-to-End Pipeline

```mermaid
flowchart TD
    CAM["Camera\n30fps source"] -->|"every 2nd frame\n→ 15fps"| YOLO

    subgraph YOLO["Phase A — YOLO11n-pose"]
        Y1["Person detection\n640×640 input"]
        Y2["17 COCO keypoints\n+ bounding boxes"]
        Y1 --> Y2
    end

    Y2 -->|"Eye bbox crop"| CNN_E
    Y2 -->|"Mouth bbox crop"| CNN_M
    Y2 -->|"Face keypoints\n(nose, eyes, ears)"| HPE
    Y2 -->|"Shoulder + hip\nkeypoints"| TPE

    subgraph PHASEB["Phase B — Feature Extraction"]
        CNN_E["Eyes CNN\nMobileNetV3-Small\n→ p_eyes_closed"]
        CNN_M["Mouth CNN\nMobileNetV3-Small\n→ p_mouth_open"]
        HPE["HeadPoseExtractor\nPnP solver\n→ pitch, yaw, roll"]
        TPE["TorsoPostureExtractor\nGeometry\n→ slouch ratio, lean angle"]
    end

    CNN_E --> FV
    CNN_M --> FV
    HPE --> FV
    TPE --> FV

    subgraph PHASEC["Phase C — Temporal Modeling"]
        FV["Feature Vector\n(13-dim per frame)"]
        WIN["Sliding Window\n(T × 13) matrix"]
        ACAMF["ACAMF\nOptional occlusion weighting"]
        GRU["BiGRU\n2 layers · 64 units/dir\n+ FC regression head"]
        FV --> WIN --> ACAMF -->|"use_acamf: true"| GRU
        WIN -->|"use_acamf: false\n(default)"| GRU
    end

    GRU --> EMA["EMA Smoothing"]
    EMA --> OUT["Focus Score\n0.0 fatigued → 1.0 alert"]
    OUT --> GUI["GUI Dashboard\n+ Threshold Alerts"]
```

---

## Phase A: YOLO11n-Pose — Spatial Localization

**Why pose model over detection-only:**
Head and torso behavioral states (pitch/yaw/roll, slouch ratio) are derivable from COCO keypoint geometry — no separate CNN needed for those ROIs. Single model pass provides both bounding boxes (for Eye/Mouth crops) and keypoints (for Head/Torso features).

### Configuration
- **Model:** YOLO11n-pose (nano variant, ~6MB)
- **Input Resolution:** 640×640
- **Confidence Threshold:** 0.5 — detections below discarded; missing ROI flagged as occluded
- **NMS IoU Threshold:** 0.45
- **Frame Rate:** 15fps (process every 2nd frame at 30fps source) → ~66ms/frame budget

### Outputs per Frame

```mermaid
flowchart LR
    YOLO["YOLO11n-pose\nforward pass"] --> BB["Bounding Boxes\n(Eyes, Mouth, Head, Torso)"]
    YOLO --> KP["17 COCO Keypoints"]

    BB -->|"crop + preprocess"| EYES["Eye crop\n64×32"]
    BB -->|"crop + preprocess"| MOUTH["Mouth crop\n64×32"]

    KP -->|"keypoints 0–4\nnose, eyes, ears"| HEAD["Head region\n→ HeadPoseExtractor"]
    KP -->|"keypoints 5,6,11,12\nshoulders, hips"| TORSO["Torso region\n→ TorsoPostureExtractor"]
```

### ROI Pre-Processing (Eyes/Mouth crops)
- Resize to fixed resolution per ROI: Eyes → 64×32, Mouth → 64×32
- Grayscale → 3-channel (`Grayscale(num_output_channels=3)`)
- Normalize with ImageNet mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`
- Padding applied when bounding box clips frame boundary

---

## Phase B: Feature Extraction

### Eyes + Mouth — CNN (MobileNetV3-Small)

Selected for: depthwise separable convolutions, strong mobile track record, pretrained ImageNet weights.

```mermaid
flowchart TD
    CROP["ROI Crop\n(B, 3, H, W)"] --> BACKBONE
    subgraph BACKBONE["MobileNetV3-Small (shared backbone)"]
        B1["Depthwise Separable Conv\n+ SE Blocks + Hardswish"]
        B2["AdaptiveAvgPool2d → Flatten\n(B, 576)"]
        B1 --> B2
    end
    BACKBONE --> EH["Eyes Head\nLinear 576→128→2\n→ p_eyes_closed"]
    BACKBONE --> MH["Mouth Head\nLinear 576→128→2\n→ p_mouth_open"]
```

**Training strategy:**
- Stage 1 (epochs 1–10): frozen backbone, head only, LR=1e-3
- Stage 2 (epochs 11–25): unfreeze last 2 backbone blocks, differential LR (backbone=1e-4, head=1e-3), MixUp(α=0.4)
- Early stopping: patience=5 on val loss

**Binary classification per ROI:**

| ROI | Positive (0) | Negative (1) | Decision |
|-----|-------------|-------------|---------|
| Eyes | `eyes_closed` | `eyes_open` | `p_pos > 0.5` |
| Mouth | `mouth_open` | `mouth_closed` | `p_pos > 0.5` |

**Why binary:** Fine-grained states (partially closed, talking vs yawning) are captured by temporal patterns downstream. Binary keeps noise ceiling manageable and training data requirements low.

---

### Head — HeadPoseExtractor (keypoint geometry)

No CNN. Behavioral features derived geometrically from YOLO11n-pose face keypoints.

```mermaid
flowchart LR
    KP["5 face keypoints\nnose · eyes · ears"] --> PNP["PnP Solver\nvs canonical 3D face model"]
    PNP --> ANGLES["pitch\nyaw\nroll"]
```

| Feature | Fatigue signal |
|---------|---------------|
| Pitch | Head drooping forward (pitch < −15°) |
| Yaw | Looking away from screen |
| Roll | Lateral head tilt — postural instability |

---

### Torso — TorsoPostureExtractor (keypoint geometry)

No CNN. Derived from shoulder and hip keypoints.

```mermaid
flowchart LR
    KP["Shoulder + hip\nkeypoints"] --> GEOM["Geometric computation"]
    GEOM --> SR["Slouch ratio\nvertical shoulder-hip distance\nvs session baseline"]
    GEOM --> LA["Lean angle\nleft vs right shoulder height\nasymmetry"]
```

---

## Phase C: Temporal Modeling

Frame-level feature vectors organized into sliding windows for behavioral pattern detection.

### Feature Vector per Frame (13-dim)

```
┌─────────────┬─────────────┬─────────────────────────────┬──────────────────────────────┐
│  p_eyes_    │  p_mouth_   │   head_pitch  head_yaw       │  conf_eyes  conf_mouth        │
│  closed     │  open       │   head_roll   slouch_ratio   │  conf_head  conf_torso        │
│  (1 dim)    │  (1 dim)    │   lean_angle  (5 dims)       │  (4 dims)                     │
└─────────────┴─────────────┴─────────────────────────────┴──────────────────────────────┘
      CNN features (2)            Keypoint geometry (5)            YOLO confidence (4)
                                    total: 13 dims
```

Zero-padded for any occluded/missing ROI slot.

### Windowing Strategy

```mermaid
gantt
    title Sliding Window Scales (example at 15fps)
    dateFormat  x
    axisFormat %Ls

    section Short window
    Window 1     :0, 1000
    Window 2     :500, 1500
    Window 3     :1000, 2000

    section Moderate window
    Window A     :0, 3000
    Window B     :1500, 4500
```

| Window Scale | Duration | Frames @ 15fps | Target Events |
|---|---|---|---|
| Short | 250ms–1,000ms | 4–15 frames | Blinks, microsleeps, sudden head drops |
| Moderate | 2,000ms–5,000ms | 30–75 frames | PERCLOS, yawn frequency, postural slump |

- **Overlap:** 30–50% between successive windows
- **PERCLOS:** % frames with `p_eyes_closed > 0.5` in window — primary fatigue indicator

### Temporal Model: BiGRU

**Why BiGRU over BiLSTM:** Comparable accuracy on short sequences, ~30% fewer parameters, faster inference. GRU fuses forget/input gates into single update gate — fewer ops per step.

```mermaid
flowchart TD
    WIN["Sliding window matrix\n(T × 13)"] --> GRU1
    subgraph BIGRU["BiGRU Stack"]
        GRU1["BiGRU Layer 1\n64 units/direction → 128 total\nDropout 0.3"]
        GRU2["BiGRU Layer 2\n64 units/direction → 128 total\nDropout 0.3"]
        GRU1 --> GRU2
    end
    GRU2 --> HS["Final hidden state\n(128-dim)"]
    HS --> FC["FC head\nLinear 128→1 · Sigmoid"]
    FC --> FS["Focus Score  [0.0 – 1.0]"]
```

| Component | Detail |
|-----------|--------|
| Layers | 2 stacked BiGRU |
| Hidden size | 64 per direction (128 total) |
| Dropout | 0.3 between layers |
| Loss | MSE (regression on Focus Score) |
| Optimizer | Adam, LR=1e-3, ReduceLROnPlateau |
| Augmentation | ±1–2 frame temporal jitter |

### ACAMF — Optional Occlusion Module

By default, features feed directly into the BiGRU. The BiGRU's gating implicitly learns to suppress noisy/zero-padded inputs given sufficient training examples with occlusion.

```mermaid
flowchart TD
    WIN["Window matrix\n(T × 13)"]

    WIN -->|"use_acamf: false\n(default)"| GRU["BiGRU"]

    WIN -->|"use_acamf: true"| ACAMF
    subgraph ACAMF["ACAMF Module"]
        OR["Compute occlusion ratio\nper ROI stream"]
        SW["stream_weight = 1 − occlusion_ratio"]
        SC["Scale each ROI stream\nby its weight"]
        OR --> SW --> SC
    end
    ACAMF --> GRU
```

Enable when:
- Occlusion is frequent in deployment but underrepresented in training data
- BiGRU shows degraded Focus Score stability during occluded frames

**Default:** disabled. Toggle via `use_acamf: true` in config.

---

## Phase D: Output and GUI

```mermaid
flowchart LR
    FS["Raw Focus Score"] --> EMA["EMA Smoothing\nα = 0.15"]
    EMA --> GAUGE["Focus Gauge\nreal-time"]
    EMA --> TIMELINE["Session Timeline\nscrollable history"]
    EMA --> THRESH{Score < 0.40?}
    THRESH -->|"sustained > 1 min"| SOFT["Soft break prompt"]
    THRESH -->|"sustained > 3 min"| HARD["Intrusive alert"]

    subgraph PANEL["Indicator Panel"]
        P1["PERCLOS"]
        P2["Yawn frequency"]
        P3["Head pose stability"]
        P4["Posture deviation"]
    end
    EMA --> PANEL
```

### Performance Targets

| Metric | Target |
|---|---|
| Processing rate | 15fps (every 2nd frame) |
| RTF | < 1 (CPU viable) |
| YOLO11n-pose latency | ~20–30ms/frame CPU |
| CNN latency (Eyes + Mouth) | ~5–10ms combined |
| BiGRU latency | < 5ms |
| Total deployed footprint | ≤ 20MB |

---

## Labeling Pipeline (Eyes)

```mermaid
flowchart TD
    TR["training_data/data/train\n{Closed, Open}/"] --> TRAIN["Train BinaryROIClassifier\n2-stage frozen → partial unfreeze"]
    TRAIN --> CKPT["runs/binary/eyes/best.pt"]
    CKPT --> INF["scripts/label_eyes.py\ninference on datasets/Eyes/images/"]
    INF --> THRESH{Confidence}
    THRESH -->|"p_pos > 0.5"| CL["eyes_closed"]
    THRESH -->|"p_neg > 0.5"| OP["eyes_open"]
    THRESH -->|"neither"| SKIP["skip — ambiguous"]
    CL --> OUT["datasets/Eyes/eyes_labels.txt\nstem  label  p_pos"]
    OP --> OUT
```

---

> **Design Principle:** Binary frame-level classification + temporal pattern detection separates concerns cleanly. The CNN answers "what is visible now" — the BiGRU answers "what behavioral pattern is emerging over time."
