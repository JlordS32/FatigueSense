# Research Proposal: FatigueSense
### AI Detection of Mental Fatigue Using Multi-Modal Behavior Analysis

## 1. Introduction and Problem Statement
Mental fatigue—a biological state resulting from prolonged cognitive load or sleep deprivation—drastically impairs reaction times and decision-making. While the modern workforce is increasingly susceptible to cognitive decline during desk work, existing solutions are often:
* **Intrusive:** Requiring wearable EEG or heart-rate monitors.
* **Over-simplified:** Relying on single-metric thresholds that fail to capture the nuanced, multi-modal nature of human exhaustion.

**FatigueSense** proposes a non-intrusive, real-time Vision AI system that utilizes a hybrid deep learning architecture to monitor alertness and provide proactive productivity recommendations.

---

## 2. Research Objectives
1. **Lightweight Pipeline:** Develop a multi-modal system capable of real-time spatial and temporal inference on average laptop CPU hardware.
2. **Probabilistic Modeling:** Move beyond discrete $argmax$ classification to a fusion approach that captures the "gradual transition" of fatigue.
3. **Edge Optimization:** Achieve real-time inference (RTF < 1) without GPU, targeting 15fps via frame skipping on standard laptops.

---

## 3. Proposed Methodology and Architecture

The system is structured into a three-phase pipeline:

### Phase A: Spatial Perception and Structural Extraction

* **Model:** **YOLO11n-pose** (nano pose variant) — single forward pass produces both bounding boxes and 17 COCO keypoints per detected person.
* **Eyes + Mouth:** Bounding box crops fed into binary CNN classifiers (Phase B).
* **Head:** 5 face keypoints fed into a `HeadPoseExtractor` — computes pitch, yaw, and roll via Perspective-n-Point (PnP) solver. No CNN required.
* **Torso:** Shoulder and hip keypoints fed into a `TorsoPostureExtractor` — computes slouch ratio and lean angle geometrically. No CNN required.
* **Frame rate:** 15fps processing (every 2nd frame at 30fps source) to meet CPU real-time budget.

### Phase B: CNN Binary Classification (Eyes + Mouth only)

Lightweight **MobileNetV3-Small** classifiers (shared backbone, two independent heads) classify Eye and Mouth crops:

| ROI | Classes |
|-----|---------|
| Eyes | `eyes_closed`, `eyes_open` |
| Mouth | `mouth_open` (yawn), `mouth_closed` |

Binary classification per frame. Temporal patterns (blink rate, yawn frequency, PERCLOS) are computed downstream by the BiGRU, not inferred from single frames.

### Phase C: Temporal Modeling and Adaptive Fusion

Per-frame feature vectors (CNN softmax + keypoint geometry + YOLO confidence scores) are organized into sliding windows and processed by the temporal model:

* **ACAMF Fusion:** The **Adaptive Confidence-Aware Multimodal Fusion** module computes per-ROI occlusion ratios and dynamically down-weights unreliable streams before temporal modeling.
* **BiGRU Decision Head:** A Bidirectional GRU (lighter than BiLSTM, comparable accuracy on short sequences) performs temporal reasoning across sliding windows, reducing frame-level jitter and capturing progressive fatigue onset.
* **Key Metrics:** PERCLOS (% frames eyes_closed in window), yawn frequency, head pitch deviation, postural slump ratio.
* **Output:** Continuous **Focus Score** ($0.0$ = severely fatigued, $1.0$ = fully alert).

---

## 4. Data Collection and Training

* **Binary Classifiers (Eyes/Mouth):** Trained on public datasets (MRL+CEW for eyes, 4-class Drowsiness for mouth). Pseudo-labeling pipeline auto-labels raw FatigueSense video crops.
* **Temporal Model:** Proprietary video recordings of participants in natural and simulated fatigue states. Frame sequences labeled with continuous Focus Score ($0.0$–$1.0$).
* **Optimization:** Adam optimizer with MSE loss for the regression task.

---

## 5. Expected Outcomes

* **Transparency Dashboard:** GUI visualizing real-time Focus Score and per-indicator breakdowns (PERCLOS, yawn count, head pose stability, posture deviation).
* **Automated Notifications:** Personalized break recommendations triggered when Focus Score falls below threshold (default 0.40) for sustained duration.
* **Performance:** Real-time CPU inference at 15fps (RTF < 1) on average laptop hardware. Total deployed model footprint ≤ 20MB.
