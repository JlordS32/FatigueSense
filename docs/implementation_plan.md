# Implementation Plan for YOLO11 (Standard Detection)

## Phase 1: Spatial Localization and ROI Extraction

The system leverages **YOLO11**, a backbone specifically designed for high-speed, real-time Vision AI tasks on low-power edge devices. Rather than relying on a skeletal mesh, the model is trained to detect and generate bounding boxes for four primary target classes: **Eyes, Mouth, Head, and Torso**.

### Detection Configuration
* **Model Variant:** YOLO11n or YOLO11s (nano/small) selected based on latency profiling; target inference time ≤10ms per frame.
* **Input Resolution:** 640×640 (standard YOLO input); ROI crops are subsequently resized to class-specific dimensions (e.g., 64×32 for eyes, 128×128 for head).
* **Confidence Threshold:** Detections below 0.5 confidence are discarded. If a class is undetected (e.g., mouth occluded), its ROI slot is flagged as missing and handled downstream by ACAMF.
* **Non-Maximum Suppression (NMS):** IoU threshold of 0.45 to suppress duplicate bounding boxes.

### ROI Pre-Processing
Each confirmed bounding box is cropped from the frame and normalized before CNN ingestion:
* Resized to a fixed resolution per ROI class.
* Pixel values normalized to `[0, 1]`.
* A **padding strategy** is applied when the bounding box clips the frame boundary to avoid distorting spatial features.

Because YOLO11 is optimized for edge efficiency, this stage produces clean ROI crops with minimal latency, facilitating a seamless **30 FPS pipeline**.

---

## Phase 2: Lightweight CNN Architecture Research

Before committing to a backbone, a structured evaluation of candidate architectures is conducted to identify the optimal balance between accuracy, parameter count, and inference latency for each ROI stream.

### Candidate Architectures
| Architecture      | Strengths                                                     | Weaknesses                                        |
|-------------------|---------------------------------------------------------------|---------------------------------------------------|
| MobileNetV3       | Depthwise separable convolutions; strong mobile track record  | Less expressive than full CNNs                    |
| GhostNet          | "Ghost modules" generate redundant features cheaply           | Less community support / pretrained weights       |
| EfficientNet-Lite | Strong accuracy/efficiency scaling via compound coefficient   | Slightly higher latency than MobileNet            |
| ShuffleNetV2      | Channel shuffle for cross-group information flow; extremely low FLOPs | Accuracy ceiling lower than EfficientNet  |

### Evaluation Metrics
Each candidate is benchmarked against a consistent set of criteria:
* **Classification accuracy** on a held-out ROI validation set (per-class: eyes, mouth, head, torso).
* **Inference latency** (ms/frame) on target CPU hardware.
* **Parameter count** and **model size (MB)** post-training.
* **FLOPs** per forward pass as a hardware-agnostic efficiency proxy.

### Multimodal-Specific Evaluation
Because each ROI has structurally distinct visual characteristics, the research must assess two competing strategies:

* **Shared Backbone:** A single CNN processes all four ROIs. Lower total parameter count but may underfit fine-grained eye features.
* **Per-ROI Specialized Heads:** Separate lightweight branches per ROI. Higher fidelity but increases model complexity — must still meet the **~0.42M parameter** target via Knowledge Distillation.

### Knowledge Distillation Candidate Selection
The selected architecture must be evaluated as a viable **student model** target. Key criteria:
* Compressible to ~0.42M parameters without significant accuracy degradation.
* Compatible with a Teacher-Student training strategy using a higher-capacity backbone (e.g., ResNet-50 or EfficientNet-B3) as the teacher.
* Teacher model is trained first to convergence; student is then trained to minimize a combined loss of task loss (MSE/Cross-Entropy) and distillation loss (KL divergence on softened logits).

### Phase Output
A justified architecture selection and configuration (shared vs. per-ROI) that produces a **1D probability vector** per ROI — the required input format for Phase 4's temporal pipeline.

---

## Phase 3: CNN-Based Feature Extraction

Each ROI crop produced in Phase 1 is independently processed through the selected lightweight CNN backbone to extract behavioral state probabilities.

### Per-ROI Output Structure
The CNN outputs a **1D Softmax probability vector** for each ROI, representing the likelihood of discrete behavioral states:

| ROI    | Output Classes (Example)                              |
|--------|-------------------------------------------------------|
| Eyes   | Open, Partially Closed, Closed, Microsleep            |
| Mouth  | Neutral, Talking, Yawning                             |
| Head   | Upright, Nodding, Tilted Left, Tilted Right           |
| Torso  | Upright, Slumping, Asymmetric, Off-Task               |

### Inference Batching
To maximize hardware utilization, all four ROI crops are batched into a single forward pass where the shared backbone strategy is used. In the per-ROI head strategy, each branch is executed concurrently.

### Occlusion Flagging
If YOLO11 fails to detect an ROI in a given frame (confidence below threshold), the CNN step is skipped for that slot and a **null vector** (all zeros) is passed downstream. This signals ACAMF in Phase 5 to down-weight that stream accordingly.

### Confidence Score Passthrough
Alongside the Softmax vector, the **YOLO11 detection confidence score** for each ROI is preserved and passed to Phase 5 as part of the ACAMF weighting input.

---

## Phase 4: Temporal Feature Engineering

Frame-level probability vectors are organized into structured time windows for longitudinal behavioral analysis.

### Multi-Scale Windowing Strategy
The system maintains two concurrent temporal window scales to capture behaviors at different time resolutions:

* **Short Windows (250ms–1,000ms):** Target rapid, transient events. Captures microsleeps, individual blink patterns, and sudden head drops.
* **Moderate Windows (2,000ms–5,000ms):** Target sustained, progressive states. Captures head nodding trends, yawn frequency, and postural slumping over time.

### 2D Matrix Construction
Per-frame Softmax vectors are stacked row-by-row to form a **2D matrix** of shape `(Sequence Length × Feature Dimension)`, where:
* **Sequence Length** is determined by the window scale and frame rate (e.g., at 30 FPS, a 1,000ms window = 30 rows).
* **Feature Dimension** is the concatenated length of all four ROI Softmax vectors plus their associated YOLO11 confidence scores.

### Overlapping Windows
A **30% to 50% overlap** is maintained between successive temporal buckets. This ensures behavioral transitions at window boundaries are not missed and provides the BiLSTM with smoother sequential context.

### Normalization
Each column (feature dimension) of the 2D matrix is normalized using a **running min-max scaler** fitted on a calibration window at session start, accounting for per-user baseline variation.

---

## Phase 5: Temporal Modeling and Adaptive Fusion

The 2D matrices from Phase 4 are consumed by the **Bidirectional LSTM (BiLSTM)** decision head, which performs temporal behavioral reasoning across both past and future context within each window.

### BiLSTM Architecture
* **Layers:** 2 stacked BiLSTM layers to capture hierarchical temporal patterns.
* **Hidden Size:** 128 units per direction (256 total per layer).
* **Dropout:** 0.3 applied between layers to reduce overfitting on sequential correlations.
* **Output:** The final hidden state is passed to a fully connected regression head producing a scalar **Focus Score** in `[0.0, 1.0]`.

### Stability and Jitter Reduction
By analyzing behavioral trends across multiple frames rather than reacting to individual predictions, the temporal model reduces frame-level decision variation (jitter) from **0.042 to 0.011**, providing a smooth and trustworthy output signal.

### ACAMF Integration
**Adaptive Confidence-Aware Multimodal Fusion** is applied before the BiLSTM input to handle unreliable ROI streams:
* An **occlusion ratio** is computed per ROI as the proportion of frames in the current window where the YOLO11 confidence fell below threshold.
* Each ROI stream's contribution is scaled by `(1 - occlusion_ratio)` before concatenation, effectively down-weighting noisy or intermittently missing streams.
* This allows the model to remain robust to transient occlusions (e.g., a hand in front of the mouth, glasses glare on the eyes) without discarding the entire window.

### Training Configuration
* **Loss Function:** Mean Squared Error (MSE) for the Focus Score regression task.
* **Optimizer:** Adam with a learning rate of `1e-3`, decayed via ReduceLROnPlateau.
* **Sequence Augmentation:** Random temporal jitter (±1–2 frames) applied during training to improve robustness to frame-rate inconsistencies.

---

## Phase 6: Output and GUI Integration

The final phase converts the BiLSTM's continuous output into actionable, user-facing insights.

### Focus Score Pipeline
* **Regression Output:** The BiLSTM's fully connected head outputs a scalar Focus Score (`0.0` = severely fatigued, `1.0` = fully alert).
* **Smoothing:** A lightweight **exponential moving average (EMA)** is applied to the raw score stream before display to prevent UI flickering from minor frame-to-frame variation.

### GUI Dashboard Components
* **Real-Time Focus Gauge:** Visual representation of the current smoothed Focus Score.
* **Granular Metric Panel:** Per-indicator breakdowns displayed alongside the score — **PERCLOS**, **EAR trend**, **yawn frequency**, and **head pose stability** — to give the user transparency into what is driving their score.
* **Session Timeline:** A scrollable historical graph of the Focus Score over the current session, allowing users to identify when and how fatigue onset occurred.

### Threshold Notifications
* A **personalized fatigue threshold** (default: Focus Score < 0.40) triggers an automated break recommendation.
* Notification severity escalates with duration below threshold: a soft prompt at 1 minute, an intrusive alert at 3 minutes of sustained low focus.

### Performance Targets
* End-to-end pipeline latency: **31ms–35ms per frame** (capture → YOLO11 → CNN → BiLSTM → GUI update).
* Validated CPU-only real-time inference at **27–30 FPS**.
* Total deployed model footprint: **≤14MB**.

---

> **Summary of Benefits:** By utilizing standard YOLO11 for detection, the architecture benefits from a model already optimized for mobile hardware. This approach achieves state-of-the-art accuracy with significantly lower resource consumption than previous YOLO iterations.
