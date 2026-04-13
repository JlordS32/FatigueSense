# Research Proposal: FatigueSense
### AI Detection of Mental Fatigue Using Multi-Modal Behavior Analysis

## 1. Introduction and Problem Statement
Mental fatigue—a biological state resulting from prolonged cognitive load or sleep deprivation—drastically impairs reaction times and decision-making. While the modern workforce is increasingly susceptible to cognitive decline during desk work, existing solutions are often:
* **Intrusive:** Requiring wearable EEG or heart-rate monitors.
* **Over-simplified:** Relying on single-metric thresholds that fail to capture the nuanced, multi-modal nature of human exhaustion.

**FatigueSense** proposes a non-intrusive, real-time Vision AI system that utilizes a hybrid deep learning architecture to monitor alertness and provide proactive productivity recommendations.

---

## 2. Research Objectives
1.  **Lightweight Pipeline:** Develop a multi-modal system capable of real-time spatial and temporal inference.
2.  **Probabilistic Modeling:** Move beyond discrete $argmax$ classification to a fusion approach that captures the "gradual transition" of fatigue.
3.  **Edge Optimization:** Achieve high inference stability on resource-constrained hardware using Knowledge Distillation (aiming for $\approx 0.42M$ parameters).

---

## 3. Proposed Methodology and Architecture

The system is structured into a three-level hierarchy:

### Phase A: Spatial Perception and Structural Extraction
* **Localization:** Employs **YOLO11** for high-speed identification of four key Regions of Interest (ROI): Eyes, Mouth, Head, and Torso.
* **Classification:** ROIs are processed by lightweight CNNs (MobileNetV3/GhostNet) to generate state probabilities.
* **Knowledge Distillation:** A Teacher-Student strategy compresses high-performance models into efficient "student" networks for edge deployment.

### Phase B: Temporal Feature Engineering
The system stacks frame-level Softmax vectors into 2D matrices (Sequence Length $\times$ Features) using sliding windows:
* **Short Windows (250ms–1,000ms):** Captures micro-sleeps and blink patterns.
* **Moderate Windows (2,000ms–5,000ms):** Monitors head-nodding, yawning, and postural slumping.
* **Key Metrics:** PERCLOS (Percentage of Eye Closure), MAR (Mouth Aspect Ratio), and postural asymmetry.

### Phase C: Temporal Modeling and Adaptive Fusion
* **BiLSTM Decision Head:** A Bidirectional LSTM performs temporal reasoning, reducing frame-level jitter from $0.042$ to $0.011$.
* **ACAMF Fusion:** The **Adaptive Confidence-Aware Multimodal Fusion** calculates occlusion ratios to dynamically down-weight noisy inputs (e.g., glasses glare).

---

## 4. Data Collection and Training
* **Dataset:** Proprietary video recordings of participants in both natural and simulated fatigue states.
* **Labeling:** Mapping frame ranges to a continuous **Focus Metric** ($0.0$ to $1.0$).
* **Optimization:** Training via Adam optimizer using **Mean Squared Error (MSE)** loss for the regression task.

---

## 5. Expected Outcomes
* **Transparency Dashboard:** A GUI visualizing real-time focus metrics and behavioral triggers.
* **Automated Notifications:** Personalized break recommendations based on individual fatigue thresholds.
* **Performance:** Validated CPU-only real-time inference at **27–30 FPS**.