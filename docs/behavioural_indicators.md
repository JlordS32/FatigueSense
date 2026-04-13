Based on the provided research and project documentation, the following table summarizes potential behavioral features for detecting mental fatigue and cognitive decline, organized by body region.

## Behavioral Indicators for Fatigue and Focus Monitoring

| Body Region | Feature / Indicator | Measurement / Threshold | Significance as Fatigue Indicator |
| :--- | :--- | :--- | :--- |
| **Eyes** | PERCLOS (Percentage of Eye Closure) | Percentage of time eyelids are closed (typically 80% closure) over a sliding window. | Most reliable vision-based indicator; thresholds around 0.8 distinguish between alert and drowsy states. |
| | Eye Aspect Ratio (EAR) | Ratio of vertical eyelid distances to horizontal eye span. | Normal alert values range between 0.25 and 0.35; values below 0.20 or 0.25 indicate drowsiness. |
| | Blink Frequency / Rate | Count of blinks over a one-minute interval. | Normal persons blink more than 10 times per minute; frequency decreases as fatigue sets in. |
| | Blink Duration | The length of time eyelids remain closed during a standard blink. | Normal blinks last 300–400 ms; average duration increases significantly during fatigue. |
| | Microsleeps | Sustained eye closure/EAR depression. | Detected when closure persists for 500 ms to 1 second (approx. 15–30 consecutive frames). |
| | Visual Attention / Gaze | Pupil diameter dynamics and gaze direction vectors. | Narrower gaze distribution and changing pupil dynamics signal declining focus and cognitive load. |
| **Mouth** | Mouth Aspect Ratio (MAR) / MOR | Geometric ratio identifying mouth configuration and openness. | Used to distinguish between resting, talking, and active yawning behavior. |
| | Yawn Frequency / Duration | Sustained high MAR values over a specific frame count. | Yawning events are confirmed when MAR exceeds 0.5 or 0.6 for 20+ frames (~666 ms) or up to 5 seconds. |
| **Head** | Head Nodding Patterns | Pitch angle analysis relative to a stable, neutral position. | Identified when head pitch exceeds ±15 to ±25 degrees for durations exceeding 2 seconds. |
| | Head Pose Instability | Variance in pitch, yaw, and roll angles. | General orientation variance indicates a lack of focus and declining postural control. |
| **Torso** | Torso Slumping | Postural-shifts and collapsing sitting position. | Decline in stability; transition to a collapsed or heavily slouched posture. |
| | Postural Asymmetry | Deviation from a centered, engaged sitting posture. | Semantic indicator of cognitive decline and physical restlessness during prolonged work. |
| | Secondary Activities | Interaction with the environment (off-task actions). | Activities like reaching behind, grooming (hair and makeup), or drinking indicate focus has drifted. |

---

### Summary of Indicators

* **Gradual Transition:** Fatigue is a continuous process; identifying the rising probability of "heavy" states (e.g., rising EAR depression or increasing blink duration) is more effective than binary classification.
* **Multi-Modal Nature:** The "complex, semantic, multi-modal nature of cognitive decline" is best captured by fusing these streams (e.g., combining high PERCLOS with head nodding and torso slumping).
* **Focus Metric:** These behaviors collectively feed into a continuous numerical metric (**Focus Score**) to provide users with transparency regarding their alertness levels.