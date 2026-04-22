# MediaPipe + Region Cropping Plan for Fatigue Detection

## Goal
Build a reliable preprocessing pipeline that uses **MediaPipe Face Mesh** to detect facial landmarks, then crops the **left eye**, **right eye**, and **mouth** regions from each frame for use in a fatigue detection model.

---

## Overall Strategy
Use MediaPipe to detect facial landmarks on each video frame, convert those landmark points into pixel coordinates, and define padded bounding boxes for the regions of interest. Save or stream the cropped regions into the learning pipeline, then use them as input to a CNN or CNN + temporal model.

---

## Phase 1: Set Up the Environment

### Objective
Prepare the tools and dependencies needed for face landmark detection and image processing.

### Tasks
- Install Python packages:
  - `mediapipe`
  - `opencv-python`
  - `numpy`
  - optionally `matplotlib` for debugging visualizations
- Prepare a small sample dataset of fatigue-related videos or webcam recordings.
- Organize the project folders, for example:
  - `data/raw_videos/`
  - `data/frames/`
  - `data/crops/eyes/`
  - `data/crops/mouth/`
  - `scripts/`

### Deliverable
A working environment that can read video frames and run MediaPipe successfully.

---

## Phase 2: Detect Face Landmarks with MediaPipe

### Objective
Extract facial landmarks from each frame.

### Tasks
- Load each video or camera stream frame by frame.
- Run **MediaPipe Face Mesh** on each frame.
- Convert normalized landmark coordinates into image pixel coordinates using:
  - `x_pixel = int(landmark.x * image_width)`
  - `y_pixel = int(landmark.y * image_height)`
- Verify detection visually by drawing landmarks on sample frames.

### Notes
- Start with one face only if the project assumes a single subject.
- Track failure cases such as poor lighting, occlusion, or strong head rotation.

### Deliverable
A script that successfully detects and visualizes face landmarks on frames.

---

## Phase 3: Define Landmark Groups for Each Region

### Objective
Select the landmark indices for the regions relevant to fatigue detection.

### Target Regions
- Left eye
- Right eye
- Mouth

### Tasks
- Identify the MediaPipe landmark indices for:
  - eyelids and eye contour
  - lip contour
- Store these indices in reusable Python lists or config variables.
- Keep the design modular so region definitions can be changed later.

### Deliverable
A reusable region definition file or code block containing landmark index groups.

---

## Phase 4: Convert Landmarks into Cropping Boxes

### Objective
Turn region landmarks into crop coordinates.

### Tasks
For each region:
- Gather all x and y values from the selected landmarks.
- Compute:
  - `x_min`, `x_max`
  - `y_min`, `y_max`
- Compute the width and height of the region.
- Add padding around the region so the crop is not too tight.

### Recommended Padding
- Horizontal padding: about 20% to 30% of region width
- Vertical padding: about 20% to 30% of region height

### Boundary Handling
Clamp the crop box so it stays inside the image boundaries:
- `x1 = max(0, x1)`
- `y1 = max(0, y1)`
- `x2 = min(image_width, x2)`
- `y2 = min(image_height, y2)`

### Deliverable
A function that returns a valid crop box for each facial region.

---

## Phase 5: Crop and Resize the Regions

### Objective
Extract the eye and mouth patches in a fixed size suitable for model training.

### Tasks
- Crop the regions from the **original image**, not the landmark visualization.
- Resize each crop to a fixed shape.

### Suggested Sizes
- Eye crop: `64 x 64` or `96 x 96`
- Mouth crop: `96 x 96` or `128 x 128`

### Notes
- Keep the input size consistent across the whole dataset.
- Save the crops for inspection during early development.

### Deliverable
A script that outputs aligned, consistently sized eye and mouth images.

---

## Phase 6: Stabilize the Crops Across Time

### Objective
Reduce frame-to-frame jitter caused by small landmark movement.

### Tasks
- Smooth landmark coordinates or bounding boxes using:
  - moving average
  - exponential smoothing
  - reuse of previous valid coordinates
- Compare raw crops vs smoothed crops visually.

### Why This Matters
Jittery crops can make temporal learning harder because the model sees motion from box instability rather than real eye or mouth behavior.

### Deliverable
A more stable sequence of crops across consecutive frames.

---

## Phase 7: Handle Missing or Failed Detections

### Objective
Make the pipeline robust when MediaPipe misses a face or region.

### Tasks
Implement fallback logic such as:
- reuse the previous valid crop box
- skip the current frame
- mark the frame as invalid for later filtering
- optionally interpolate between nearby valid frames

### Recommendation
For the first version, reuse the previous valid crop box when possible.

### Deliverable
A cropping pipeline that does not break when landmark detection fails temporarily.

---

## Phase 8: Save Outputs for Training

### Objective
Prepare the cropped data for model development.

### Options

#### Option A: Offline Preprocessing
- Extract and save all crops before training.
- Easier to debug and faster during model training.

#### Option B: Online Preprocessing
- Detect landmarks and crop during training or inference.
- More flexible but computationally heavier.

### Recommendation
Start with **offline preprocessing**.

### Deliverable
A structured dataset of cropped eye and mouth images or sequences.

---

## Phase 9: Connect Crops to the Model Pipeline

### Objective
Use the cropped regions as model input for fatigue classification.

### Possible Modeling Paths

#### Path 1: Per-frame classification
- Input eye and mouth crops into a CNN
- Predict whether the frame looks alert or fatigued

#### Path 2: Sequence modeling
- Extract crops for a sliding window of frames
- Pass each frame through a CNN feature extractor
- Feed the feature sequence into:
  - LSTM
  - GRU
  - Temporal CNN
- Output a sequence-level prediction such as `alert`, `slightly tired`, or `fatigued`

### Recommendation
Because fatigue is temporal, use **CNN + LSTM/GRU** after the cropping pipeline is stable.

### Deliverable
A complete preprocessing-to-model training pathway.

---

## Real-Time Optimization Notes

### Objective
Keep the MediaPipe + cropping pipeline fast enough for live webcam or video inference.

### Key Point
MediaPipe is usually lightweight enough for real-time face landmarking, especially when the system tracks **one face** and the downstream model is kept small. In many cases, the heavier part is not MediaPipe itself but the extra CNN or temporal model you run after cropping.

### Practical Optimization Steps
- Limit the pipeline to **one face only**.
- Use a moderate frame size such as `640 x 480` instead of very high-resolution frames.
- Crop only the regions you actually need:
  - left eye
  - right eye
  - mouth
- Keep the CNN small in the first version.
- Use a short temporal buffer instead of a very deep sequence model.
- Turn off any outputs you do not need.
- Reuse previous crop boxes briefly if detection is lost.

### Recommended Real-Time Strategy
A practical live pipeline could be:
1. Capture webcam frame.
2. Run MediaPipe Face Mesh on the frame.
3. Extract left eye, right eye, and mouth crop boxes.
4. Resize crops to fixed sizes.
5. Compute either:
   - simple geometric fatigue measures, or
   - lightweight CNN features.
6. Smooth predictions across time.
7. Trigger fatigue output only when the signal is sustained over several frames.

### Extra Efficiency Ideas
- Run landmark detection on every frame, but run the heavier classifier every 2 to 3 frames.
- Use grayscale for downstream CNN input if color is not important.
- Cache recent predictions and apply temporal smoothing instead of expensive reprocessing.
- Start with handcrafted signals like eye openness and mouth opening before adding a full sequence model.

### Warning
A system can fail to feel real-time even if MediaPipe itself is fast enough. Common causes are:
- too large video resolution
- expensive CNN backbone
- running a sequence model every frame
- saving images to disk during live inference
- drawing too many debug overlays

### Deliverable
A version of the pipeline that can run live on webcam input with stable region crops and lightweight fatigue predictions.
