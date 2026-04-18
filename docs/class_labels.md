# Class Labels for FatigueSense Cropped Regions

This file defines labeling classes for cropped body-region images.

## What "occluded" Means

Use `occluded` when the target region is not clearly visible enough to assign its true state.

Examples:
- Covered by hand, hair, or object
- Strong glare or blur
- Region is partially or fully outside the crop
- Pose angle hides key visual cues

Rule:
- If the annotator cannot confidently assign a true state in 1-2 seconds, use `occluded`.

## Primary Class Set by Body Region

### Eyes
- `eyes_open`
- `eyes_closed`
- `eyes_occluded`

Optional add-on:
- `gaze_on_screen`
- `gaze_off_screen`

### Mouth
- `mouth_closed`
- `mouth_open`
- `mouth_occluded`

> Binary: `mouth_open` = yawning/wide open. `mouth_closed` = neutral/slight open collapsed into closed.

### Head
- `head_neutral`
- `head_down`
- `head_up`
- `head_left`
- `head_right`
- `head_occluded`

### Torso
- `torso_upright`
- `torso_slight_slouch`
- `torso_heavy_slouch`
- `torso_lean_left`
- `torso_lean_right`
- `torso_occluded`

## Minimum Viable Class Set (Faster Annotation)

If annotation speed is the priority, use this reduced set first:

### Eyes
- `eyes_open`
- `eyes_closed`

### Mouth
- `mouth_closed`
- `mouth_open`

### Head
- `neutral`
- `down`

### Torso
- `upright`
- `heavy_slouch`

## Annotation Principles

- Label what is visible in the current frame only.
- Do not label temporal events directly in per-frame labels.
- Keep labels objective and consistent across annotators.
- Prefer `occluded` over guessing.

## Temporal Events Derived After Labeling

These should be computed from frame sequences, not manually labeled per frame:
- Blink: brief `eyes_closed` sequence in temporal window
- Microsleep: sustained `eyes_closed`
- Yawn: sustained `mouth_open`
- Nodding: repeated `head_down` transitions

## Suggested Label File Format

For region label files (for example `mouth_labels.txt`), use tab-separated columns:

- Column 1: sample ID (for example `vid_001_frame_000130`)
- Column 2: class label (filled by annotator)

Example:

`vid_001_frame_000130\tmouth_open`
