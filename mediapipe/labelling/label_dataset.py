from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import cv2

from mediapipe_labelling import MediaPipeRegionExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def sanitize_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)



def save_crop(crop, output_path: Path) -> bool:
    if crop is None:
        return False
    return bool(cv2.imwrite(str(output_path), crop))



def process_video_to_region_dataset(
    video_path: str | Path,
    model_path: str | Path,
    output_root: str | Path = "cropped_images",
    sample_every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    flip_horizontal: bool = False,
) -> dict[str, int]:
    """Process a video and save cropped eyes/mouth images into class folders.

    Output structure:
        dataset/
          eyes/
            open/
            closed/
          mouth/
            open/
            closed/

    Notes:
    - Eye labels are taken from the extractor's frame-level eye state.
    - Both left and right eye crops are saved into the same eyes/open or eyes/closed folder.
    - Mouth labels are taken from the extractor's mouth state.
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    label_file_path = output_root / "labels.txt"
    label_file = open(label_file_path, "a")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    extractor = MediaPipeRegionExtractor(model_path=model_path, draw_landmarks=False)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    video_stem = sanitize_stem(video_path)
    frame_index = 0
    processed_frames = 0

    stats = {
        "frames_seen": 0,
        "frames_processed": 0,
        "frames_detected": 0,
        "eyes_open": 0,
        "eyes_closed": 0,
        "mouth_open": 0,
        "mouth_closed": 0,
        "left_eye_saved": 0,
        "right_eye_saved": 0,
        "mouth_saved": 0,
        "skipped_no_face": 0,
    }

    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            stats["frames_seen"] += 1

            if sample_every_n_frames > 1 and frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            outputs = extractor.process_frame(frame)
            processed_frames += 1
            stats["frames_processed"] += 1

            if not outputs.detected:
                stats["skipped_no_face"] += 1
                frame_index += 1
                if max_frames is not None and processed_frames >= max_frames:
                    break
                continue

            stats["frames_detected"] += 1

            ear_left = outputs.ear_left
            ear_right = outputs.ear_right

            left_eye_label = 1 if ear_left is not None and ear_left > 0.21 else 0
            right_eye_label = 1 if ear_right is not None and ear_right > 0.21 else 0
            mouth_label = (outputs.mouth_state or "unknown").lower()

            stats["eyes_open"] += left_eye_label + right_eye_label
            stats["eyes_closed"] += (1 - left_eye_label) + (1 - right_eye_label)

            if mouth_label not in {"open", "closed"}:
                mouth_label = "closed"

            if mouth_label == "open":
                stats["mouth_open"] += 1
            else:
                stats["mouth_closed"] += 1

            sample_id = f"{video_stem}_f{frame_index:06d}"

            

            sample_dir = frames_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            left_eye_path = sample_dir / "left_eye.jpg"
            right_eye_path = sample_dir / "right_eye.jpg"
            mouth_path = sample_dir / "mouth.jpg"

            if save_crop(outputs.left_eye_crop, left_eye_path):
                stats["left_eye_saved"] += 1
            if save_crop(outputs.right_eye_crop, right_eye_path):
                stats["right_eye_saved"] += 1
            if save_crop(outputs.mouth_crop, mouth_path):
                stats["mouth_saved"] += 1

            mouth_val = int(mouth_label == "open")

            label_file.write(f"{sample_id} {left_eye_label} {right_eye_label} {mouth_val}\n")

            frame_index += 1

            if max_frames is not None and processed_frames >= max_frames:
                break
    finally:
        cap.release()
        extractor.close()
        label_file.close()
    return stats

def process_videos_in_directory(
    videos_dir: str | Path,
    model_path: str | Path,
    output_root: str | Path = "cropped_images",
    sample_every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    flip_horizontal: bool = False,
) -> dict[str, dict[str, int]]:
    videos_dir = Path(videos_dir)
    output_root = Path(output_root)

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    video_files = sorted(
        [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    )

    if not video_files:
        raise FileNotFoundError(f"No supported video files found in: {videos_dir}")

    all_results = {}

    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        results = process_video_to_region_dataset(
            video_path=video_path,
            model_path=model_path,
            output_root=output_root,
            sample_every_n_frames=sample_every_n_frames,
            max_frames=max_frames,
            flip_horizontal=flip_horizontal,
        )
        all_results[video_path.name] = results

        print(f"Done: {video_path.name}")
        for key, value in results.items():
            print(f"  {key}: {value}")

    return all_results


if __name__ == "__main__":
    # Example usage:
    # python build_region_dataset.py
    MODEL_PATH = Path("mediapipe/models/face_landmarker.task")
    VIDEOS_DIR = Path("videos")
    OUTPUT_ROOT = Path("cropped_images")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Missing videos directory: {VIDEOS_DIR}")

    all_results = process_videos_in_directory(
        videos_dir=VIDEOS_DIR,
        model_path=MODEL_PATH,
        output_root=OUTPUT_ROOT,
        sample_every_n_frames=15,
        max_frames=None,
        flip_horizontal=False,
    )

    print("\nAll videos done.")
