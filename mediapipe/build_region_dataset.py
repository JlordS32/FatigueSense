from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from mediapipe_region_extractor import MediaPipeRegionExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def ensure_output_dirs(output_root: Path) -> None:
    for subdir in [
        output_root / "eyes" / "open",
        output_root / "eyes" / "closed",
        output_root / "mouth" / "open",
        output_root / "mouth" / "closed",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)



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

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    ensure_output_dirs(output_root)

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

            eye_label = (outputs.eye_state or "unknown").lower()
            mouth_label = (outputs.mouth_state or "unknown").lower()

            if eye_label not in {"open", "closed"}:
                eye_label = "open"
            if mouth_label not in {"open", "closed"}:
                mouth_label = "closed"

            if eye_label == "open":
                stats["eyes_open"] += 1
            else:
                stats["eyes_closed"] += 1

            if mouth_label == "open":
                stats["mouth_open"] += 1
            else:
                stats["mouth_closed"] += 1

            frame_tag = f"{video_stem}_f{frame_index:06d}"

            left_eye_path = output_root / "eyes" / eye_label / f"{frame_tag}_left.jpg"
            right_eye_path = output_root / "eyes" / eye_label / f"{frame_tag}_right.jpg"
            mouth_path = output_root / "mouth" / mouth_label / f"{frame_tag}_mouth.jpg"

            if save_crop(outputs.left_eye_crop, left_eye_path):
                stats["left_eye_saved"] += 1
            if save_crop(outputs.right_eye_crop, right_eye_path):
                stats["right_eye_saved"] += 1
            if save_crop(outputs.mouth_crop, mouth_path):
                stats["mouth_saved"] += 1

            frame_index += 1

            if max_frames is not None and processed_frames >= max_frames:
                break
    finally:
        cap.release()
        extractor.close()

    return stats


if __name__ == "__main__":
    # Example usage:
    # python build_region_dataset.py
    MODEL_PATH = Path("mediapipe/models/face_landmarker.task")
    VIDEO_PATH = Path("videos/vid_009.mp4")
    OUTPUT_ROOT = Path("cropped_images")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Set VIDEO_PATH in __main__ before running. Missing: {VIDEO_PATH}"
        )

    results = process_video_to_region_dataset(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_root=OUTPUT_ROOT,
        sample_every_n_frames=1,
        max_frames=None,
        flip_horizontal=False,
    )

    print("Done")
    for key, value in results.items():
        print(f"{key}: {value}")
