import os

import cv2
from dotenv import load_dotenv

load_dotenv()


def extract_frames(
    video_path,
    output_dir="./data/images",
    metadata_path=None,
    frame_step=15,
    extension=".png",
):
    """
    Extract frames from a video and save them to the specified output directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        metadata_path (str, optional): Path to a metadata file for verification.
        frame_step (int): Step size for frame extraction (e.g., 15 means every 15th frame).
        extension (str): File extension for saved frames (e.g., .png, .jpg, .jpeg).
    """
    if extension not in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Unsupported file extension. Use .png, .jpg, or .jpeg.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count, success = 0, True
    while success:
        success, frame = cap.read()
        if success:
            if frame_count % frame_step == 0:
                frame_filename = os.path.join(
                    output_dir, f"frame_{frame_count:06d}{extension}"
                )
                cv2.imwrite(frame_filename, frame)
            frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")

    if metadata_path and os.path.exists(metadata_path):
        found_images = []
        missing_images = []

        with open(metadata_path, "r") as f:
            meta_data = {line.strip() for line in f}

        for image in os.listdir(output_dir):
            formatted_target = f"{output_dir}/{image}"

            if formatted_target in meta_data:
                found_images.append(image)
            else:
                missing_images.append(image)

        print(f"Found images: {found_images}")
        print(f"Missing images: {missing_images}")
    else:
        print("No metadata path provided, skipping metadata verification...")


if __name__ == "__main__":
    from clearml import Task, Dataset

    task = Task.init(project_name="Main", task_name="Frame Extraction")
    args = {
        "video_path": "videos/2026-03-23 07-57-07.mp4",
        "output_dir": "data/images",
        "metadata_path": "dataset_1/train.txt",
        "frame_step": 13,
        "extension": ".png",
    }

    ds = Dataset.create(
        dataset_name="fatigue_raw_frames",
        dataset_project="FatigueSense",
        dataset_version="1.0",
    )

    task.connect(args)

    # Start frame extraction
    extract_frames(**args)

    # Upload the extracted frames to the ClearML dataset
    ds.add_files(args["output_dir"])
    ds.upload()
    ds.finalize()
