from upload_dataset import upload_dataset

# Change these values to match your dataset details
args = {
    "dataset_name": "fatigue-region-labels",
    "dataset_project": "FatigueSense",
    "dataset_version": "1.0.0",
    "data_path": ["dataset_1/labels", "dataset_1/data.yaml", "dataset_1/train.txt"],
    "dataset_tags": ["yolo", "4-class"],
}

dataset_id = upload_dataset(**args)
print(f"Dataset ID: {dataset_id}")
