from clearml import Dataset


def upload_dataset(
    dataset_name,
    dataset_project,
    dataset_version,
    data_path=[],
    parent_dataset_id=None,
    dataset_tags=None,
):
    ds = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=dataset_version,
        parent_datasets=[parent_dataset_id] if parent_dataset_id else None,
        dataset_tags=dataset_tags or [],
    )

    if data_path:
        for path in data_path:
            ds.add_files(path)

    ds.upload()
    ds.finalize()

    print(
        f"Uploaded {data_path} to dataset {dataset_name} (version: {dataset_version})"
    )
    return ds.id
