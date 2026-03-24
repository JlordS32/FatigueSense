import os


def verify_files(metadata_path, output_dir):
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
    # Example usage
    verify_files(
        "C:\\Users\\jlord\\.clearml\\cache\\storage_manager\\datasets\\ds_a960a05b7b0c4c9da1cb8cbff790bbad\\vid_002\\metadata\\train.txt",
        "dataset_2/images",
    )
