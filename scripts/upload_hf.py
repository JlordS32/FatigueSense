from pathlib import Path

from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

# Define the local folder containing all your vid_XXX folders
DATA_FOLDER = Path(__file__).parent.parent / "dataset"

print(
    "Starting massive upload... This might take a while depending on your internet speed."
)

# Upload the entire folder structure directly to your dataset repository
api.upload_large_folder(
    folder_path=DATA_FOLDER,
    repo_id="Jlords32/fatigue-sense",
    repo_type="dataset",
)

print("Upload complete! Check your Hugging Face repository.")
