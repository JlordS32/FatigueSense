"""
Eyes Labeler

Runs BinaryROIClassifier on raw crops at datasets/Eyes/images/.
Writes per-image binary labels to datasets/Eyes/eyes_labels.txt.

Output format:
    filename  label  p_positive

Usage:
    python scripts/label_eyes.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.models.binary_roi_classifier import build_binary_classifier

CHECKPOINT_PATH = Path("runs/binary/eyes/best.pt")
IMAGES_DIR = Path("datasets/Eyes/images")
OUTPUT_PATH = Path("datasets/Eyes/eyes_labels.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.5

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def main() -> None:
    print(f"Device: {DEVICE}")

    model = build_binary_classifier(pretrained=False, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded: {CHECKPOINT_PATH}")

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    print(f"Images: {len(image_paths)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f, torch.no_grad():
        for img_path in tqdm(image_paths, desc="Labeling"):
            img = (
                TRANSFORM(Image.open(img_path).convert("L").convert("RGB"))
                .unsqueeze(0)
                .to(DEVICE)
            )
            probs = torch.softmax(model(img), dim=-1)
            p_pos = probs[0, 0].item()
            p_neg = probs[0, 1].item()
            print(probs)

            if p_pos > CONFIDENCE_THRESHOLD:
                label = "eyes_closed"
            elif p_neg > CONFIDENCE_THRESHOLD:
                label = "eyes_open"
            else:
                continue

            f.write(f"{img_path.stem} {label} {p_pos:.4f}\n")

    print(f"Labels written to {OUTPUT_PATH}")

    counts: Counter = Counter()
    with OUTPUT_PATH.open() as f:
        for line in f:
            counts[line.split()[1]] += 1
    for label, count in sorted(counts.items()):
        print(f"  {label:<15} {count:>5}  ({100 * count / len(image_paths):.1f}%)")


if __name__ == "__main__":
    main()
