"""
Eyes ROI Dataset
Loads Closed_Eyes / Open_Eyes crops from training_data/Eyes/.
Label 0 = eyes_closed (positive), Label 1 = eyes_open (negative).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

_EYES_INPUT_H = 32
_EYES_INPUT_W = 64

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_CLASS_TO_LABEL: dict[str, int] = {
    "Closed_Eyes": 0,
    "Open_Eyes": 1,
}


class EyesDataset(Dataset):
    def __init__(self, root: str | Path, transform=None) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        root = Path(root)
        for class_name, label in _CLASS_TO_LABEL.items():
            class_dir = root / class_name
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(
    root: str | Path,
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    import torch

    root = Path(root)
    all_samples = EyesDataset(root, transform=None).samples

    total = len(all_samples)
    val_size = int(total * val_split)
    train_size = total - val_size

    indices = torch.randperm(total).tolist()
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_dataset = EyesDataset(root, transform=_TRAIN_TRANSFORMS)
    val_dataset = EyesDataset(root, transform=_VAL_TRANSFORMS)

    train_dataset.samples = [all_samples[i] for i in train_indices]
    val_dataset.samples = [all_samples[i] for i in val_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
