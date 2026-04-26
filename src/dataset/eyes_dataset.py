"""
Eyes ROI Dataset
Loads closed / open crops from dataset/eyes/.
Label 0 = closed, Label 1 = open.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

_EYES_INPUT_H = 64
_EYES_INPUT_W = 64

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_VAL_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class EyesDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        class_to_label: dict[str, int],
        transform=None,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        root = Path(root)
        for class_name, label in class_to_label.items():
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
    class_to_label: dict[str, int],
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    sampler: WeightedRandomSampler | None = None,
    downsample_majority: bool = False,
) -> tuple[DataLoader, DataLoader]:
    import random
    root = Path(root)
    all_samples = EyesDataset(root, class_to_label, transform=None).samples

    # --- Downsample majority to match minority count ---
    if downsample_majority:
        from collections import Counter
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, (_, label) in enumerate(all_samples):
            label_to_indices[label].append(i)

        min_count = min(len(idxs) for idxs in label_to_indices.values())
        print(f"  Downsampling all classes to {min_count} samples")

        balanced_indices = []
        for label, idxs in label_to_indices.items():
            random.seed(42)
            sampled = random.sample(idxs, min_count)
            balanced_indices.extend(sampled)

        all_samples = [all_samples[i] for i in balanced_indices]

    # --- Stratified split ---
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, (_, label) in enumerate(all_samples):
        label_to_indices[label].append(i)

    train_indices: list[int] = []
    val_indices:   list[int] = []

    for label, idxs in label_to_indices.items():
        idxs_tensor = torch.tensor(idxs)[torch.randperm(len(idxs))].tolist()
        n_val = max(1, int(len(idxs_tensor) * val_split))
        val_indices.extend(idxs_tensor[:n_val])
        train_indices.extend(idxs_tensor[n_val:])

    train_dataset = EyesDataset(root, class_to_label, transform=_TRAIN_TRANSFORMS)
    val_dataset   = EyesDataset(root, class_to_label, transform=_VAL_TRANSFORMS)

    train_dataset.samples = [all_samples[i] for i in train_indices]
    val_dataset.samples   = [all_samples[i] for i in val_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
    return train_loader, val_loader, train_indices