"""
Binary ROI Classifier — Training Script

Stage 1 (epochs 1–10):  frozen backbone, head only, LR=1e-3
Stage 2 (epochs 11–25): unfreeze last 2 backbone blocks, differential LR,
                         backbone=1e-4 / head=1e-3, MixUp(alpha=0.4)

Usage:
    python scripts/train_binary_classifier.py --roi eyes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.dataset.eyes_dataset import build_dataloaders
from src.models.binary_roi_classifier import build_binary_classifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_1_EPOCHS = 10
STAGE_2_EPOCHS = 15
BATCH_SIZE = 64
STAGE_1_LR = 1e-3
STAGE_2_HEAD_LR = 1e-3
STAGE_2_BACKBONE_LR = 1e-4
MIXUP_ALPHA = 0.4
EARLY_STOP_PATIENCE = 5

DATASET_ROOTS: dict[str, str] = {
    "eyes": "training_data/Eyes",
}

CHECKPOINT_DIR = Path("runs/binary")


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()

    # Variables
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Get batch logits and loss
            logits = model(images)
            loss = criterion(logits, labels)

            # Get prediction
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Calculate validation loss, precision, recall and f1
    avg_loss = total_loss / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, precision, recall, f1


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def run_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    start_epoch: int,
    use_mixup: bool,
    checkpoint_path: Path,
) -> int:
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_lose = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup:
                mixed, labels_a, labels_b, lam = mixup_batch(
                    images, labels, MIXUP_ALPHA
                )
                logits = model(mixed)
                loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_lose / len(train_loader)
        val_loss, precision, recall, f1 = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:>3} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch} (best={best_epoch})")
                break

    return best_epoch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train(roi: str) -> None:
    if roi not in DATASET_ROOTS:
        raise ValueError(
            f"No dataset registered for ROI '{roi}'. Add to DATASET_ROOTS."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(
        root=DATASET_ROOTS[roi],
        batch_size=BATCH_SIZE,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = build_binary_classifier(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = CHECKPOINT_DIR / roi
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"

    # Stage 1 — frozen backbone
    print("\n--- Stage 1: frozen backbone ---")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE_1_LR,
    )
    run_stage(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs=STAGE_1_EPOCHS,
        start_epoch=1,
        use_mixup=False,
        checkpoint_path=checkpoint_path,
    )

    # Stage 2 — partial unfreeze + differential LR
    print("\n--- Stage 2: unfreeze last 2 blocks ---")
    model.freeze_backbone(toggle=False, last_n_blocks=2)
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": STAGE_2_BACKBONE_LR},
            {"params": model.head.parameters(), "lr": STAGE_2_HEAD_LR},
        ]
    )
    run_stage(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs=STAGE_2_EPOCHS,
        start_epoch=STAGE_1_EPOCHS + 1,
        use_mixup=True,
        checkpoint_path=checkpoint_path,
    )

    print(f"\nBest checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", default="eyes", choices=list(DATASET_ROOTS.keys()))
    args = parser.parse_args()
    train(args.roi)
