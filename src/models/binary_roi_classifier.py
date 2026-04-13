"""
Binary ROI Classifier — Bootstrap Stage

Shared architecture for all four ROI binary classifiers (Eyes, Mouth, Head, Torso).
Same MobileNetV3-Small backbone as MultiHeadCNN; single two-class head per instance.

Outputs [p_positive, p_negative] via Softmax. ThresholdPredictor maps confidence
scores to final three-way labels (positive / intermediate / negative) using
configurable per-ROI thresholds.

See docs/binary_classifier_bootstrap.md for the full pseudo-label pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROIName = Literal["eyes", "mouth", "head", "torso"]

# ---------------------------------------------------------------------------
# Per-ROI threshold config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdConfig:
    low: float   # p_positive < low  → negative (confident)
    high: float  # p_positive > high → positive (confident)
    # Between low and high → intermediate (review queue or auto-labeled intermediate)


THRESHOLD_DEFAULTS: dict[ROIName, ThresholdConfig] = {
    "eyes":  ThresholdConfig(low=0.15, high=0.85),
    "mouth": ThresholdConfig(low=0.15, high=0.85),
    "head":  ThresholdConfig(low=0.20, high=0.80),
    "torso": ThresholdConfig(low=0.20, high=0.80),
}

# Intermediate class labels emitted when p_positive is in the ambiguous zone
INTERMEDIATE_CLASS: dict[ROIName, str | None] = {
    "eyes":  "eyes_partially_closed",  # auto-labeled
    "mouth": "mouth_slight_open",       # auto-labeled
    "head":  None,                      # → review queue
    "torso": None,                      # → review queue
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BinaryROIClassifier(nn.Module):
    """
    Shared-architecture binary classifier for a single ROI stream.

    Forward contract
    ----------------
    Input : (B, 3, H, W)  — ROI crop, any spatial size (AdaptiveAvgPool handles it)
    Output: (B, 2)         — Softmax [p_positive, p_negative]

    Instantiate once per ROI; weights are independent.
    """

    BACKBONE_OUT_FEATURES = 576

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_OUT_FEATURES, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 2) -> None:
        """Unfreeze the last N inverted residual blocks for Stage 2 fine-tuning."""
        blocks = list(self.backbone.children())
        for block in blocks[-last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        logits = self.head(x)
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Threshold predictor — maps softmax scores to final labels
# ---------------------------------------------------------------------------

PredictedLabel = Literal["positive", "negative", "intermediate"]


class ThresholdPredictor:
    """
    Converts BinaryROIClassifier softmax output to three-way labels.

    Positive  : p_positive > config.high  (confident, auto-label as positive class)
    Negative  : p_positive < config.low   (confident, auto-label as negative class)
    Intermediate: otherwise               (review queue, or intermediate class if defined)

    Usage
    -----
    predictor = ThresholdPredictor("eyes")
    probs = model(crop_batch)          # (B, 2)
    labels, masks = predictor.predict(probs)
    """

    def __init__(self, roi: ROIName, config: ThresholdConfig | None = None) -> None:
        self.roi = roi
        self.config = config or THRESHOLD_DEFAULTS[roi]
        self.intermediate_class = INTERMEDIATE_CLASS[roi]

    def predict(
        self, probs: torch.Tensor
    ) -> tuple[list[PredictedLabel], dict[str, torch.Tensor]]:
        """
        Args:
            probs: (B, 2) softmax output from BinaryROIClassifier

        Returns:
            labels : list[PredictedLabel] — one per sample
            masks  : dict with boolean tensors:
                       "confident"     — auto-label candidates
                       "intermediate"  — review queue candidates
        """
        p_pos = probs[:, 0]

        positive_mask = p_pos > self.config.high
        negative_mask = p_pos < self.config.low
        intermediate_mask = ~positive_mask & ~negative_mask
        confident_mask = positive_mask | negative_mask

        labels: list[PredictedLabel] = []
        for i in range(probs.size(0)):
            if positive_mask[i]:
                labels.append("positive")
            elif negative_mask[i]:
                labels.append("negative")
            else:
                labels.append("intermediate")

        return labels, {
            "confident": confident_mask,
            "intermediate": intermediate_mask,
            "positive": positive_mask,
            "negative": negative_mask,
        }

    def resolve_class_name(
        self,
        label: PredictedLabel,
        positive_class: str,
        negative_class: str,
    ) -> str | None:
        """
        Map PredictedLabel to a final string class name.

        Returns None for intermediate crops that have no defined auto-label
        (i.e., head and torso) — these go to the manual review queue.
        """
        if label == "positive":
            return positive_class
        if label == "negative":
            return negative_class
        return self.intermediate_class  # None for head/torso → review queue


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_binary_classifier(
    roi: ROIName,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> BinaryROIClassifier:
    """Construct a fresh BinaryROIClassifier for the given ROI."""
    return BinaryROIClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for roi in ("eyes", "mouth", "head", "torso"):
        model = build_binary_classifier(roi)
        model.eval()

        # Spatial sizes match roi_config.yaml: (W, H) → tensor (B, 3, H, W)
        size_map = {
            "eyes":  (1, 3, 32, 64),
            "mouth": (1, 3, 64, 64),
            "head":  (1, 3, 128, 128),
            "torso": (1, 3, 128, 128),
        }
        dummy = torch.randn(*size_map[roi])

        with torch.no_grad():
            probs = model(dummy)

        predictor = ThresholdPredictor(roi)
        labels, masks = predictor.predict(probs)

        print(
            f"{roi:6s} | probs={probs.squeeze().tolist()} "
            f"| label={labels[0]} "
            f"| intermediate_class={predictor.intermediate_class}"
        )
