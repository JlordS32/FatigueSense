"""
Threshold Predictor — Pseudo-Label Pipeline

Maps BinaryROIClassifier softmax output to three-way labels
(positive / intermediate / negative) using configurable per-ROI thresholds.

See docs/binary_classifier_bootstrap.md for the full pseudo-label pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ROIName = Literal["eyes", "mouth", "head", "torso"]
PredictedLabel = Literal["positive", "negative", "intermediate"]


# ---------------------------------------------------------------------------
# Per-ROI threshold config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdConfig:
    low: float  # p_positive < low  → negative (confident)
    high: float  # p_positive > high → positive (confident)


THRESHOLD_DEFAULTS: dict[ROIName, ThresholdConfig] = {
    "eyes": ThresholdConfig(low=0.15, high=0.85),
    "mouth": ThresholdConfig(low=0.15, high=0.85),
    "head": ThresholdConfig(low=0.20, high=0.80),
    "torso": ThresholdConfig(low=0.20, high=0.80),
}

# Intermediate label emitted when p_positive is in the ambiguous zone.
# None → crop goes to manual review queue (no auto-label).
INTERMEDIATE_CLASS: dict[ROIName, str | None] = {
    "eyes": "eyes_partially_closed",
    "mouth": "mouth_slight_open",
    "head": None,
    "torso": None,
}


# ---------------------------------------------------------------------------
# Threshold predictor
# ---------------------------------------------------------------------------


class ThresholdPredictor:
    """
    Converts BinaryROIClassifier softmax output to three-way labels.

    Positive    : p_positive > config.high  (confident, auto-label)
    Negative    : p_positive < config.low   (confident, auto-label)
    Intermediate: otherwise                 (review queue, or auto-labeled if defined)

    Usage
    -----
    predictor = ThresholdPredictor("eyes")
    probs = model(crop_batch) # (B, 2)
    labels, masks = predictor.predict(probs)
    """

    def __init__(self, roi: ROIName, config: ThresholdConfig | None = None) -> None:
        self.roi = roi
        self.config = config or THRESHOLD_DEFAULTS[roi]
        self.intermediate_class = INTERMEDIATE_CLASS[roi]

    def predict(
        self, logits: torch.Tensor
    ) -> tuple[list[PredictedLabel], dict[str, torch.Tensor]]:
        """
        Args:
            logits: (B, 2) raw logits from BinaryROIClassifier

        Returns:
            labels : list[PredictedLabel] — one per sample
            masks  : dict with boolean tensors:
                       "confident"     — auto-label candidates
                       "intermediate"  — review queue candidates
                       "positive"      — positive-class candidates
                       "negative"      — negative-class candidates
        """
        probs = torch.softmax(logits, dim=-1)
        p_pos = probs[:, 0]

        positive_mask = p_pos > self.config.high
        negative_mask = p_pos < self.config.low
        intermediate_mask = (
            ~positive_mask & ~negative_mask
        )  # If both pos & neg = 0, 0 -> 1
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

        Returns None for intermediate crops with no defined auto-label
        (head, torso) — these go to the manual review queue.
        """
        if label == "positive":
            return positive_class
        if label == "negative":
            return negative_class
        return self.intermediate_class
