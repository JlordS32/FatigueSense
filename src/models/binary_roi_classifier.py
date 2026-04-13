"""
Binary ROI Classifier — Model
Shared MobileNetV3-Small backbone for all four ROI binary classifiers
(Eyes, Mouth, Head, Torso). Outputs [p_positive, p_negative] via Softmax.

For threshold-based pseudo-labeling, see src/pipeline/threshold_predictor.py.
See docs/binary_classifier_bootstrap.md for the full bootstrap pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


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


def build_binary_classifier(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> BinaryROIClassifier:
    """Construct a fresh BinaryROIClassifier."""
    return BinaryROIClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.pipeline.threshold_predictor import ThresholdPredictor

    size_map = {
        "eyes": (1, 3, 32, 64),
        "mouth": (1, 3, 64, 64),
        "head": (1, 3, 128, 128),
        "torso": (1, 3, 128, 128),
    }

    for roi, shape in size_map.items():
        model = build_binary_classifier()
        model.eval()

        dummy = torch.randn(*shape)
        with torch.no_grad():
            probs = model(dummy)

        predictor = ThresholdPredictor(roi)
        labels, masks = predictor.predict(probs)

        print(
            f"{roi:6s} | probs={probs.squeeze().tolist()} "
            f"| label={labels[0]} "
            f"| intermediate_class={predictor.intermediate_class}"
        )
