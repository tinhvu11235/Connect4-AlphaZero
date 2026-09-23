"""Chapter 15 Connect Four value network used by AlphaGo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from connect4_ai.models.position_net import (
    CLASS_NAMES,
    PositionEvalNet,
    board_to_tensor,
    load_keras_value_weights,
)


class ValueNet(PositionEvalNet):
    """Named AlphaGo role for the exact Chapter 15 three-class value model."""


def load_value_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[ValueNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ValueNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


__all__ = [
    "CLASS_NAMES",
    "ValueNet",
    "board_to_tensor",
    "load_keras_value_weights",
    "load_value_checkpoint",
]
