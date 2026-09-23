"""PyTorch equivalent of the Chapter 15 model used by Chapter 7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


BOARD_SHAPE = (7, 6)
CLASS_NAMES = ("draw", "win", "loss")


def board_to_tensor(
    board: np.ndarray | Tensor,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Convert column-major boards to explicit PyTorch NCHW tensors.

    Accepted shapes are ``(7, 6)``, ``(N, 7, 6)``, ``(N, 1, 7, 6)``, and the
    source Keras layout ``(N, 7, 6, 1)``. Column and row axes are never swapped.
    """

    tensor = torch.as_tensor(board, dtype=torch.float32)
    if tensor.ndim == 2 and tuple(tensor.shape) == BOARD_SHAPE:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3 and tuple(tensor.shape[1:]) == BOARD_SHAPE:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4 and tuple(tensor.shape[1:]) == (1, *BOARD_SHAPE):
        pass
    elif tensor.ndim == 4 and tuple(tensor.shape[1:]) == (*BOARD_SHAPE, 1):
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "expected board shape (7, 6), (N, 7, 6), (N, 1, 7, 6), "
            f"or (N, 7, 6, 1); got {tuple(tensor.shape)}"
        )
    if device is not None:
        tensor = tensor.to(device)
    return tensor.contiguous()


class PositionEvalNet(nn.Module):
    """350,659-parameter value classifier matching ``value_conn.h5``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 128, kernel_size=(4, 4), padding=0)
        self.fc1 = nn.Linear(7 * 6 * 128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 3)

    def forward_logits(self, boards_nchw: Tensor) -> Tensor:
        if boards_nchw.ndim != 4 or tuple(boards_nchw.shape[1:]) != (1, 7, 6):
            raise ValueError(
                "PositionEvalNet expects NCHW shape (N, 1, 7, 6); "
                f"got {tuple(boards_nchw.shape)}"
            )
        # TensorFlow SAME for an even 4x4 kernel pads one cell before and two
        # after each spatial axis. Make this asymmetric padding explicit.
        padded = F.pad(boards_nchw, (1, 2, 1, 2))
        features = F.relu(self.conv(padded))
        # Keras Flatten consumed NHWC data. Restore that exact feature order
        # before applying directly converted Dense weights.
        features = features.permute(0, 2, 3, 1).contiguous().flatten(1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.output(features)

    def forward(self, boards_nchw: Tensor) -> Tensor:
        return torch.softmax(self.forward_logits(boards_nchw), dim=1)


def load_keras_value_weights(
    model: PositionEvalNet, source_h5: str | Path
) -> PositionEvalNet:
    """Load Chapter 15 Keras weights without importing TensorFlow/Keras."""

    import h5py

    source_h5 = Path(source_h5)
    with h5py.File(source_h5, "r") as handle:
        root = handle["model_weights"]

        def array(path: str) -> np.ndarray:
            return np.asarray(root[path], dtype=np.float32)

        conv_kernel = array("conv2d/conv2d/kernel:0").transpose(3, 2, 0, 1)
        conv_bias = array("conv2d/conv2d/bias:0")
        dense_kernel = array("dense/dense/kernel:0").T
        dense_bias = array("dense/dense/bias:0")
        dense1_kernel = array("dense_1/dense_1/kernel:0").T
        dense1_bias = array("dense_1/dense_1/bias:0")
        output_kernel = array("dense_2/dense_2/kernel:0").T
        output_bias = array("dense_2/dense_2/bias:0")

    with torch.no_grad():
        model.conv.weight.copy_(torch.from_numpy(conv_kernel.copy()))
        model.conv.bias.copy_(torch.from_numpy(conv_bias.copy()))
        model.fc1.weight.copy_(torch.from_numpy(dense_kernel.copy()))
        model.fc1.bias.copy_(torch.from_numpy(dense_bias.copy()))
        model.fc2.weight.copy_(torch.from_numpy(dense1_kernel.copy()))
        model.fc2.bias.copy_(torch.from_numpy(dense1_bias.copy()))
        model.output.weight.copy_(torch.from_numpy(output_kernel.copy()))
        model.output.bias.copy_(torch.from_numpy(output_bias.copy()))
    return model


def load_position_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[PositionEvalNet, dict[str, Any]]:
    """Load a migrated/trained position model and its metadata."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = PositionEvalNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
