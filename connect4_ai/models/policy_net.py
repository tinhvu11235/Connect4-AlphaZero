"""PyTorch policy networks used by the book's Connect Four AlphaGo agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from connect4_ai.models.position_net import BOARD_SHAPE, board_to_tensor


def policy_board_to_tensor(
    board: np.ndarray | Tensor,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Explicitly convert column-major boards to NCHW policy input."""

    return board_to_tensor(board, device=device)


def board_from_player_perspective(
    board: np.ndarray | Tensor, turn: str
) -> np.ndarray | Tensor:
    """Represent the current player as +1, as in Chapters 11, 15, and 18."""

    if turn == "red":
        return board
    if turn == "yellow":
        return -board
    raise ValueError(f"unsupported Connect Four turn: {turn!r}")


class _PolicyConvBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 128, kernel_size=(4, 4), padding=0)

    def _conv_features(self, boards_nchw: Tensor) -> Tensor:
        if boards_nchw.ndim != 4 or tuple(boards_nchw.shape[1:]) != (
            1,
            *BOARD_SHAPE,
        ):
            raise ValueError(
                "policy network expects NCHW shape (N, 1, 7, 6); "
                f"got {tuple(boards_nchw.shape)}"
            )
        # TensorFlow SAME with an even 4x4 kernel pads 1 before and 2 after.
        features = F.relu(self.conv(F.pad(boards_nchw, (1, 2, 1, 2))))
        # Keras Flatten reads NHWC order.
        return features.permute(0, 2, 3, 1).contiguous().flatten(1)


class FastPolicyNet(_PolicyConvBase):
    """Chapter 11 policy classifier from ``policy_conn.h5``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(7 * 6 * 128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 7)

    def forward_logits(self, boards_nchw: Tensor) -> Tensor:
        features = self._conv_features(boards_nchw)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.output(features)

    def forward(self, boards_nchw: Tensor) -> Tensor:
        return torch.softmax(self.forward_logits(boards_nchw), dim=1)


class PolicyGradientNet(_PolicyConvBase):
    """Chapter 15 dual-input policy-gradient network from ``PG_conn.h5``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(7 * 6 * 128 + 42, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 7)

    def forward_logits(
        self,
        boards_nchw: Tensor,
        flat_boards: Tensor | None = None,
    ) -> Tensor:
        conv_features = self._conv_features(boards_nchw)
        if flat_boards is None:
            # NCHW's H/W axes remain source columns/rows, so this is the exact
            # C-order reshape used by state.reshape(-1, 42) in Chapter 15.
            flat_boards = boards_nchw[:, 0].contiguous().flatten(1)
        if flat_boards.ndim != 2 or flat_boards.shape[1] != 42:
            raise ValueError(
                "flat policy input must have shape (N, 42); "
                f"got {tuple(flat_boards.shape)}"
            )
        if flat_boards.shape[0] != boards_nchw.shape[0]:
            raise ValueError("flat and convolutional policy batch sizes differ")
        combined = torch.cat((conv_features, flat_boards), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        return self.output(combined)

    def forward(
        self,
        boards_nchw: Tensor,
        flat_boards: Tensor | None = None,
    ) -> Tensor:
        return torch.softmax(
            self.forward_logits(boards_nchw, flat_boards), dim=1
        )


def _load_common_keras_weights(
    model: FastPolicyNet | PolicyGradientNet,
    source_h5: str | Path,
) -> FastPolicyNet | PolicyGradientNet:
    import h5py

    with h5py.File(Path(source_h5), "r") as handle:
        root = handle["model_weights"]

        def array(path: str) -> np.ndarray:
            return np.asarray(root[path], dtype=np.float32)

        arrays = {
            "conv_weight": array("conv2d/conv2d/kernel:0").transpose(3, 2, 0, 1),
            "conv_bias": array("conv2d/conv2d/bias:0"),
            "fc1_weight": array("dense/dense/kernel:0").T,
            "fc1_bias": array("dense/dense/bias:0"),
            "fc2_weight": array("dense_1/dense_1/kernel:0").T,
            "fc2_bias": array("dense_1/dense_1/bias:0"),
            "output_weight": array("dense_2/dense_2/kernel:0").T,
            "output_bias": array("dense_2/dense_2/bias:0"),
        }

    with torch.no_grad():
        model.conv.weight.copy_(torch.from_numpy(arrays["conv_weight"].copy()))
        model.conv.bias.copy_(torch.from_numpy(arrays["conv_bias"].copy()))
        model.fc1.weight.copy_(torch.from_numpy(arrays["fc1_weight"].copy()))
        model.fc1.bias.copy_(torch.from_numpy(arrays["fc1_bias"].copy()))
        model.fc2.weight.copy_(torch.from_numpy(arrays["fc2_weight"].copy()))
        model.fc2.bias.copy_(torch.from_numpy(arrays["fc2_bias"].copy()))
        model.output.weight.copy_(
            torch.from_numpy(arrays["output_weight"].copy())
        )
        model.output.bias.copy_(torch.from_numpy(arrays["output_bias"].copy()))
    return model


def load_keras_fast_policy_weights(
    model: FastPolicyNet, source_h5: str | Path
) -> FastPolicyNet:
    return _load_common_keras_weights(model, source_h5)  # type: ignore[return-value]


def load_keras_policy_gradient_weights(
    model: PolicyGradientNet, source_h5: str | Path
) -> PolicyGradientNet:
    return _load_common_keras_weights(model, source_h5)  # type: ignore[return-value]


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_type: str,
    device: torch.device | str,
) -> tuple[FastPolicyNet | PolicyGradientNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if model_type == "fast":
        model: FastPolicyNet | PolicyGradientNet = FastPolicyNet()
    elif model_type == "policy_gradient":
        model = PolicyGradientNet()
    else:
        raise ValueError("model_type must be 'fast' or 'policy_gradient'")
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


__all__ = [
    "FastPolicyNet",
    "PolicyGradientNet",
    "board_from_player_perspective",
    "load_keras_fast_policy_weights",
    "load_keras_policy_gradient_weights",
    "load_policy_checkpoint",
    "policy_board_to_tensor",
]
