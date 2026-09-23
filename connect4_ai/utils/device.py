"""Central PyTorch device policy."""

from __future__ import annotations

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CudaRequiredError(RuntimeError):
    """Raised when a GPU-required entry point cannot access CUDA."""


def resolve_device(
    *,
    allow_cpu: bool = False,
    requested: torch.device | str | None = None,
) -> torch.device:
    """Resolve and validate a requested device.

    ``None``, ``"auto"``, and the unindexed ``"cuda"`` all select the first
    CUDA device.  An explicit index is useful on hosted dual-GPU runtimes such
    as Kaggle, where the learner and reference model can use different GPUs.
    """

    if requested is None or str(requested) == "auto":
        selected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(requested)
        if selected.type == "cuda" and selected.index is None:
            selected = torch.device("cuda:0")

    if selected.type == "cuda" and torch.cuda.is_available():
        index = 0 if selected.index is None else selected.index
        if index < torch.cuda.device_count():
            return torch.device(f"cuda:{index}")
        raise CudaRequiredError(
            f"requested CUDA device {selected} is unavailable; "
            f"only {torch.cuda.device_count()} CUDA device(s) were detected"
        )
    if selected.type == "cpu" and allow_cpu:
        return selected
    if selected.type != "cuda":
        raise CudaRequiredError(
            f"CUDA is required but device {selected!s} was requested. "
            "Use --allow-cpu only for an explicitly requested CPU run."
        )
    if allow_cpu:
        return torch.device("cpu")
    raise CudaRequiredError(
        "CUDA is required but torch.cuda.is_available() is False. "
        "Use --allow-cpu only for an explicitly requested CPU run."
    )


def resolve_opponent_device(
    primary: torch.device | str,
    requested: torch.device | str | None = "auto",
) -> torch.device:
    """Choose GPU 1 automatically when it exists, otherwise use ``primary``."""

    primary_device = resolve_device(requested=primary)
    if requested is None or str(requested) == "auto":
        if torch.cuda.device_count() > 1:
            secondary_index = 1 if primary_device.index != 1 else 0
            return torch.device(f"cuda:{secondary_index}")
        return primary_device
    return resolve_device(requested=requested)
