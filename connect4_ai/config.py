"""Central runtime paths and default hyperparameters for the eight agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
PRETRAINED_CHECKPOINTS = CHECKPOINT_ROOT / "pretrained"
TRAINING_CHECKPOINTS = CHECKPOINT_ROOT / "training"


@dataclass(frozen=True)
class AgentCheckpointPaths:
    position_eval: Path = PRETRAINED_CHECKPOINTS / "position_eval.pt"
    fast_policy: Path = PRETRAINED_CHECKPOINTS / "policy_conn.pt"
    policy_gradient: Path = PRETRAINED_CHECKPOINTS / "policy_gradient_conn.pt"
    value: Path = PRETRAINED_CHECKPOINTS / "value_conn.pt"
    alphazero_book: Path = PRETRAINED_CHECKPOINTS / "connzero_book_iter_004.pt"
    alphazero_training_state: Path = (
        TRAINING_CHECKPOINTS / "connzero_training_state.pt"
    )
    alphazero_final: Path = TRAINING_CHECKPOINTS / "connzero_final_trained.pt"


AGENT_CHECKPOINTS = AgentCheckpointPaths()


__all__ = [
    "AGENT_CHECKPOINTS",
    "AgentCheckpointPaths",
    "CHECKPOINT_ROOT",
    "PRETRAINED_CHECKPOINTS",
    "PROJECT_ROOT",
    "TRAINING_CHECKPOINTS",
]
