"""Common agent API and instrumentation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any


@dataclass
class SearchMetrics:
    """Metrics from the most recent call to ``select_action``."""

    nodes_visited: int = 0
    nodes_pruned: int = 0
    search_time: float = 0.0


class BaseAgent(ABC):
    """Uniform interface shared by all refactored agents."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.last_action_time = 0.0

    def set_seed(self, seed: int | None) -> None:
        self._rng.seed(seed)

    @abstractmethod
    def select_action(self, env: Any) -> int:
        """Select and return one action from ``env.validinputs``."""

    @staticmethod
    def _legal_actions(env: Any) -> tuple[int, ...]:
        legal_actions = tuple(env.validinputs)
        if not legal_actions:
            raise ValueError("agent cannot act because no legal moves remain")
        return legal_actions

    @staticmethod
    def _clone_environment(env: Any) -> Any:
        copy_method = getattr(env, "copy", None)
        return copy_method() if callable(copy_method) else deepcopy(env)

    @staticmethod
    def _validate_result(action: int, legal_actions: tuple[int, ...]) -> int:
        if action not in legal_actions:
            raise RuntimeError(
                f"agent selected illegal action {action}; legal actions are "
                f"{list(legal_actions)}"
            )
        return action
