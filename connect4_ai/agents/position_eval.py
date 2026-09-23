"""Chapter 7 alpha-beta search with learned cutoff evaluation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from connect4_ai.agents.base import BaseAgent, SearchMetrics
from connect4_ai.models.position_net import (
    PositionEvalNet,
    board_to_tensor,
    load_position_checkpoint,
)
from connect4_ai.utils.device import CudaRequiredError, resolve_device


class PositionEvalAgent(BaseAgent):
    """Faithful PyTorch transcription of ``MiniMax_conn_eval``."""

    def __init__(
        self,
        model: PositionEvalNet,
        *,
        depth: int = 3,
        device: torch.device | str | None = None,
        allow_cpu: bool = False,
    ) -> None:
        super().__init__(seed=None)
        if depth < 0:
            raise ValueError("depth must be non-negative")
        selected_device = (
            resolve_device(allow_cpu=allow_cpu)
            if device is None
            else torch.device(device)
        )
        if selected_device.type != "cuda" and not allow_cpu:
            raise CudaRequiredError(
                "PositionEvalAgent requires CUDA unless allow_cpu=True"
            )
        self.device = selected_device
        self.model = model.to(self.device).eval()
        self.depth = depth
        self.metrics = SearchMetrics()
        self.last_action_values: dict[int, float] = {}
        self.inference_calls = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        depth: int = 3,
        device: torch.device | str | None = None,
        allow_cpu: bool = False,
    ) -> PositionEvalAgent:
        selected_device = (
            resolve_device(allow_cpu=allow_cpu)
            if device is None
            else torch.device(device)
        )
        model, _ = load_position_checkpoint(
            checkpoint_path, device=selected_device
        )
        return cls(
            model,
            depth=depth,
            device=selected_device,
            allow_cpu=allow_cpu,
        )

    @property
    def nodes_visited(self) -> int:
        return self.metrics.nodes_visited

    @property
    def nodes_pruned(self) -> int:
        return self.metrics.nodes_pruned

    @property
    def search_time(self) -> float:
        return self.metrics.search_time

    def position_evaluation(self, env: Any) -> float:
        """Return raw Chapter 7 value: P(win) minus P(loss)."""

        state = board_to_tensor(env.state, device=self.device)
        with torch.inference_mode():
            probabilities = self.model(state)
        self.inference_calls += 1
        value = probabilities[0, 1] - probabilities[0, 2]
        return float(value.item())

    def cutoff_evaluation(self, env: Any) -> float:
        """Apply the original red/yellow sign logic at a search cutoff."""

        value = self.position_evaluation(env)
        return value if env.turn == "red" else -value

    def select_action(self, env: Any) -> int:
        started = perf_counter()
        self.metrics = SearchMetrics()
        self.last_action_values = {}
        self.inference_calls = 0
        legal_actions = self._legal_actions(env)

        for move in legal_actions:
            env_copy = self._clone_environment(env)
            _, reward, done, _ = env_copy.step(move)
            self.metrics.nodes_visited += 1
            if done and reward != 0:
                self.last_action_values[move] = 1.0
                return self._finish(move, legal_actions, started)
            opponent_payoff = self._eval_payoff_conn(
                env_copy,
                reward,
                done,
                self.depth,
                alpha=-2.0,
                beta=-2.0,
            )
            self.last_action_values[move] = -opponent_payoff

        # Python dict insertion order preserves the source's first-move tie.
        action = max(self.last_action_values, key=self.last_action_values.get)
        return self._finish(action, legal_actions, started)

    def _finish(
        self, action: int, legal_actions: tuple[int, ...], started: float
    ) -> int:
        elapsed = perf_counter() - started
        self.metrics.search_time = elapsed
        self.last_action_time = elapsed
        return self._validate_result(action, legal_actions)

    def _eval_payoff_conn(
        self,
        env: Any,
        reward: int,
        done: bool,
        depth: int,
        alpha: float | None,
        beta: float | None,
    ) -> float:
        if done:
            return -1.0 if reward != 0 else 0.0
        if depth == 0:
            return self.cutoff_evaluation(env)
        if alpha is None:
            alpha = -2.0
        if beta is None:
            beta = -2.0

        best_payoff = alpha if env.turn == "red" else beta
        legal_actions = tuple(env.validinputs)
        for index, move in enumerate(legal_actions):
            env_copy = self._clone_environment(env)
            _, child_reward, child_done, _ = env_copy.step(move)
            self.metrics.nodes_visited += 1
            opponent_payoff = self._eval_payoff_conn(
                env_copy,
                child_reward,
                child_done,
                depth - 1,
                alpha,
                beta,
            )
            my_payoff = -opponent_payoff
            if my_payoff > best_payoff:
                best_payoff = my_payoff
                if env.turn == "red":
                    alpha = best_payoff
                else:
                    beta = best_payoff
            if alpha >= -beta:
                self.metrics.nodes_pruned += len(legal_actions) - index - 1
                break
        return best_payoff
