"""Faithful PyTorch refactor of the book's simplified Connect Four AlphaGo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from connect4_ai.agents.base import BaseAgent
from connect4_ai.models.policy_net import (
    FastPolicyNet,
    PolicyGradientNet,
    board_from_player_perspective,
    load_policy_checkpoint,
    policy_board_to_tensor,
)
from connect4_ai.models.value_net import ValueNet, load_value_checkpoint
from connect4_ai.utils.device import CudaRequiredError, resolve_device


@dataclass
class AlphaGoMetrics:
    """Instrumentation from the latest root-only AlphaGo search."""

    rollouts: int = 0
    simulation_steps: int = 0
    policy_inference_calls: int = 0
    value_inference_calls: int = 0
    action_time: float = 0.0


class AlphaGoAgent(BaseAgent):
    """Chapter 17/18 root-child search enhanced by three neural networks."""

    def __init__(
        self,
        policy_model: PolicyGradientNet,
        value_model: ValueNet,
        rollout_policy_model: FastPolicyNet,
        *,
        weight: float = 0.75,
        depth: int = 45,
        policy_rollout: bool = False,
        num_rollouts: int = 584,
        seed: int | None = None,
        device: torch.device | str | None = None,
        allow_cpu: bool = False,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError("depth must be an integer")
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if isinstance(num_rollouts, bool) or not isinstance(num_rollouts, int):
            raise TypeError("num_rollouts must be an integer")
        if num_rollouts < 0:
            raise ValueError("num_rollouts must be non-negative")

        selected_device = (
            resolve_device(allow_cpu=allow_cpu)
            if device is None
            else torch.device(device)
        )
        if selected_device.type != "cuda" and not allow_cpu:
            raise CudaRequiredError("AlphaGoAgent requires CUDA unless allow_cpu=True")

        self.device = selected_device
        self.policy_model = policy_model.to(self.device).eval()
        self.value_model = value_model.to(self.device).eval()
        self.rollout_policy_model = rollout_policy_model.to(self.device).eval()
        self.weight = float(weight)
        self.depth = depth
        self.policy_rollout = policy_rollout
        self.num_rollouts = num_rollouts
        self.metrics = AlphaGoMetrics()
        self.last_priors: dict[int, float] = {}
        self.last_results: dict[int, list[float]] = {}
        self.last_visits: dict[int, int] = {}

    @classmethod
    def from_checkpoints(
        cls,
        *,
        policy_checkpoint: str | Path,
        value_checkpoint: str | Path,
        rollout_policy_checkpoint: str | Path,
        device: torch.device | str | None = None,
        allow_cpu: bool = False,
        **agent_config: Any,
    ) -> AlphaGoAgent:
        selected_device = (
            resolve_device(allow_cpu=allow_cpu)
            if device is None
            else torch.device(device)
        )
        policy, _ = load_policy_checkpoint(
            policy_checkpoint,
            model_type="policy_gradient",
            device=selected_device,
        )
        rollout_policy, _ = load_policy_checkpoint(
            rollout_policy_checkpoint,
            model_type="fast",
            device=selected_device,
        )
        value, _ = load_value_checkpoint(value_checkpoint, device=selected_device)
        return cls(
            policy,  # type: ignore[arg-type]
            value,
            rollout_policy,  # type: ignore[arg-type]
            device=selected_device,
            allow_cpu=allow_cpu,
            **agent_config,
        )

    @property
    def rollouts(self) -> int:
        return self.metrics.rollouts

    @property
    def simulation_steps(self) -> int:
        return self.metrics.simulation_steps

    @property
    def action_time(self) -> float:
        return self.metrics.action_time

    def policy_probabilities(self, env: Any) -> np.ndarray:
        board = board_from_player_perspective(env.state, env.turn)
        tensor = policy_board_to_tensor(board, device=self.device)
        with torch.inference_mode():
            probabilities = self.policy_model(tensor)
        self.metrics.policy_inference_calls += 1
        return probabilities[0].detach().cpu().numpy()

    def value_probabilities(self, env: Any) -> np.ndarray:
        board = board_from_player_perspective(env.state, env.turn)
        tensor = policy_board_to_tensor(board, device=self.device)
        with torch.inference_mode():
            probabilities = self.value_model(tensor)
        self.metrics.value_inference_calls += 1
        return probabilities[0].detach().cpu().numpy()

    def cutoff_reward(self, env: Any) -> float:
        """Return the exact source reward convention: global red perspective."""

        probabilities = self.value_probabilities(env)
        current_player_value = float(probabilities[1] - probabilities[2])
        return current_player_value if env.turn == "red" else -current_player_value

    def select_action(self, env: Any) -> int:
        started = perf_counter()
        legal_actions = self._legal_actions(env)
        self.metrics = AlphaGoMetrics()
        self.last_priors = {}
        self.last_results = {move: [] for move in legal_actions}
        self.last_visits = {move: 0 for move in legal_actions}

        if len(legal_actions) == 1:
            return self._finish(legal_actions[0], legal_actions, started)

        probabilities = self.policy_probabilities(env)
        self.last_priors = {
            move: float(probabilities[move - 1]) for move in legal_actions
        }

        for _ in range(self.num_rollouts):
            move = self._select(
                probabilities, legal_actions, self.last_results, self.weight
            )
            child, done, reward = self._expand(env, move)
            reward, steps = self._simulate(child, done, reward)
            self._backpropagate(env, move, reward, self.last_results)
            self.metrics.rollouts += 1
            self.metrics.simulation_steps += steps

        self.last_visits = {
            move: len(results) for move, results in self.last_results.items()
        }
        action = max(self.last_visits, key=self.last_visits.get)
        return self._finish(action, legal_actions, started)

    def _finish(
        self, action: int, legal_actions: tuple[int, ...], started: float
    ) -> int:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - started
        self.metrics.action_time = elapsed
        self.last_action_time = elapsed
        return self._validate_result(action, legal_actions)

    @staticmethod
    def _select(
        priors: np.ndarray,
        legal_actions: tuple[int, ...],
        results: dict[int, list[float]],
        weight: float,
    ) -> int:
        scores: dict[int, float] = {}
        for move in legal_actions:
            outcomes = results[move]
            rollout_value = 0.0 if not outcomes else sum(outcomes) / len(outcomes)
            scaled_prior = float(priors[move - 1]) / (1 + len(outcomes))
            scores[move] = (
                weight * scaled_prior + (1.0 - weight) * rollout_value
            )
        return max(scores, key=scores.get)

    def _expand(self, env: Any, move: int) -> tuple[Any, bool, int]:
        child = self._clone_environment(env)
        _, reward, done, _ = child.step(move)
        return child, done, reward

    def _rollout_policy_action(self, env: Any) -> int:
        legal_actions = tuple(sorted(env.validinputs))
        board = board_from_player_perspective(env.state, env.turn)
        tensor = policy_board_to_tensor(board, device=self.device)
        with torch.inference_mode():
            probabilities = self.rollout_policy_model(tensor)[0]
        self.metrics.policy_inference_calls += 1
        legal_weights = [float(probabilities[move - 1].item()) for move in legal_actions]
        total = sum(legal_weights)
        if not np.isfinite(total) or total <= 0.0:
            return self._rng.choice(legal_actions)
        return self._rng.choices(legal_actions, weights=legal_weights, k=1)[0]

    def _simulate(
        self, env_copy: Any, done: bool, reward: int
    ) -> tuple[float, int]:
        if done:
            return float(reward), 0
        steps = 0
        for _ in range(self.depth):
            if self.policy_rollout:
                # Preserve ch17util.py's fallback to a random legal move when
                # policy-guided rollout selection cannot produce an action.
                try:
                    move = self._rollout_policy_action(env_copy)
                except Exception:
                    move = self._rng.choice(env_copy.validinputs)
            else:
                move = self._rng.choice(env_copy.validinputs)
            _, reward, done, _ = env_copy.step(move)
            steps += 1
            if done:
                return float(reward), steps
        return self.cutoff_reward(env_copy), steps

    @staticmethod
    def _backpropagate(
        env: Any,
        move: int,
        reward: float,
        results: dict[int, list[float]],
    ) -> None:
        if env.turn == "red":
            results[move].append(float(reward))
        elif env.turn == "yellow":
            results[move].append(float(-reward))
        else:
            raise ValueError(f"unsupported Connect Four turn: {env.turn!r}")


__all__ = ["AlphaGoAgent", "AlphaGoMetrics"]
