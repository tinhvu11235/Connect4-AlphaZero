"""Chapter 8's root-child Monte Carlo Tree Search agent.

This intentionally mirrors ``utils/ch08util.py``.  It is not a persistent or
multi-level tree: statistics are kept only for legal moves at the root, and
every selected child is completed with a random simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from time import perf_counter
from typing import Any

from connect4_ai.agents.base import BaseAgent


@dataclass
class MCTSMetrics:
    """Instrumentation from the most recent ``select_action`` call."""

    rollouts: int = 0
    simulation_steps: int = 0
    action_time: float = 0.0
    rollout_time: float = 0.0


class MCTSAgent(BaseAgent):
    """Book-style root-only UCT search with random terminal rollouts."""

    def __init__(
        self,
        *,
        num_rollouts: int = 100,
        exploration_constant: float = 1.4,
        seed: int | None = None,
    ) -> None:
        if isinstance(num_rollouts, bool) or not isinstance(num_rollouts, int):
            raise TypeError("num_rollouts must be an integer")
        if num_rollouts < 0:
            raise ValueError("num_rollouts must be non-negative")
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")

        super().__init__(seed=seed)
        self.num_rollouts = num_rollouts
        self.exploration_constant = float(exploration_constant)
        self.metrics = MCTSMetrics()
        self.last_counts: dict[int, int] = {}
        self.last_wins: dict[int, int] = {}
        self.last_losses: dict[int, int] = {}
        self.last_scores: dict[int, float] = {}

    @property
    def rollouts(self) -> int:
        return self.metrics.rollouts

    @property
    def simulation_steps(self) -> int:
        return self.metrics.simulation_steps

    @property
    def action_time(self) -> float:
        return self.metrics.action_time

    @property
    def rollout_time(self) -> float:
        return self.metrics.rollout_time

    def select_action(self, env: Any) -> int:
        started = perf_counter()
        legal_actions = self._legal_actions(env)
        self.metrics = MCTSMetrics()
        self.last_counts = {move: 0 for move in legal_actions}
        self.last_wins = {move: 0 for move in legal_actions}
        self.last_losses = {move: 0 for move in legal_actions}
        self.last_scores = {move: 0.0 for move in legal_actions}

        # Preserve the book's direct return when only one move remains.
        if len(legal_actions) == 1:
            action = legal_actions[0]
            elapsed = perf_counter() - started
            self.metrics.action_time = elapsed
            self.last_action_time = elapsed
            return self._validate_result(action, legal_actions)

        rollout_started = perf_counter()
        for _ in range(self.num_rollouts):
            move = self._select(
                legal_actions,
                self.last_counts,
                self.last_wins,
                self.last_losses,
            )
            child, done, reward = self._expand(env, move)
            reward, simulation_steps = self._simulate(child, done, reward)
            self._backpropagate(
                env,
                move,
                reward,
                self.last_counts,
                self.last_wins,
                self.last_losses,
            )
            self.metrics.rollouts += 1
            self.metrics.simulation_steps += simulation_steps
        self.metrics.rollout_time = perf_counter() - rollout_started

        action = self._next_move(
            self.last_counts, self.last_wins, self.last_losses
        )
        self.last_scores = self._result_scores(
            self.last_counts, self.last_wins, self.last_losses
        )
        elapsed = perf_counter() - started
        self.metrics.action_time = elapsed
        self.last_action_time = elapsed
        return self._validate_result(action, legal_actions)

    def _select(
        self,
        legal_actions: tuple[int, ...],
        counts: dict[int, int],
        wins: dict[int, int],
        losses: dict[int, int],
    ) -> int:
        """Select a root move using the Chapter 8 UCT expression."""

        for move in legal_actions:
            if counts[move] == 0:
                return move

        total_simulations = sum(counts.values())
        scores: dict[int, float] = {}
        for move in legal_actions:
            visits = counts[move]
            value = (wins.get(move, 0) - losses.get(move, 0)) / visits
            exploration = self.exploration_constant * sqrt(
                log(total_simulations) / visits
            )
            scores[move] = value + exploration
        return max(scores, key=scores.get)

    def _expand(self, env: Any, move: int) -> tuple[Any, bool, int]:
        child = self._clone_environment(env)
        _, reward, done, _ = child.step(move)
        return child, done, reward

    def _simulate(
        self, env_copy: Any, done: bool, reward: int
    ) -> tuple[int, int]:
        """Randomly complete one child, returning reward and random-move count."""

        simulation_steps = 0
        while not done:
            move = self._rng.choice(env_copy.validinputs)
            _, reward, done, _ = env_copy.step(move)
            simulation_steps += 1
        return reward, simulation_steps

    @staticmethod
    def _backpropagate(
        env: Any,
        move: int,
        reward: int,
        counts: dict[int, int],
        wins: dict[int, int],
        losses: dict[int, int],
    ) -> None:
        """Update root-child statistics from the root player's perspective."""

        counts[move] = counts.get(move, 0) + 1
        root_is_red = env.turn in (1, "X", "red")
        root_is_yellow = env.turn in (2, "O", "yellow")
        if (reward == 1 and root_is_red) or (reward == -1 and root_is_yellow):
            wins[move] = wins.get(move, 0) + 1
        if (reward == -1 and root_is_red) or (reward == 1 and root_is_yellow):
            losses[move] = losses.get(move, 0) + 1

    @staticmethod
    def _result_scores(
        counts: dict[int, int],
        wins: dict[int, int],
        losses: dict[int, int],
    ) -> dict[int, float]:
        return {
            move: 0.0
            if visits == 0
            else (wins.get(move, 0) - losses.get(move, 0)) / visits
            for move, visits in counts.items()
        }

    @classmethod
    def _next_move(
        cls,
        counts: dict[int, int],
        wins: dict[int, int],
        losses: dict[int, int],
    ) -> int:
        scores = cls._result_scores(counts, wins, losses)
        return max(scores, key=scores.get)


__all__ = ["MCTSAgent", "MCTSMetrics"]
