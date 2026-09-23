"""Depth-limited Connect Four MiniMax from Chapter 5."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from connect4_ai.agents.base import BaseAgent, SearchMetrics


class DepthMinimaxAgent(BaseAgent):
    """Preserve the book's depth-limited negamax-style payoff search."""

    def __init__(self, *, depth: int = 3, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.depth = depth
        self.metrics = SearchMetrics()
        self.last_action_values: dict[int, int] = {}

    @property
    def nodes_visited(self) -> int:
        return self.metrics.nodes_visited

    @property
    def nodes_pruned(self) -> int:
        return self.metrics.nodes_pruned

    @property
    def search_time(self) -> float:
        return self.metrics.search_time

    def select_action(self, env: Any) -> int:
        started = perf_counter()
        self.metrics = SearchMetrics()
        self.last_action_values = {}
        legal_actions = self._legal_actions(env)
        wins: list[int] = []
        ties: list[int] = []

        for move in legal_actions:
            env_copy = self._clone_environment(env)
            _, reward, done, _ = env_copy.step(move)
            self.metrics.nodes_visited += 1
            if done and reward != 0:
                self.last_action_values[move] = 1
                return self._finish(move, legal_actions, started)

            opponent_payoff = self._maximized_payoff(
                env_copy, reward, done, self.depth
            )
            my_payoff = -opponent_payoff
            self.last_action_values[move] = my_payoff
            if my_payoff == 1:
                wins.append(move)
            elif my_payoff == 0:
                ties.append(move)

        if wins:
            action = self._rng.choice(wins)
        elif ties:
            action = self._rng.choice(ties)
        else:
            # All legal actions are losses; this matches ``env.sample()``.
            action = self._rng.choice(legal_actions)
        return self._finish(action, legal_actions, started)

    def _finish(
        self, action: int, legal_actions: tuple[int, ...], started: float
    ) -> int:
        elapsed = perf_counter() - started
        self.metrics.search_time = elapsed
        self.last_action_time = elapsed
        return self._validate_result(action, legal_actions)

    def _maximized_payoff(
        self, env: Any, reward: int, done: bool, depth: int
    ) -> int:
        if done:
            return -1 if reward != 0 else 0
        if depth == 0:
            return 0

        best_payoff = -2
        for move in env.validinputs:
            env_copy = self._clone_environment(env)
            _, child_reward, child_done, _ = env_copy.step(move)
            self.metrics.nodes_visited += 1
            opponent_payoff = self._maximized_payoff(
                env_copy, child_reward, child_done, depth - 1
            )
            my_payoff = -opponent_payoff
            if my_payoff > best_payoff:
                best_payoff = my_payoff
        return best_payoff
