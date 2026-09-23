"""Uniform random legal-move agent from Chapter 3."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from connect4_ai.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def select_action(self, env: Any) -> int:
        started = perf_counter()
        legal_actions = self._legal_actions(env)
        action = self._rng.choice(legal_actions)
        self.last_action_time = perf_counter() - started
        return self._validate_result(action, legal_actions)
