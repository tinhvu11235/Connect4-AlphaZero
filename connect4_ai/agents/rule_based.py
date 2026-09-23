"""Chapter 3 three-step rule-based Connect Four agent."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from connect4_ai.agents.base import BaseAgent
from connect4_ai.env import GameAlreadyFinishedError, InvalidActionError


class RuleBasedThink3Agent(BaseAgent):
    """Readable transcription of ``utils.ch03util.conn_think3``.

    Decision order and source quirks are intentional. In particular, an empty
    center is selected before tactical checks, and the original avoid test
    checks specifically for reward ``-1``.
    """

    def select_action(self, env: Any) -> int:
        started = perf_counter()
        legal_actions = self._legal_actions(env)

        def finish(action: int) -> int:
            self.last_action_time = perf_counter() - started
            return self._validate_result(action, legal_actions)

        if len(legal_actions) == 1:
            return finish(legal_actions[0])

        # Preserve the source's first strategic rule exactly.
        if 4 in legal_actions and len(env.occupied[3]) == 0:
            return finish(4)

        immediate_win = self._immediate_winning_move(env, legal_actions)
        if immediate_win is not None:
            return finish(immediate_win)

        blocking_move = self._blocking_move(env, legal_actions)
        if blocking_move is not None:
            return finish(blocking_move)

        to_avoid = self._moves_to_avoid(env, legal_actions)
        if to_avoid:
            leftovers = [move for move in legal_actions if move not in to_avoid]
            if leftovers:
                return finish(self._rng.choice(leftovers))

        three_step_winners = self._three_step_winners(env, legal_actions)
        if three_step_winners:
            # ``max(..., key=list.count)`` retains first-occurrence tie behavior.
            return finish(max(three_step_winners, key=three_step_winners.count))

        return finish(self._rng.choice(legal_actions))

    def _immediate_winning_move(
        self, env: Any, legal_actions: tuple[int, ...]
    ) -> int | None:
        for move in legal_actions:
            env_copy = self._clone_environment(env)
            _, reward, done, _ = env_copy.step(move)
            if done and reward != 0:
                return move
        return None

    def _blocking_move(
        self, env: Any, legal_actions: tuple[int, ...]
    ) -> int | None:
        for first_move in legal_actions:
            for second_move in legal_actions:
                if first_move == second_move:
                    continue
                env_copy = self._clone_environment(env)
                _, _, done, _ = env_copy.step(first_move)
                if done:
                    continue
                _, reward, done, _ = env_copy.step(second_move)
                if done and reward != 0:
                    return second_move
        return None

    def _moves_to_avoid(
        self, env: Any, legal_actions: tuple[int, ...]
    ) -> list[int]:
        to_avoid: list[int] = []
        for move in legal_actions:
            if len(env.occupied[move - 1]) > 4:
                continue
            env_copy = self._clone_environment(env)
            _, _, done, _ = env_copy.step(move)
            if done:
                continue
            _, reward, done, _ = env_copy.step(move)
            # Deliberately preserve the source's red/yellow-asymmetric check.
            if done and reward == -1:
                to_avoid.append(move)
        return to_avoid

    def _three_step_winners(
        self, env: Any, legal_actions: tuple[int, ...]
    ) -> list[int]:
        winners: list[int] = []
        for first_move in legal_actions:
            for second_move in legal_actions:
                for third_move in legal_actions:
                    try:
                        env_copy = self._clone_environment(env)
                        env_copy.step(first_move)
                        env_copy.step(second_move)
                        _, reward, done, _ = env_copy.step(third_move)
                    except (InvalidActionError, GameAlreadyFinishedError, IndexError):
                        # The source used a broad exception around hypothetical
                        # paths. Restrict it to invalid-path failures.
                        continue
                    if done and reward != 0:
                        winners.append(first_move)
        return winners
