from __future__ import annotations

import unittest

from connect4_ai.agents import DepthMinimaxAgent
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


class DepthMinimaxAgentTests(unittest.TestCase):
    def test_returns_legal_actions_on_fixed_states(self) -> None:
        agent = DepthMinimaxAgent(depth=1, seed=11)
        for actions in REGRESSION_ACTION_SEQUENCES:
            env = make_env(actions)
            self.assertIn(agent.select_action(env), env.validinputs)

    def test_immediate_winning_move(self) -> None:
        env = make_env(TACTICAL_POSITIONS["immediate_vertical_win"])
        self.assertEqual(1, DepthMinimaxAgent(depth=2, seed=1).select_action(env))

    def test_forced_blocking_move(self) -> None:
        env = make_env(TACTICAL_POSITIONS["forced_vertical_block"])
        self.assertEqual(4, DepthMinimaxAgent(depth=1, seed=1).select_action(env))

    def test_full_column_avoidance(self) -> None:
        env = make_env(TACTICAL_POSITIONS["full_first_column"])
        action = DepthMinimaxAgent(depth=1, seed=1).select_action(env)
        self.assertIn(action, env.validinputs)
        self.assertNotEqual(1, action)

    def test_seed_controls_tie_breaking(self) -> None:
        env = make_env(TACTICAL_POSITIONS["empty"])
        first = DepthMinimaxAgent(depth=0, seed=42).select_action(env)
        second = DepthMinimaxAgent(depth=0, seed=42).select_action(env)
        self.assertEqual(first, second)

    def test_negative_depth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DepthMinimaxAgent(depth=-1)

if __name__ == "__main__":
    unittest.main()
