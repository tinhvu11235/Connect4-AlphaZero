from __future__ import annotations

import unittest

from connect4_ai.agents import RuleBasedThink3Agent
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


class RuleBasedThink3AgentTests(unittest.TestCase):
    def test_returns_legal_actions_on_fixed_states(self) -> None:
        for index, actions in enumerate(REGRESSION_ACTION_SEQUENCES):
            env = make_env(actions)
            action = RuleBasedThink3Agent(seed=index).select_action(env)
            self.assertIn(action, env.validinputs)

    def test_empty_center_keeps_original_priority(self) -> None:
        env = make_env(TACTICAL_POSITIONS["empty"])
        self.assertEqual(4, RuleBasedThink3Agent(seed=1).select_action(env))

    def test_immediate_winning_move(self) -> None:
        env = make_env(TACTICAL_POSITIONS["immediate_vertical_win"])
        self.assertEqual(1, RuleBasedThink3Agent(seed=1).select_action(env))

    def test_forced_blocking_move(self) -> None:
        env = make_env(TACTICAL_POSITIONS["forced_vertical_block"])
        self.assertEqual(4, RuleBasedThink3Agent(seed=1).select_action(env))

    def test_full_column_avoidance(self) -> None:
        env = make_env(TACTICAL_POSITIONS["full_first_column"])
        action = RuleBasedThink3Agent(seed=1).select_action(env)
        self.assertIn(action, env.validinputs)
        self.assertNotEqual(1, action)

    def test_fallback_randomness_is_seeded(self) -> None:
        env = make_env(TACTICAL_POSITIONS["fallback_after_center_opening"])
        first = RuleBasedThink3Agent(seed=77).select_action(env)
        second = RuleBasedThink3Agent(seed=77).select_action(env)
        self.assertEqual(first, second)

if __name__ == "__main__":
    unittest.main()
