from __future__ import annotations

import unittest

from connect4_ai.agents import RandomAgent
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


class RandomAgentTests(unittest.TestCase):
    def test_returns_only_legal_actions_on_fixed_states(self) -> None:
        agent = RandomAgent(seed=101)
        for actions in REGRESSION_ACTION_SEQUENCES:
            env = make_env(actions)
            for _ in range(20):
                self.assertIn(agent.select_action(env), env.validinputs)

    def test_seed_controls_action_sequence(self) -> None:
        env = make_env(TACTICAL_POSITIONS["fallback_after_center_opening"])
        first = RandomAgent(seed=2026)
        second = RandomAgent(seed=2026)
        first_actions = [first.select_action(env) for _ in range(50)]
        second_actions = [second.select_action(env) for _ in range(50)]
        self.assertEqual(first_actions, second_actions)

    def test_full_column_is_never_selected(self) -> None:
        env = make_env(TACTICAL_POSITIONS["full_first_column"])
        agent = RandomAgent(seed=7)
        selected = [agent.select_action(env) for _ in range(200)]
        self.assertNotIn(1, selected)
        self.assertTrue(set(selected).issubset(set(env.validinputs)))


if __name__ == "__main__":
    unittest.main()
