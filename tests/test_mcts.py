from __future__ import annotations

import unittest

import numpy as np

from connect4_ai.agents import MCTSAgent
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


ONE_ACTION_REMAINING = [
    4, 4, 6, 1, 5, 7, 4, 4, 7, 6, 2, 7, 5, 2, 1, 2, 6, 7,
    2, 2, 6, 6, 2, 6, 4, 1, 7, 4, 5, 5, 5, 1, 1, 1, 7, 5,
]


class MCTSAgentTests(unittest.TestCase):
    def test_returns_legal_actions_on_fixed_states(self) -> None:
        for index, actions in enumerate(REGRESSION_ACTION_SEQUENCES):
            env = make_env(actions)
            action = MCTSAgent(num_rollouts=20, seed=100 + index).select_action(env)
            self.assertIn(action, env.validinputs)

    def test_one_available_action_returns_without_rollouts(self) -> None:
        env = make_env(ONE_ACTION_REMAINING)
        self.assertEqual([3], env.validinputs)
        agent = MCTSAgent(num_rollouts=50, seed=2)
        self.assertEqual(3, agent.select_action(env))
        self.assertEqual(0, agent.rollouts)
        self.assertEqual(0, agent.simulation_steps)

    def test_terminal_child_needs_no_simulation_steps(self) -> None:
        env = make_env(TACTICAL_POSITIONS["immediate_vertical_win"])
        agent = MCTSAgent(num_rollouts=1, seed=3)
        self.assertEqual(1, agent.select_action(env))
        self.assertEqual(1, agent.last_counts[1])
        self.assertEqual(1, agent.last_wins[1])
        self.assertEqual(0, agent.simulation_steps)

    def test_uct_prioritizes_first_unvisited_legal_action(self) -> None:
        agent = MCTSAgent(seed=4)
        legal = (1, 2, 3)
        move = agent._select(
            legal,
            counts={1: 8, 2: 0, 3: 0},
            wins={1: 8, 2: 0, 3: 0},
            losses={1: 0, 2: 0, 3: 0},
        )
        self.assertEqual(2, move)

    def test_statistics_update_and_draw_handling(self) -> None:
        env = make_env([])
        counts = {1: 0}
        wins = {1: 0}
        losses = {1: 0}
        MCTSAgent._backpropagate(env, 1, 0, counts, wins, losses)
        self.assertEqual({1: 1}, counts)
        self.assertEqual({1: 0}, wins)
        self.assertEqual({1: 0}, losses)

    def test_reward_is_recorded_from_root_player_perspective(self) -> None:
        red_env = make_env([])
        yellow_env = make_env([4])
        for env, reward, expected in (
            (red_env, 1, (1, 0)),
            (red_env, -1, (0, 1)),
            (yellow_env, -1, (1, 0)),
            (yellow_env, 1, (0, 1)),
        ):
            counts, wins, losses = {1: 0}, {1: 0}, {1: 0}
            MCTSAgent._backpropagate(env, 1, reward, counts, wins, losses)
            self.assertEqual(expected, (wins[1], losses[1]))

    def test_final_result_uses_empirical_value_not_visit_count(self) -> None:
        action = MCTSAgent._next_move(
            counts={1: 10, 2: 2},
            wins={1: 5, 2: 2},
            losses={1: 5, 2: 0},
        )
        self.assertEqual(2, action)

    def test_does_not_mutate_caller_environment(self) -> None:
        env = make_env([4, 3, 4, 3, 5])
        state = env.state.copy()
        occupied = [column.copy() for column in env.occupied]
        validinputs = env.validinputs.copy()
        turn = env.turn
        MCTSAgent(num_rollouts=40, seed=8).select_action(env)
        np.testing.assert_array_equal(state, env.state)
        self.assertEqual(occupied, env.occupied)
        self.assertEqual(validinputs, env.validinputs)
        self.assertEqual(turn, env.turn)
        self.assertFalse(env.done)

    def test_fixed_seed_is_reproducible(self) -> None:
        actions = REGRESSION_ACTION_SEQUENCES[10]
        left = MCTSAgent(num_rollouts=80, seed=91)
        right = MCTSAgent(num_rollouts=80, seed=91)
        self.assertEqual(left.select_action(make_env(actions)), right.select_action(make_env(actions)))
        self.assertEqual(left.last_counts, right.last_counts)
        self.assertEqual(left.last_wins, right.last_wins)
        self.assertEqual(left.last_losses, right.last_losses)
        self.assertEqual(left.simulation_steps, right.simulation_steps)

    def test_instrumentation_and_validation(self) -> None:
        agent = MCTSAgent(num_rollouts=12, seed=5)
        agent.select_action(make_env([4, 3]))
        self.assertEqual(12, agent.rollouts)
        self.assertGreater(agent.simulation_steps, 0)
        self.assertGreater(agent.action_time, 0.0)
        self.assertGreater(agent.rollout_time, 0.0)
        self.assertLessEqual(agent.rollout_time, agent.action_time)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MCTSAgent(num_rollouts=-1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MCTSAgent(exploration_constant=-0.1)


if __name__ == "__main__":
    unittest.main()
