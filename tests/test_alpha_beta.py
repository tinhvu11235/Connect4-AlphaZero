from __future__ import annotations

import unittest

from connect4_ai.agents import AlphaBetaAgent, DepthMinimaxAgent
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


class AlphaBetaAgentTests(unittest.TestCase):
    def test_returns_legal_actions_on_fixed_states(self) -> None:
        agent = AlphaBetaAgent(depth=1, seed=11)
        for actions in REGRESSION_ACTION_SEQUENCES:
            env = make_env(actions)
            self.assertIn(agent.select_action(env), env.validinputs)

    def test_immediate_winning_move(self) -> None:
        env = make_env(TACTICAL_POSITIONS["immediate_vertical_win"])
        self.assertEqual(1, AlphaBetaAgent(depth=2, seed=1).select_action(env))

    def test_forced_blocking_move(self) -> None:
        env = make_env(TACTICAL_POSITIONS["forced_vertical_block"])
        self.assertEqual(4, AlphaBetaAgent(depth=1, seed=1).select_action(env))

    def test_full_column_avoidance(self) -> None:
        env = make_env(TACTICAL_POSITIONS["full_first_column"])
        action = AlphaBetaAgent(depth=1, seed=1).select_action(env)
        self.assertIn(action, env.validinputs)
        self.assertNotEqual(1, action)

    def test_negative_depth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            AlphaBetaAgent(depth=-1)

    def test_matches_minimax_and_never_visits_more_nodes(self) -> None:
        cases = [
            (0, REGRESSION_ACTION_SEQUENCES),
            (1, REGRESSION_ACTION_SEQUENCES),
            (2, REGRESSION_ACTION_SEQUENCES),
            (3, REGRESSION_ACTION_SEQUENCES[:12]),
        ]
        for depth, positions in cases:
            for index, actions in enumerate(positions):
                seed = 18000 + depth * 100 + index
                minimax = DepthMinimaxAgent(depth=depth, seed=seed)
                alpha_beta = AlphaBetaAgent(depth=depth, seed=seed)
                minimax_action = minimax.select_action(make_env(actions))
                alpha_beta_action = alpha_beta.select_action(make_env(actions))
                self.assertEqual(
                    minimax_action,
                    alpha_beta_action,
                    msg=f"depth={depth}, actions={actions}",
                )
                self.assertEqual(
                    minimax.last_action_values,
                    alpha_beta.last_action_values,
                    msg=f"value mismatch at depth={depth}, actions={actions}",
                )
                self.assertLessEqual(
                    alpha_beta.nodes_visited,
                    minimax.nodes_visited,
                    msg=f"depth={depth}, actions={actions}",
                )

    def test_pruning_metrics_are_recorded(self) -> None:
        agent = AlphaBetaAgent(depth=3, seed=3)
        agent.select_action(make_env(TACTICAL_POSITIONS["empty"]))
        self.assertGreater(agent.nodes_visited, 0)
        self.assertGreater(agent.nodes_pruned, 0)
        self.assertGreater(agent.search_time, 0.0)


if __name__ == "__main__":
    unittest.main()
