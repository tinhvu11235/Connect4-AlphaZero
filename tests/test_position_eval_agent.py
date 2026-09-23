from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import torch

from connect4_ai.agents import PositionEvalAgent
from connect4_ai.models import load_position_checkpoint
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)
CHECKPOINT = Path("checkpoints/pretrained/position_eval.pt")


class PositionEvalAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("gate 3 requires CUDA inference")
        cls.device = torch.device("cuda")
        cls.model, _ = load_position_checkpoint(CHECKPOINT, device=cls.device)

    def make_agent(self, *, depth: int = 0) -> PositionEvalAgent:
        return PositionEvalAgent(
            deepcopy(self.model), depth=depth, device=self.device
        )

    def test_position_evaluation_is_finite(self) -> None:
        agent = self.make_agent()
        for actions in REGRESSION_ACTION_SEQUENCES[:10]:
            value = agent.position_evaluation(make_env(actions))
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_exact_player_perspective_sign(self) -> None:
        agent = self.make_agent()
        red_env = make_env([])
        red_raw = agent.position_evaluation(red_env)
        red_cutoff = agent.cutoff_evaluation(red_env)
        self.assertAlmostEqual(red_raw, red_cutoff, places=6)

        yellow_env = make_env([4])
        yellow_raw = agent.position_evaluation(yellow_env)
        yellow_cutoff = agent.cutoff_evaluation(yellow_env)
        self.assertAlmostEqual(-yellow_raw, yellow_cutoff, places=6)

    def test_returns_legal_action_on_fixed_states(self) -> None:
        for actions in REGRESSION_ACTION_SEQUENCES[:12]:
            env = make_env(actions)
            action = self.make_agent(depth=0).select_action(env)
            self.assertIn(action, env.validinputs)

    def test_immediate_winning_move_is_selected(self) -> None:
        env = make_env(TACTICAL_POSITIONS["immediate_vertical_win"])
        agent = self.make_agent(depth=1)
        action = agent.select_action(env)
        self.assertEqual(1, action)
        self.assertEqual(0, agent.inference_calls)

    def test_search_does_not_mutate_original_environment(self) -> None:
        env = make_env(REGRESSION_ACTION_SEQUENCES[12])
        before_state = env.state.copy()
        before_occupied = deepcopy(env.occupied)
        before_turn = env.turn
        before_legal = list(env.validinputs)
        self.make_agent(depth=1).select_action(env)
        self.assertTrue(np.array_equal(before_state, env.state))
        self.assertEqual(before_occupied, env.occupied)
        self.assertEqual(before_turn, env.turn)
        self.assertEqual(before_legal, env.validinputs)

if __name__ == "__main__":
    unittest.main()
