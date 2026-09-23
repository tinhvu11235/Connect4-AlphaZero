from __future__ import annotations

import unittest

import numpy as np
import torch

from connect4_ai.agents import AlphaZeroAgent
from connect4_ai.models import load_policy_checkpoint, policy_board_to_tensor
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


class AlphaZeroAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.checkpoint = load_policy_checkpoint(
            "checkpoints/pretrained/connzero_book_iter_004.pt",
            model_type="policy_gradient",
            device="cuda",
        )

    def agent(self, **overrides: object) -> AlphaZeroAgent:
        config: dict[str, object] = {
            "weight": 0.75,
            "num_rollouts": 24,
            "seed": 17,
            "device": "cuda",
        }
        config.update(overrides)
        return AlphaZeroAgent(self.model, **config)  # type: ignore[arg-type]

    def test_model_inference_is_finite_on_cuda(self) -> None:
        boards = np.stack(
            [make_env(actions).state for actions in REGRESSION_ACTION_SEQUENCES[:5]]
        ).astype(np.float32)
        with torch.inference_mode():
            actual = self.model(policy_board_to_tensor(boards, device="cuda"))
        self.assertEqual((5, 7), tuple(actual.shape))
        self.assertTrue(actual.is_cuda)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(
            torch.allclose(actual.sum(dim=1), torch.ones(5, device="cuda"), atol=1e-5)
        )

    def test_returns_legal_actions_on_fixed_states(self) -> None:
        for index, actions in enumerate(REGRESSION_ACTION_SEQUENCES):
            env = make_env(actions)
            action = self.agent(num_rollouts=8, seed=index).select_action(env)
            self.assertIn(action, env.validinputs)

    def test_search_phase_instrumentation_is_populated(self) -> None:
        agent = self.agent(num_rollouts=4, seed=123)
        agent.select_action(make_env([4, 3]))
        self.assertEqual(1, agent.metrics.policy_inference_calls)
        self.assertGreater(agent.metrics.policy_inference_time, 0.0)
        self.assertGreater(agent.metrics.rollout_time, 0.0)

    def test_checkpoint_loading_and_cuda_placement(self) -> None:
        agent = AlphaZeroAgent.from_checkpoint(
            "checkpoints/pretrained/connzero_book_iter_004.pt",
            num_rollouts=4,
            seed=3,
        )
        self.assertEqual("cuda", agent.device.type)
        self.assertTrue(next(agent.model.parameters()).is_cuda)
        self.assertEqual(4, self.checkpoint["iteration"])
        self.assertEqual(
            "0eef19648c8c7f089c5f5b0c23ad2223856981076147274ea4e66f3e558b021e",
            self.checkpoint["training_metadata"]["source_sha256"],
        )

    def test_full_column_is_excluded_from_root_statistics(self) -> None:
        env = make_env(TACTICAL_POSITIONS["full_first_column"])
        agent = self.agent()
        action = agent.select_action(env)
        self.assertNotEqual(1, action)
        self.assertNotIn(1, agent.last_priors)
        self.assertNotIn(1, agent.last_visits)

    def test_reward_is_backpropagated_from_root_perspective(self) -> None:
        red = make_env([])
        yellow = make_env([4])
        red_results = {1: []}
        yellow_results = {1: []}
        AlphaZeroAgent._backpropagate(red, 1, 1.0, red_results)
        AlphaZeroAgent._backpropagate(yellow, 1, 1.0, yellow_results)
        self.assertEqual([1.0], red_results[1])
        self.assertEqual([-1.0], yellow_results[1])

    def test_search_does_not_mutate_caller(self) -> None:
        env = make_env([4, 3, 4, 3, 5])
        state = env.state.copy()
        occupied = [column.copy() for column in env.occupied]
        legal = env.validinputs.copy()
        turn = env.turn
        self.agent().select_action(env)
        np.testing.assert_array_equal(state, env.state)
        self.assertEqual(occupied, env.occupied)
        self.assertEqual(legal, env.validinputs)
        self.assertEqual(turn, env.turn)

    def test_fixed_seed_is_reproducible(self) -> None:
        actions = REGRESSION_ACTION_SEQUENCES[14]
        left = self.agent(seed=99)
        right = self.agent(seed=99)
        self.assertEqual(
            left.select_action(make_env(actions)),
            right.select_action(make_env(actions)),
        )
        self.assertEqual(left.last_results, right.last_results)
        self.assertEqual(left.last_visits, right.last_visits)

    def test_no_silent_cpu_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            AlphaZeroAgent(
                self.model,
                num_rollouts=1,
                device="cpu",
                allow_cpu=False,
            )


if __name__ == "__main__":
    unittest.main()
