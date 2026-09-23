from __future__ import annotations

import unittest

import numpy as np
import torch

from connect4_ai.agents import AlphaGoAgent
from connect4_ai.models import (
    load_policy_checkpoint,
    load_value_checkpoint,
)
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    TACTICAL_POSITIONS,
    make_env,
)


class AlphaGoAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fast, _ = load_policy_checkpoint(
            "checkpoints/pretrained/policy_conn.pt",
            model_type="fast",
            device="cuda",
        )
        cls.policy, _ = load_policy_checkpoint(
            "checkpoints/pretrained/policy_gradient_conn.pt",
            model_type="policy_gradient",
            device="cuda",
        )
        cls.value, _ = load_value_checkpoint(
            "checkpoints/pretrained/value_conn.pt", device="cuda"
        )

    def agent(self, **overrides: object) -> AlphaGoAgent:
        config: dict[str, object] = {
            "weight": 0.75,
            "depth": 5,
            "policy_rollout": False,
            "num_rollouts": 24,
            "seed": 17,
            "device": "cuda",
        }
        config.update(overrides)
        return AlphaGoAgent(
            self.policy, self.value, self.fast, **config  # type: ignore[arg-type]
        )

    def test_returns_legal_actions_on_fixed_states(self) -> None:
        for index, actions in enumerate(REGRESSION_ACTION_SEQUENCES):
            env = make_env(actions)
            action = self.agent(num_rollouts=8, depth=3, seed=index).select_action(env)
            self.assertIn(action, env.validinputs)

    def test_rollout_policy_masks_full_columns(self) -> None:
        env = make_env(TACTICAL_POSITIONS["full_first_column"])
        agent = self.agent(policy_rollout=True)
        for _ in range(20):
            action = agent._rollout_policy_action(env)
            self.assertIn(action, env.validinputs)
            self.assertNotEqual(1, action)

    def test_policy_and_value_inference(self) -> None:
        agent = self.agent(num_rollouts=0)
        env = make_env([4, 3, 4])
        policy = agent.policy_probabilities(env)
        value = agent.value_probabilities(env)
        self.assertEqual((7,), policy.shape)
        self.assertEqual((3,), value.shape)
        self.assertAlmostEqual(1.0, float(policy.sum()), places=5)
        self.assertAlmostEqual(1.0, float(value.sum()), places=5)
        self.assertTrue(np.isfinite(policy).all())
        self.assertTrue(np.isfinite(value).all())

    def test_value_reward_and_backpropagation_perspective(self) -> None:
        agent = self.agent()
        red = make_env([])
        yellow = make_env([4])
        red_probabilities = agent.value_probabilities(red)
        yellow_probabilities = agent.value_probabilities(yellow)
        self.assertAlmostEqual(
            float(red_probabilities[1] - red_probabilities[2]),
            agent.cutoff_reward(red),
            places=6,
        )
        self.assertAlmostEqual(
            float(yellow_probabilities[2] - yellow_probabilities[1]),
            agent.cutoff_reward(yellow),
            places=6,
        )
        red_results = {1: []}
        yellow_results = {1: []}
        AlphaGoAgent._backpropagate(red, 1, 0.4, red_results)
        AlphaGoAgent._backpropagate(yellow, 1, 0.4, yellow_results)
        self.assertEqual([0.4], red_results[1])
        self.assertEqual([-0.4], yellow_results[1])

    def test_search_does_not_mutate_caller(self) -> None:
        env = make_env([4, 3, 4, 3, 5])
        state = env.state.copy()
        occupied = [column.copy() for column in env.occupied]
        validinputs = env.validinputs.copy()
        turn = env.turn
        self.agent().select_action(env)
        np.testing.assert_array_equal(state, env.state)
        self.assertEqual(occupied, env.occupied)
        self.assertEqual(validinputs, env.validinputs)
        self.assertEqual(turn, env.turn)

    def test_fixed_seed_is_reproducible(self) -> None:
        actions = REGRESSION_ACTION_SEQUENCES[12]
        left = self.agent(seed=99)
        right = self.agent(seed=99)
        self.assertEqual(
            left.select_action(make_env(actions)),
            right.select_action(make_env(actions)),
        )
        self.assertEqual(left.last_results, right.last_results)
        self.assertEqual(left.last_visits, right.last_visits)

    def test_checkpoint_factory_and_cuda_execution(self) -> None:
        agent = AlphaGoAgent.from_checkpoints(
            policy_checkpoint="checkpoints/pretrained/policy_gradient_conn.pt",
            value_checkpoint="checkpoints/pretrained/value_conn.pt",
            rollout_policy_checkpoint="checkpoints/pretrained/policy_conn.pt",
            num_rollouts=4,
            depth=2,
            seed=3,
        )
        self.assertEqual("cuda", agent.device.type)
        action = agent.select_action(make_env([]))
        self.assertIn(action, range(1, 8))
        self.assertTrue(next(agent.policy_model.parameters()).is_cuda)
        self.assertTrue(next(agent.value_model.parameters()).is_cuda)
        self.assertTrue(next(agent.rollout_policy_model.parameters()).is_cuda)

    def test_instrumentation_and_configuration_validation(self) -> None:
        agent = self.agent(num_rollouts=7, depth=2)
        agent.select_action(make_env([4, 3]))
        self.assertEqual(7, agent.rollouts)
        self.assertGreater(agent.simulation_steps, 0)
        self.assertGreater(agent.action_time, 0.0)
        self.assertGreater(agent.metrics.policy_inference_calls, 0)
        with self.assertRaisesRegex(ValueError, "between"):
            self.agent(weight=1.1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.agent(depth=-1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.agent(num_rollouts=-1)


if __name__ == "__main__":
    unittest.main()
