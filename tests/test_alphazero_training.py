from __future__ import annotations

import unittest

import numpy as np
import torch

from connect4_ai.models import PolicyGradientNet
from connect4_ai.training.parallel_self_play import (
    ParallelSelfPlayConfig,
    ParallelSelfPlayPool,
)
from connect4_ai.training.train_alphazero import (
    AlphaZeroTrainer,
    initialize_like_keras,
    seed_everything,
)
from connect4_ai.utils.device import CudaRequiredError


def make_trainer(seed: int, *, max_steps: int = 50) -> AlphaZeroTrainer:
    seed_everything(seed)
    model = PolicyGradientNet()
    initialize_like_keras(model)
    return AlphaZeroTrainer(
        model,
        device="cuda",
        seed=seed,
        opponent_weight=0.05,
        opponent_rollouts=2,
        max_steps=max_steps,
    )


class AlphaZeroTrainingTests(unittest.TestCase):
    def test_self_play_games_complete(self) -> None:
        trainer = make_trainer(81001)
        red = trainer.play_episode(learner_side="red")
        yellow = trainer.play_episode(learner_side="yellow")
        self.assertTrue(red.completed)
        self.assertTrue(yellow.completed)
        self.assertGreater(red.policy_steps, 0)
        self.assertGreater(yellow.policy_steps, 0)

    def test_reward_discount_and_illegal_assignment(self) -> None:
        trainer = make_trainer(81002)
        actual = trainer.discounted_returns(
            rewards=[0.0, 0.0, 1.0], wrong_moves=[0, -1, 0]
        )
        self.assertAlmostEqual(0.95, actual[0], places=6)
        self.assertEqual([-1.0, 1.0], actual[1:])

    def test_parameter_update_and_frozen_opponent(self) -> None:
        trainer = make_trainer(81003)
        opponent_before = {
            name: tensor.detach().clone()
            for name, tensor in trainer.opponent_model.state_dict().items()
        }
        metric = trainer.train_batch(learner_side="red", batch_size=2)
        self.assertNotEqual(0.0, metric["loss"])
        self.assertGreater(metric["parameter_change_l2"], 0.0)
        self.assertGreater(metric["gradient_norm_before_clip"], 0.0)
        self.assertTrue(trainer.optimizer.state)
        for name, tensor in trainer.opponent_model.state_dict().items():
            torch.testing.assert_close(opponent_before[name], tensor)

    def test_detached_collection_preserves_episode_semantics(self) -> None:
        graph_trainer = make_trainer(81005)
        graph_trace = graph_trainer.play_episode(learner_side="red")
        detached_trainer = make_trainer(81005)
        detached_trace = detached_trainer.collect_episode(learner_side="red")
        self.assertEqual(graph_trace.episode_reward, detached_trace["episode_reward"])
        self.assertEqual(graph_trace.completed, detached_trace["completed"])
        self.assertEqual(graph_trace.policy_steps, detached_trace["policy_steps"])
        np.testing.assert_array_equal(
            graph_trace.combined_returns, detached_trace["returns"]
        )

    def test_vectorized_update_changes_parameters(self) -> None:
        trainer = make_trainer(81006)
        before = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
        metric = trainer.train_batch_vectorized(
            learner_side="yellow", batch_size=2
        )
        self.assertEqual(2, metric["completed_games"])
        self.assertGreater(metric["policy_steps"], 0)
        self.assertTrue(
            any(
                not torch.equal(before[name], tensor)
                for name, tensor in trainer.model.state_dict().items()
            )
        )

    def test_evaluation_collection_cannot_mutate_either_model(self) -> None:
        trainer = make_trainer(81008)
        candidate_before = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
        reference_before = {
            name: tensor.detach().clone()
            for name, tensor in trainer.opponent_model.state_dict().items()
        }
        trainer.collect_episode(learner_side="red")
        for name, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(candidate_before[name], tensor)
        for name, tensor in trainer.opponent_model.state_dict().items():
            torch.testing.assert_close(reference_before[name], tensor)

    def test_parallel_pool_uses_single_cuda_service_and_completes(self) -> None:
        trainer = make_trainer(81007)
        config = ParallelSelfPlayConfig(
            workers=2,
            inference_batch_size=2,
            opponent_rollouts=2,
            max_steps=50,
        )
        with ParallelSelfPlayPool(
            trainer.model,
            trainer.opponent_model,
            config=config,
            device="cuda",
        ) as pool:
            # Workers own no model copy: the centralized CUDA service observes
            # the trainer's exact live candidate and frozen reference objects.
            self.assertIs(pool.learner_model, trainer.model)
            self.assertIs(pool.opponent_model, trainer.opponent_model)
            trajectories, metrics = pool.generate(
                2, seed=81007, learner_side="red"
            )
        self.assertEqual(2, len(trajectories))
        self.assertEqual(2, metrics["completed_games"])
        self.assertGreater(metrics["inference_requests"], 0)
        self.assertLessEqual(metrics["maximum_inference_batch_size"], 2)
        self.assertGreater(metrics["model_inference_batches"], 0)
        self.assertGreaterEqual(metrics["average_model_batch_size"], 1.0)
        self.assertGreaterEqual(metrics["mcts_seconds"], 0.0)
        self.assertGreater(metrics["forward_seconds"], 0.0)
        self.assertTrue(all(item["learner_side"] == "red" for item in trajectories))

    def test_short_run_is_reproducible_with_fixed_seed(self) -> None:
        first = make_trainer(81004)
        first_metric = first.train_batch(learner_side="red", batch_size=1)
        first_state = {
            name: tensor.detach().clone()
            for name, tensor in first.model.state_dict().items()
        }
        second = make_trainer(81004)
        second_metric = second.train_batch(learner_side="red", batch_size=1)
        self.assertEqual(first_metric["loss"], second_metric["loss"])
        self.assertEqual(
            first_metric["episode_rewards"], second_metric["episode_rewards"]
        )
        for name, tensor in second.model.state_dict().items():
            torch.testing.assert_close(first_state[name], tensor)

    def test_training_rejects_cpu(self) -> None:
        with self.assertRaisesRegex(CudaRequiredError, "requires CUDA"):
            AlphaZeroTrainer(PolicyGradientNet(), device="cpu")


if __name__ == "__main__":
    unittest.main()
