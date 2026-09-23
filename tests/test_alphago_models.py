from __future__ import annotations

import unittest

import numpy as np
import torch

from connect4_ai.models import (
    FastPolicyNet,
    PolicyGradientNet,
    ValueNet,
    load_policy_checkpoint,
    load_value_checkpoint,
    policy_board_to_tensor,
)
from tests.fixtures.connect4_positions import REGRESSION_ACTION_SEQUENCES, make_env


class AlphaGoModelTests(unittest.TestCase):
    def test_network_input_and_output_shapes(self) -> None:
        boards = np.stack(
            [make_env(actions).state for actions in REGRESSION_ACTION_SEQUENCES[:4]]
        )
        tensor = policy_board_to_tensor(boards)
        self.assertEqual((4, 1, 7, 6), tuple(tensor.shape))
        self.assertEqual((4, 7), tuple(FastPolicyNet()(tensor).shape))
        self.assertEqual((4, 7), tuple(PolicyGradientNet()(tensor).shape))
        self.assertEqual((4, 3), tuple(ValueNet()(tensor).shape))

    def test_checkpoint_load_save_and_metadata(self) -> None:
        cases = (
            ("checkpoints/pretrained/policy_conn.pt", "fast", "FastPolicyNet"),
            (
                "checkpoints/pretrained/policy_gradient_conn.pt",
                "policy_gradient",
                "PolicyGradientNet",
            ),
        )
        for path, model_type, expected_name in cases:
            model, checkpoint = load_policy_checkpoint(
                path, model_type=model_type, device="cpu"
            )
            self.assertEqual(expected_name, checkpoint["model_name"])
            self.assertTrue(model.state_dict())
            self.assertIn("source_sha256", checkpoint["training_metadata"])
        value, checkpoint = load_value_checkpoint(
            "checkpoints/pretrained/value_conn.pt", device="cpu"
        )
        self.assertIsInstance(value, ValueNet)
        self.assertEqual("ValueNet", checkpoint["model_name"])

    def test_policy_and_value_inference_run_on_cuda(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        board = policy_board_to_tensor(np.zeros((7, 6)), device="cuda")
        fast, _ = load_policy_checkpoint(
            "checkpoints/pretrained/policy_conn.pt",
            model_type="fast",
            device="cuda",
        )
        policy, _ = load_policy_checkpoint(
            "checkpoints/pretrained/policy_gradient_conn.pt",
            model_type="policy_gradient",
            device="cuda",
        )
        value, _ = load_value_checkpoint(
            "checkpoints/pretrained/value_conn.pt", device="cuda"
        )
        with torch.inference_mode():
            outputs = (fast(board), policy(board), value(board))
        self.assertTrue(all(output.is_cuda for output in outputs))
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))

if __name__ == "__main__":
    unittest.main()
