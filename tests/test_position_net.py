from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from connect4_ai.models.position_net import (
    PositionEvalNet,
    board_to_tensor,
    load_position_checkpoint,
)
from tests.fixtures.connect4_positions import (
    REGRESSION_ACTION_SEQUENCES,
    make_env,
)
CHECKPOINT = Path("checkpoints/pretrained/position_eval.pt")


class PositionEvalNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.checkpoint = load_position_checkpoint(
            CHECKPOINT, device="cpu"
        )

    def test_explicit_single_board_conversion(self) -> None:
        board = np.zeros((7, 6), dtype=np.int64)
        board[6, 5] = 1
        tensor = board_to_tensor(board)
        self.assertEqual((1, 1, 7, 6), tuple(tensor.shape))
        self.assertEqual(torch.float32, tensor.dtype)
        self.assertEqual(1.0, float(tensor[0, 0, 6, 5]))

    def test_explicit_batch_and_nhwc_conversion(self) -> None:
        boards = np.zeros((3, 7, 6), dtype=np.int64)
        nchw = board_to_tensor(boards)
        nhwc = board_to_tensor(boards[..., np.newaxis])
        self.assertEqual((3, 1, 7, 6), tuple(nchw.shape))
        self.assertTrue(torch.equal(nchw, nhwc))

    def test_conventional_row_major_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected board shape"):
            board_to_tensor(np.zeros((6, 7), dtype=np.int64))

    def test_model_output_dimensions_and_probabilities(self) -> None:
        probabilities = self.model(board_to_tensor(np.zeros((4, 7, 6))))
        self.assertEqual((4, 3), tuple(probabilities.shape))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)
        )

    def test_checkpoint_metadata(self) -> None:
        self.assertEqual("PositionEvalNet", self.checkpoint["model_name"])
        self.assertEqual(100, self.checkpoint["epoch"])
        self.assertIsNone(self.checkpoint["optimizer_state_dict"])
        self.assertEqual(
            "6e0f0e7408fea7a6cb873df03de52402e8656e9f449165598cf4966e558d8e77",
            self.checkpoint["training_metadata"]["source_sha256"],
        )

    def test_actual_model_inference_on_cuda(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        model, _ = load_position_checkpoint(CHECKPOINT, device="cuda")
        inputs = board_to_tensor(np.zeros((2, 7, 6)), device="cuda")
        with torch.inference_mode():
            probabilities = model(inputs)
        self.assertTrue(next(model.parameters()).is_cuda)
        self.assertTrue(inputs.is_cuda)
        self.assertTrue(probabilities.is_cuda)
        self.assertTrue(torch.isfinite(probabilities).all())

if __name__ == "__main__":
    unittest.main()
