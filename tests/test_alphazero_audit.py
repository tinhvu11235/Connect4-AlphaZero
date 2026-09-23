from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest

import torch

from connect4_ai.models import PolicyGradientNet
from connect4_ai.training.arena_metrics import (
    add_arena_scalars,
    promotion_threshold_met,
    summarize_arena_results,
)
from connect4_ai.training.run_full_alphazero import (
    _empty_accumulators,
    _state_payload,
)
from connect4_ai.training.parallel_self_play import (
    policy_gradient_loss_from_logits,
)
from connect4_ai.training.train_alphazero import save_training_checkpoint


class RecordingWriter:
    def __init__(self) -> None:
        self.scalars: dict[str, tuple[float, int]] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars[tag] = (float(value), step)


def fake_trainer() -> SimpleNamespace:
    model = PolicyGradientNet()
    opponent = deepcopy(model).eval()
    for parameter in opponent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, eps=1e-7)
    return SimpleNamespace(
        model=model,
        opponent_model=opponent,
        optimizer=optimizer,
        seed=17,
        device=torch.device("cpu"),
        sample_generator=torch.Generator().manual_seed(17),
        opponent=SimpleNamespace(_rng=random.Random(18)),
        _opponent_seed=19,
    )


class ArenaAccountingTests(unittest.TestCase):
    def test_wld_score_and_side_accounting(self) -> None:
        summary = summarize_arena_results(
            [("red", 1), ("red", 0), ("yellow", -1), ("yellow", 1)]
        )
        self.assertEqual((2, 1, 1), (summary["wins"], summary["losses"], summary["draws"]))
        self.assertAlmostEqual(0.25, summary["score"])
        self.assertEqual(2, summary["as_first"]["games"])
        self.assertEqual(2, summary["as_second"]["games"])
        self.assertAlmostEqual(0.5, summary["as_first"]["score"])
        self.assertAlmostEqual(0.0, summary["as_second"]["score"])
        self.assertAlmostEqual(0.5, summary["first_move_gap"])

    def test_promotion_uses_score_not_win_rate(self) -> None:
        # 55% wins, 20% losses, and 25% draws gives score 0.35, not 0.10.
        summary = summarize_arena_results(
            [("red" if i % 2 == 0 else "yellow", result)
             for i, result in enumerate([1] * 55 + [-1] * 20 + [0] * 25)]
        )
        self.assertAlmostEqual(0.55, summary["win_rate"])
        self.assertAlmostEqual(0.35, summary["score"])
        self.assertTrue(
            promotion_threshold_met(
                games=1000, score=0.10, minimum_games=1000, threshold=0.10
            )
        )
        self.assertFalse(
            promotion_threshold_met(
                games=999, score=1.0, minimum_games=1000, threshold=0.10
            )
        )

    def test_tensorboard_arena_metric_generation(self) -> None:
        writer = RecordingWriter()
        summary = summarize_arena_results([("red", 1), ("yellow", 0)])
        add_arena_scalars(writer, summary, 123, promoted=True)
        required = {
            "arena/win_rate",
            "arena/loss_rate",
            "arena/draw_rate",
            "arena/score_vs_previous",
            "arena/games",
            "arena/wins",
            "arena/losses",
            "arena/draws",
            "arena/promoted",
            "arena/as_first/win_rate",
            "arena/as_first/loss_rate",
            "arena/as_first/draw_rate",
            "arena/score_as_first",
            "arena/as_second/win_rate",
            "arena/as_second/loss_rate",
            "arena/as_second/draw_rate",
            "arena/score_as_second",
            "arena/first_move_gap",
        }
        self.assertTrue(required.issubset(writer.scalars))
        self.assertEqual((1.0, 123), writer.scalars["arena/promoted"])


class CheckpointInvariantTests(unittest.TestCase):
    def test_extreme_logits_have_finite_policy_gradient(self) -> None:
        logits = torch.tensor([[-50.0, 50.0]], requires_grad=True)
        actions = torch.tensor([0], dtype=torch.long)
        returns = torch.tensor([1.0])
        loss = policy_gradient_loss_from_logits(logits, actions, returns)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(
            logits.grad, torch.tensor([[-1.0, 1.0]]), atol=1e-6, rtol=1e-6
        )

    def test_checkpoint_rejects_non_finite_candidate(self) -> None:
        trainer = fake_trainer()
        with torch.no_grad():
            next(trainer.model.parameters()).view(-1)[0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FloatingPointError, "candidate tensor"):
                save_training_checkpoint(
                    Path(directory) / "corrupt.pt",
                    trainer,
                    iteration=0,
                    config={},
                    history=[],
                )

    def test_candidate_reference_are_independent_and_checkpointed_correctly(self) -> None:
        trainer = fake_trainer()
        candidate_parameter = next(trainer.model.parameters())
        reference_parameter = next(trainer.opponent_model.parameters())
        self.assertNotEqual(candidate_parameter.data_ptr(), reference_parameter.data_ptr())
        reference_before = reference_parameter.detach().clone()
        with torch.no_grad():
            candidate_parameter.add_(1.0)
        torch.testing.assert_close(reference_before, reference_parameter)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "promoted.pt"
            save_training_checkpoint(
                output,
                trainer,
                iteration=3,
                config={"promoted": True},
                history=[],
            )
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
        candidate_name = next(iter(trainer.model.state_dict()))
        torch.testing.assert_close(
            checkpoint["model_state_dict"][candidate_name],
            trainer.model.state_dict()[candidate_name],
        )
        torch.testing.assert_close(
            checkpoint["opponent_model_state_dict"][candidate_name],
            trainer.opponent_model.state_dict()[candidate_name],
        )
        self.assertFalse(
            torch.equal(
                checkpoint["model_state_dict"][candidate_name],
                checkpoint["opponent_model_state_dict"][candidate_name],
            )
        )

    def test_resume_payload_restores_training_and_reference_state(self) -> None:
        trainer = fake_trainer()
        reference_optimizer = deepcopy(trainer.optimizer.state_dict())
        payload = _state_payload(
            trainer,
            iteration=2,
            in_iteration=True,
            episodes=120,
            batches=6,
            running_rewards=deque([1.0, 0.0, -1.0], maxlen=1000),
            history=[{"batch": 6}],
            accumulators=_empty_accumulators(),
            replay_buffer=deque([{"generation": 2}], maxlen=10),
            total_games=220,
            total_positions=1700,
            total_optimizer_steps=11,
            reference_optimizer_state_dict=reference_optimizer,
            config={"seed": 17},
        )
        restored = fake_trainer()
        restored.model.load_state_dict(payload["model_state_dict"])
        restored.opponent_model.load_state_dict(payload["opponent_model_state_dict"])
        restored.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.assertEqual(2, payload["format_version"])
        self.assertEqual(120, payload["episodes"])
        self.assertEqual(220, payload["total_games"])
        self.assertEqual(1, len(payload["replay_buffer"]))
        self.assertIn("reference_optimizer_state_dict", payload)
        for name, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(tensor, restored.model.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
