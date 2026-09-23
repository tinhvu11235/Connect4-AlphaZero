from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from benchmark.final_benchmark import GAME_FIELDS, play_game, write_summary
from connect4_ai.agents import RandomAgent


class FinalPipelineTests(unittest.TestCase):
    def test_smoke_game_completes_with_legal_agents(self) -> None:
        game = play_game(
            RandomAgent(seed=1),  # type: ignore[arg-type]
            RandomAgent(seed=2),
            alpha_zero_side="red",
            seed=3,
        )
        self.assertIn(game["result_from_alpha_zero_perspective"], (-1, 0, 1))
        self.assertGreaterEqual(game["move_count"], 7)
        self.assertLessEqual(game["move_count"], 42)
        self.assertGreaterEqual(game["alpha_zero_decision_time"], 0.0)
        self.assertGreaterEqual(game["opponent_decision_time"], 0.0)

    def test_summary_preserves_side_split_and_move_time_units(self) -> None:
        games = [
            {
                "matchup": "AlphaZeroAgent vs RandomAgent",
                "game_id": str(index),
                "seed": str(100 + index),
                "alpha_zero_side": side,
                "opponent": "RandomAgent",
                "winner": winner,
                "result_from_alpha_zero_perspective": str(result),
                "move_count": str(moves),
                "alpha_zero_decision_time": str(alpha_time),
                "opponent_decision_time": str(opponent_time),
            }
            for index, (side, winner, result, moves, alpha_time, opponent_time) in enumerate(
                [
                    ("red", "AlphaZeroAgent", 1, 9, 0.5, 0.4),
                    ("yellow", "draw", 0, 10, 0.6, 0.5),
                ]
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.csv"
            write_summary(output, games)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("2", row["games"])
        self.assertEqual("1", row["wins_as_first"])
        self.assertEqual("1", row["draws_as_second"])
        self.assertAlmostEqual(0.5, float(row["win_rate"]))
        self.assertAlmostEqual(9.5, float(row["average_game_length"]))

    def test_required_raw_columns_are_stable(self) -> None:
        self.assertEqual(
            [
                "matchup",
                "game_id",
                "seed",
                "alpha_zero_side",
                "opponent",
                "winner",
                "result_from_alpha_zero_perspective",
                "move_count",
                "alpha_zero_decision_time",
                "opponent_decision_time",
            ],
            GAME_FIELDS,
        )


if __name__ == "__main__":
    unittest.main()
