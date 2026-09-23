"""Outcome accounting and TensorBoard logging for checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _summary(results: list[int]) -> dict[str, float | int]:
    games = len(results)
    wins = sum(result == 1 for result in results)
    losses = sum(result == -1 for result in results)
    draws = sum(result == 0 for result in results)
    denominator = max(games, 1)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / denominator,
        "loss_rate": losses / denominator,
        "draw_rate": draws / denominator,
        "score": (wins - losses) / denominator,
    }


def summarize_arena_results(
    games: Iterable[tuple[str, float | int]],
) -> dict[str, Any]:
    """Summarize candidate outcomes, including an exact first/second split."""

    normalized: list[tuple[str, int]] = []
    for side, reward in games:
        if side not in ("red", "yellow"):
            raise ValueError("arena side must be 'red' or 'yellow'")
        result = 1 if reward > 0 else -1 if reward < 0 else 0
        normalized.append((side, result))
    aggregate = _summary([result for _, result in normalized])
    as_first = _summary([result for side, result in normalized if side == "red"])
    as_second = _summary([result for side, result in normalized if side == "yellow"])
    return {
        **aggregate,
        "as_first": as_first,
        "as_second": as_second,
        "first_move_gap": float(as_first["score"]) - float(as_second["score"]),
    }


def promotion_threshold_met(
    *, games: int, score: float, minimum_games: int, threshold: float
) -> bool:
    """Return the exact checkpoint criterion; draws are not treated as half-wins."""

    return games >= minimum_games and score >= threshold


def add_arena_scalars(
    writer: Any,
    summary: dict[str, Any],
    global_step: int,
    *,
    promoted: bool | None = None,
) -> None:
    """Emit the stable arena metric namespace used by the training runner."""

    for key in (
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "score",
    ):
        tag = "arena/score_vs_previous" if key == "score" else f"arena/{key}"
        writer.add_scalar(tag, float(summary[key]), global_step)
    for side_key, tag_prefix in (
        ("as_first", "arena/as_first"),
        ("as_second", "arena/as_second"),
    ):
        side = summary[side_key]
        for key in ("win_rate", "loss_rate", "draw_rate"):
            writer.add_scalar(f"{tag_prefix}/{key}", float(side[key]), global_step)
        score_tag = (
            "arena/score_as_first"
            if side_key == "as_first"
            else "arena/score_as_second"
        )
        writer.add_scalar(score_tag, float(side["score"]), global_step)
    writer.add_scalar(
        "arena/first_move_gap", float(summary["first_move_gap"]), global_step
    )
    if promoted is not None:
        writer.add_scalar("arena/promoted", float(promoted), global_step)


__all__ = [
    "add_arena_scalars",
    "promotion_threshold_met",
    "summarize_arena_results",
]
