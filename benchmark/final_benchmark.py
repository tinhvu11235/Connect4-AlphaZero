"""Reproducible round-robin benchmark for the trained AlphaZero checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_ai.utils.monitoring import LiveStatus, PerformanceMonitor  # noqa: E402
from connect4_ai.agents import (  # noqa: E402
    AlphaBetaAgent,
    AlphaGoAgent,
    AlphaZeroAgent,
    DepthMinimaxAgent,
    MCTSAgent,
    PositionEvalAgent,
    RandomAgent,
    RuleBasedThink3Agent,
)
from connect4_ai.env import ConnectFourEnv  # noqa: E402


GAME_FIELDS = [
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
]
SUMMARY_FIELDS = [
    "opponent",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "loss_rate",
    "draw_rate",
    "wins_as_first",
    "losses_as_first",
    "draws_as_first",
    "wins_as_second",
    "losses_as_second",
    "draws_as_second",
    "average_game_length",
    "average_alpha_zero_move_time",
    "average_opponent_move_time",
]
OPPONENTS = [
    "RandomAgent",
    "RuleBasedThink3Agent",
    "DepthMinimaxAgent",
    "AlphaBetaAgent",
    "PositionEvalAgent",
    "MCTSAgent",
    "AlphaGoAgent",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_opponent(name: str, args: argparse.Namespace, seed: int) -> Any:
    if name == "RandomAgent":
        return RandomAgent(seed=seed)
    if name == "RuleBasedThink3Agent":
        return RuleBasedThink3Agent(seed=seed)
    if name == "DepthMinimaxAgent":
        return DepthMinimaxAgent(depth=args.minimax_depth, seed=seed)
    if name == "AlphaBetaAgent":
        return AlphaBetaAgent(depth=args.alpha_beta_depth, seed=seed)
    if name == "PositionEvalAgent":
        return PositionEvalAgent.from_checkpoint(
            args.position_checkpoint,
            depth=args.position_depth,
            device="cuda",
        )
    if name == "MCTSAgent":
        return MCTSAgent(
            num_rollouts=args.mcts_rollouts,
            exploration_constant=args.mcts_exploration_constant,
            seed=seed,
        )
    if name == "AlphaGoAgent":
        return AlphaGoAgent.from_checkpoints(
            policy_checkpoint=args.policy_gradient_checkpoint,
            value_checkpoint=args.value_checkpoint,
            rollout_policy_checkpoint=args.fast_policy_checkpoint,
            weight=args.alphago_weight,
            depth=args.alphago_depth,
            policy_rollout=args.alphago_policy_rollout,
            num_rollouts=args.alphago_rollouts,
            seed=seed,
            device="cuda",
        )
    raise ValueError(f"unsupported opponent: {name}")


def play_game(
    alpha_zero: AlphaZeroAgent,
    opponent: Any,
    *,
    alpha_zero_side: str,
    seed: int,
) -> dict[str, Any]:
    env = ConnectFourEnv(seed=seed)
    alpha_zero.set_seed(seed + 1)
    opponent.set_seed(seed + 2)
    alpha_zero_time = 0.0
    opponent_time = 0.0
    move_count = 0
    while not env.done:
        alpha_turn = env.turn == alpha_zero_side
        acting_agent = alpha_zero if alpha_turn else opponent
        started = perf_counter()
        action = acting_agent.select_action(env)
        elapsed = perf_counter() - started
        if action not in env.validinputs:
            raise RuntimeError(
                f"invalid move {action} from {type(acting_agent).__name__}; "
                f"legal={env.validinputs}"
            )
        if alpha_turn:
            alpha_zero_time += elapsed
        else:
            opponent_time += elapsed
        env.step(action)
        move_count += 1
    alpha_is_red = alpha_zero_side == "red"
    if env.reward == 0:
        result = 0
        winner = "draw"
    else:
        red_won = env.reward == 1
        alpha_won = red_won == alpha_is_red
        result = 1 if alpha_won else -1
        winner = "AlphaZeroAgent" if alpha_won else type(opponent).__name__
    return {
        "winner": winner,
        "result_from_alpha_zero_perspective": result,
        "move_count": move_count,
        "alpha_zero_decision_time": alpha_zero_time,
        "opponent_decision_time": opponent_time,
    }


def append_game(output: Path, row: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GAME_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in GAME_FIELDS})


def read_games(output: Path) -> list[dict[str, str]]:
    if not output.exists() or output.stat().st_size == 0:
        return []
    with output.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_summary(output: Path, games: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for game in games:
        grouped[game["opponent"]].append(game)
    rows: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        selected = grouped.get(opponent, [])
        if not selected:
            continue
        results = [int(item["result_from_alpha_zero_perspective"]) for item in selected]
        wins = sum(result == 1 for result in results)
        losses = sum(result == -1 for result in results)
        draws = sum(result == 0 for result in results)
        first = [item for item in selected if item["alpha_zero_side"] == "red"]
        second = [item for item in selected if item["alpha_zero_side"] == "yellow"]
        alpha_moves = sum(
            (int(item["move_count"]) + (1 if item["alpha_zero_side"] == "red" else 0))
            // 2
            for item in selected
        )
        opponent_moves = sum(int(item["move_count"]) for item in selected) - alpha_moves

        def count(subset: list[dict[str, str]], result: int) -> int:
            return sum(
                int(item["result_from_alpha_zero_perspective"]) == result
                for item in subset
            )

        total = len(selected)
        rows.append(
            {
                "opponent": opponent,
                "games": total,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / total,
                "loss_rate": losses / total,
                "draw_rate": draws / total,
                "wins_as_first": count(first, 1),
                "losses_as_first": count(first, -1),
                "draws_as_first": count(first, 0),
                "wins_as_second": count(second, 1),
                "losses_as_second": count(second, -1),
                "draws_as_second": count(second, 0),
                "average_game_length": sum(int(item["move_count"]) for item in selected)
                / total,
                "average_alpha_zero_move_time": sum(
                    float(item["alpha_zero_decision_time"]) for item in selected
                )
                / alpha_moves,
                "average_opponent_move_time": sum(
                    float(item["opponent_decision_time"]) for item in selected
                )
                / opponent_moves,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "alpha_zero_checkpoint": str(args.alpha_zero_checkpoint),
        "alpha_zero_checkpoint_sha256": sha256(args.alpha_zero_checkpoint),
        "games_per_matchup": args.games_per_matchup,
        "starting_side_allocation": {
            "alpha_zero_first": args.games_per_matchup // 2,
            "alpha_zero_second": args.games_per_matchup // 2,
        },
        "base_seed": args.seed,
        "opponents": OPPONENTS,
        "alpha_zero": {
            "weight": args.alpha_zero_weight,
            "rollouts": args.alpha_zero_rollouts,
        },
        "depth_minimax_depth": args.minimax_depth,
        "alpha_beta_depth": args.alpha_beta_depth,
        "position_eval": {
            "depth": args.position_depth,
            "checkpoint": str(args.position_checkpoint),
        },
        "mcts": {
            "rollouts": args.mcts_rollouts,
            "exploration_constant": args.mcts_exploration_constant,
        },
        "alphago": {
            "weight": args.alphago_weight,
            "depth": args.alphago_depth,
            "rollouts": args.alphago_rollouts,
            "policy_rollout": args.alphago_policy_rollout,
            "policy_checkpoint": str(args.policy_gradient_checkpoint),
            "value_checkpoint": str(args.value_checkpoint),
            "rollout_policy_checkpoint": str(args.fast_policy_checkpoint),
        },
        "device": "cuda",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha-zero-checkpoint",
        type=Path,
        default=Path("checkpoints/training/connzero_final_trained.pt"),
    )
    parser.add_argument("--games-per-matchup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--alpha-zero-weight", type=float, default=0.75)
    parser.add_argument("--alpha-zero-rollouts", type=int, default=584)
    parser.add_argument("--minimax-depth", type=int, default=3)
    parser.add_argument("--alpha-beta-depth", type=int, default=3)
    parser.add_argument("--position-depth", type=int, default=3)
    parser.add_argument("--mcts-rollouts", type=int, default=100)
    parser.add_argument("--mcts-exploration-constant", type=float, default=1.4)
    parser.add_argument("--alphago-weight", type=float, default=0.75)
    parser.add_argument("--alphago-depth", type=int, default=45)
    parser.add_argument("--alphago-rollouts", type=int, default=584)
    parser.add_argument(
        "--alphago-policy-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--position-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrained/position_eval.pt"),
    )
    parser.add_argument(
        "--policy-gradient-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrained/policy_gradient_conn.pt"),
    )
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrained/value_conn.pt"),
    )
    parser.add_argument(
        "--fast-policy-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrained/policy_conn.pt"),
    )
    parser.add_argument(
        "--games-output",
        type=Path,
        default=Path("results/benchmark_games.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/benchmark_summary.csv"),
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("results/final_benchmark_config.json"),
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("results/gpu_metrics.csv")
    )
    args = parser.parse_args()
    if args.games_per_matchup < 2 or args.games_per_matchup % 2:
        parser.error("games-per-matchup must be a positive even number of at least 2")
    if not args.alpha_zero_checkpoint.exists():
        parser.error(f"checkpoint does not exist: {args.alpha_zero_checkpoint}")

    config = config_from_args(args)
    if args.config_output.exists() and args.games_output.exists():
        existing = json.loads(args.config_output.read_text(encoding="utf-8"))
        ignored_metadata = {"created_at", "validation"}
        comparable_existing = {
            key: value
            for key, value in existing.items()
            if key not in ignored_metadata
        }
        comparable_new = {
            key: value
            for key, value in config.items()
            if key not in ignored_metadata
        }
        if comparable_existing != comparable_new:
            raise RuntimeError(
                "existing benchmark outputs use a different configuration; "
                "choose different output paths"
            )
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    alpha_zero = AlphaZeroAgent.from_checkpoint(
        args.alpha_zero_checkpoint,
        weight=args.alpha_zero_weight,
        num_rollouts=args.alpha_zero_rollouts,
        seed=args.seed,
        device="cuda",
    )
    existing_games = read_games(args.games_output)
    completed = {
        (row["opponent"], int(row["game_id"])) for row in existing_games
    }
    status = LiveStatus()
    started = perf_counter()
    completed_this_run = 0
    with PerformanceMonitor(
        args.metrics_output,
        status,
        interval_seconds=1.0,
        run_label="final_benchmark",
    ):
        for opponent_index, opponent_name in enumerate(OPPONENTS):
            opponent = make_opponent(
                opponent_name,
                args,
                args.seed + opponent_index * 100_000,
            )
            for game_id in range(args.games_per_matchup):
                if (opponent_name, game_id) in completed:
                    continue
                alpha_side = (
                    "red"
                    if game_id < args.games_per_matchup // 2
                    else "yellow"
                )
                game_seed = args.seed + opponent_index * 100_000 + game_id
                game = play_game(
                    alpha_zero,
                    opponent,
                    alpha_zero_side=alpha_side,
                    seed=game_seed,
                )
                row = {
                    "matchup": f"AlphaZeroAgent vs {opponent_name}",
                    "game_id": game_id,
                    "seed": game_seed,
                    "alpha_zero_side": alpha_side,
                    "opponent": opponent_name,
                    **game,
                }
                append_game(args.games_output, row)
                existing_games.append({key: str(value) for key, value in row.items()})
                completed_this_run += 1
                elapsed = perf_counter() - started
                status.update(
                    opponent_name,
                    games_per_sec=completed_this_run / elapsed,
                )
                print(
                    f"opponent={opponent_name} game={game_id + 1}/"
                    f"{args.games_per_matchup} side={alpha_side} "
                    f"result={game['result_from_alpha_zero_perspective']}",
                    flush=True,
                )

    write_summary(args.summary_output, existing_games)
    final_games = read_games(args.games_output)
    for opponent_name in OPPONENTS:
        selected = [row for row in final_games if row["opponent"] == opponent_name]
        if len(selected) != args.games_per_matchup:
            raise RuntimeError(
                f"incomplete benchmark for {opponent_name}: {len(selected)} games"
            )
        first = sum(row["alpha_zero_side"] == "red" for row in selected)
        second = sum(row["alpha_zero_side"] == "yellow" for row in selected)
        expected = args.games_per_matchup // 2
        if first != expected or second != expected:
            raise RuntimeError(
                f"invalid side allocation for {opponent_name}: {first}/{second}"
            )
    config["validation"] = {
        "complete": True,
        "all_moves_legal": True,
        "invalid_moves": 0,
        "games": len(final_games),
        "allocation_per_matchup": {
            "first": args.games_per_matchup // 2,
            "second": args.games_per_matchup // 2,
        },
    }
    args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config["validation"], indent=2))


if __name__ == "__main__":
    main()
