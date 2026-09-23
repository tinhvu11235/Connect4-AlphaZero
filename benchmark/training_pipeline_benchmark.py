"""Short, reproducible before/after benchmark for the AlphaZero trainer.

The legacy arm uses sequential self-play and one update per 20 games.  The
optimized arm uses the persistent CPU worker/single-GPU inference service and
the configured number of replay updates.  Both use the same network, seed,
game count, opponent strength, and FP32 loss.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_ai.models.policy_net import PolicyGradientNet  # noqa: E402
from connect4_ai.training.parallel_self_play import (  # noqa: E402
    ParallelSelfPlayConfig,
    ParallelSelfPlayPool,
    optimize_trajectories,
)
from connect4_ai.training.train_alphazero import (  # noqa: E402
    AlphaZeroTrainer,
    initialize_like_keras,
    seed_everything,
)
from connect4_ai.utils.device import resolve_device  # noqa: E402
from connect4_ai.utils.monitoring import LiveStatus, PerformanceMonitor  # noqa: E402


def _trainer(seed: int, rollouts: int) -> AlphaZeroTrainer:
    seed_everything(seed)
    model = PolicyGradientNet()
    initialize_like_keras(model)
    return AlphaZeroTrainer(
        model,
        device=resolve_device(allow_cpu=False),
        seed=seed,
        opponent_weight=0.3,
        opponent_rollouts=rollouts,
        max_steps=50,
    )


def _result(
    *,
    games: int,
    positions_generated: int,
    positions_trained: int,
    updates: int,
    elapsed: float,
    inference_requests: int,
    inference_batch_items: float,
    inference_batches: int,
    monitor: dict[str, Any],
) -> dict[str, Any]:
    return {
        "games": games,
        "positions_generated": positions_generated,
        "positions_trained": positions_trained,
        "optimizer_steps": updates,
        "elapsed_seconds": elapsed,
        "games_per_second": games / elapsed,
        "positions_generated_per_second": positions_generated / elapsed,
        "positions_trained_per_second": positions_trained / elapsed,
        "optimizer_steps_per_second": updates / elapsed,
        "inference_requests": inference_requests,
        "average_inference_batch_size": (
            inference_batch_items / max(inference_batches, 1)
        ),
        "monitor": monitor,
    }


def run_legacy(args: argparse.Namespace) -> dict[str, Any]:
    trainer = _trainer(args.seed, args.rollouts)
    status = LiveStatus()
    status.update("legacy_sequential")
    positions_generated = 0
    positions_trained = 0
    updates = 0
    inference_requests = 0
    started = perf_counter()
    with PerformanceMonitor(
        args.gpu_metrics, status, interval_seconds=0.25, run_label="audit_benchmark"
    ) as monitor:
        for batch_index in range(args.games // args.batch_games):
            side = "red" if batch_index % 2 == 0 else "yellow"
            trajectories = [
                trainer.collect_episode(learner_side=side)
                for _ in range(args.batch_games)
            ]
            positions_generated += sum(item["policy_steps"] for item in trajectories)
            inference_requests += sum(
                int(item["profile"]["nn_requests"]) for item in trajectories
            )
            update = optimize_trajectories(
                trainer.model, trainer.optimizer, trajectories, device=trainer.device
            )
            positions_trained += int(update["samples"])
            updates += 1
            elapsed = perf_counter() - started
            status.update(
                "legacy_sequential",
                games_per_sec=(batch_index + 1) * args.batch_games / elapsed,
                positions_per_sec=positions_generated / elapsed,
                training_steps_per_sec=positions_trained / elapsed,
            )
        elapsed = perf_counter() - started
    return _result(
        games=args.games,
        positions_generated=positions_generated,
        positions_trained=positions_trained,
        updates=updates,
        elapsed=elapsed,
        inference_requests=inference_requests,
        inference_batch_items=float(inference_requests),
        inference_batches=inference_requests,
        monitor=monitor.summary(),
    )


def run_optimized(args: argparse.Namespace) -> dict[str, Any]:
    trainer = _trainer(args.seed, args.rollouts)
    status = LiveStatus()
    status.update("optimized_parallel_replay")
    replay: deque[dict[str, Any]] = deque(maxlen=args.replay_buffer_games)
    positions_generated = 0
    positions_trained = 0
    updates = 0
    inference_requests = 0
    inference_batch_items = 0.0
    inference_batches = 0
    started = perf_counter()
    with PerformanceMonitor(
        args.gpu_metrics, status, interval_seconds=0.25, run_label="audit_benchmark"
    ) as monitor:
        with ParallelSelfPlayPool(
            trainer.model,
            trainer.opponent_model,
            config=ParallelSelfPlayConfig(
                workers=args.workers,
                inference_batch_size=args.inference_batch_size,
                opponent_weight=0.3,
                opponent_rollouts=args.rollouts,
                pin_memory=True,
            ),
            device=trainer.device,
        ) as pool:
            for batch_index in range(args.games // args.batch_games):
                side = "red" if batch_index % 2 == 0 else "yellow"
                trajectories, generation = pool.generate(
                    args.batch_games,
                    seed=args.seed + batch_index * args.batch_games,
                    learner_side=side,
                )
                replay.extend(trajectories)
                positions_generated += int(generation["policy_steps"])
                inference_requests += int(generation["inference_requests"])
                inference_batches += int(generation["model_inference_batches"])
                inference_batch_items += (
                    float(generation["average_model_batch_size"])
                    * int(generation["model_inference_batches"])
                )
                for update_index in range(args.updates_per_batch):
                    selected = (
                        trajectories
                        if update_index == 0
                        else random.sample(
                            list(replay), min(args.batch_games, len(replay))
                        )
                    )
                    update = optimize_trajectories(
                        trainer.model,
                        trainer.optimizer,
                        selected,
                        device=trainer.device,
                    )
                    positions_trained += int(update["samples"])
                    updates += 1
                elapsed = perf_counter() - started
                status.update(
                    "optimized_parallel_replay",
                    games_per_sec=(batch_index + 1) * args.batch_games / elapsed,
                    positions_per_sec=positions_generated / elapsed,
                    training_steps_per_sec=positions_trained / elapsed,
                )
        elapsed = perf_counter() - started
    return _result(
        games=args.games,
        positions_generated=positions_generated,
        positions_trained=positions_trained,
        updates=updates,
        elapsed=elapsed,
        inference_requests=inference_requests,
        inference_batch_items=inference_batch_items,
        inference_batches=inference_batches,
        monitor=monitor.summary(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--batch-games", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=2)
    parser.add_argument("--updates-per-batch", type=int, default=2)
    parser.add_argument("--replay-buffer-games", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=Path("results/audit_before_after.json")
    )
    parser.add_argument(
        "--gpu-metrics",
        type=Path,
        default=Path("results/audit_benchmark_gpu_metrics.csv"),
    )
    args = parser.parse_args()
    if args.games < 1 or args.games % args.batch_games != 0:
        parser.error("games must be a positive multiple of batch-games")
    baseline = run_legacy(args)
    optimized = run_optimized(args)
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "workload": vars(args) | {"output": str(args.output), "gpu_metrics": str(args.gpu_metrics)},
        "before_legacy_sequential_one_update": baseline,
        "after_parallel_replay": optimized,
        "speedup": {
            "games_per_second": optimized["games_per_second"]
            / baseline["games_per_second"],
            "positions_generated_per_second": optimized[
                "positions_generated_per_second"
            ]
            / baseline["positions_generated_per_second"],
            "positions_trained_per_second": optimized["positions_trained_per_second"]
            / baseline["positions_trained_per_second"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
