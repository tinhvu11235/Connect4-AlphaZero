"""Resumable, monitored execution of the Chapter 21 AlphaZero schedule."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
import csv
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connect4_ai.utils.monitoring import LiveStatus, PerformanceMonitor  # noqa: E402
from connect4_ai.models.policy_net import PolicyGradientNet  # noqa: E402
from connect4_ai.training.parallel_self_play import (  # noqa: E402
    ParallelSelfPlayConfig,
    ParallelSelfPlayPool,
    optimize_trajectories,
)
from connect4_ai.training.arena_metrics import (  # noqa: E402
    add_arena_scalars,
    promotion_threshold_met,
    summarize_arena_results,
)
from connect4_ai.training.train_alphazero import (  # noqa: E402
    AlphaZeroTrainer,
    assert_finite_module,
    assert_finite_optimizer,
    initialize_like_keras,
    save_training_checkpoint,
    seed_everything,
)
from connect4_ai.utils.device import resolve_device  # noqa: E402


BASE_SCHEDULE = [(0.05, 50), (0.3, 100), (0.5, 150), (0.7, 200), (0.9, 250)]
PROFILE_DEFAULTS = {
    "debug": {
        "max_iterations": 1,
        "min_games_per_iteration": 40,
        "self_play_games_per_iteration": 40,
        "replay_buffer_games": 80,
        "training_updates_per_batch": 1,
        "evaluation_every_batches": 0,
        "evaluation_games": 0,
    },
    "quick_test": {
        "max_iterations": 2,
        "min_games_per_iteration": 200,
        "self_play_games_per_iteration": 400,
        "replay_buffer_games": 400,
        "training_updates_per_batch": 1,
        "evaluation_every_batches": 10,
        "evaluation_games": 20,
    },
    "full_train": {
        "max_iterations": 20,
        "min_games_per_iteration": 5000,
        "self_play_games_per_iteration": 25000,
        "replay_buffer_games": 5000,
        "training_updates_per_batch": 2,
        "evaluation_every_batches": 50,
        "evaluation_games": 40,
    },
}
ITERATION_FIELDS = [
    "iteration",
    "games_generated",
    "training_samples",
    "epochs",
    "optimizer_steps",
    "positions_trained",
    "training_updates_per_position",
    "replay_buffer_games",
    "learning_rate",
    "loss",
    "policy_loss",
    "value_loss",
    "self_play_duration",
    "training_duration",
    "evaluation_duration",
    "data_prep_duration",
    "h2d_duration",
    "forward_duration",
    "backward_duration",
    "optimizer_duration",
    "mcts_duration",
    "nn_inference_duration",
    "save_duration",
    "gpu_mean_utilization",
    "gpu_peak_vram_mb",
    "cpu_mean_utilization",
    "games_per_sec",
    "positions_per_sec",
    "optimizer_steps_per_sec",
    "average_inference_batch_size",
    "opponent_weight",
    "opponent_rollouts",
    "running_reward",
    "evaluation_win_rate",
    "arena_score",
    "promoted",
    "stop_reason",
    "checkpoint",
]


def _schedule_entry(iteration: int) -> tuple[float, int]:
    """Return the source curriculum entry, repeating its final stage thereafter."""

    return BASE_SCHEDULE[min(iteration, len(BASE_SCHEDULE) - 1)]


def _atomic_torch_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def _rng_state(trainer: AlphaZeroTrainer) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "sample_generator": trainer.sample_generator.get_state(),
        "opponent_random": trainer.opponent._rng.getstate(),
        "opponent_seed_counter": trainer._opponent_seed,
    }


def _restore_rng_state(trainer: AlphaZeroTrainer, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    trainer.sample_generator.set_state(state["sample_generator"].cpu())
    trainer.opponent._rng.setstate(state["opponent_random"])
    trainer._opponent_seed = int(state["opponent_seed_counter"])


def _state_payload(
    trainer: AlphaZeroTrainer,
    *,
    iteration: int,
    in_iteration: bool,
    episodes: int,
    batches: int,
    running_rewards: deque[float],
    history: list[dict[str, Any]],
    accumulators: dict[str, float],
    replay_buffer: deque[dict[str, Any]],
    total_games: int,
    total_positions: int,
    total_optimizer_steps: int,
    reference_optimizer_state_dict: dict[str, Any],
    config: dict[str, Any],
    complete: bool = False,
) -> dict[str, Any]:
    assert_finite_module(trainer.model, label="candidate")
    assert_finite_module(trainer.opponent_model, label="reference")
    assert_finite_optimizer(trainer.optimizer, label="candidate")
    return {
        "format_version": 2,
        "kind": "chapter21_resumable_training_state",
        "saved_at": datetime.now().astimezone().isoformat(),
        "complete": complete,
        "current_iteration": iteration,
        "in_iteration": in_iteration,
        "episodes": episodes,
        "batches": batches,
        "running_rewards": list(running_rewards),
        "iteration_history": history,
        "accumulators": accumulators,
        "replay_buffer": list(replay_buffer),
        "total_games": total_games,
        "total_positions": total_positions,
        "total_optimizer_steps": total_optimizer_steps,
        "model_state_dict": trainer.model.state_dict(),
        "opponent_model_state_dict": trainer.opponent_model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "reference_optimizer_state_dict": reference_optimizer_state_dict,
        "rng_state": _rng_state(trainer),
        "config": config,
    }


def _write_iteration_row(output: Path, row: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    fields = ITERATION_FIELDS
    if not write_header:
        with output.open(newline="", encoding="utf-8") as stream:
            existing = next(csv.reader(stream), [])
        # Preserve append compatibility with the already-produced five-stage
        # CSV.  New runs receive the richer schema; detailed metrics always go
        # to the JSONL profile stream.
        if existing:
            fields = existing
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, default=str) + "\n")


def _empty_accumulators() -> dict[str, float]:
    return {
        "samples": 0.0,
        "positions_trained": 0.0,
        "optimizer_steps": 0.0,
        "loss_sum": 0.0,
        "self_play_duration": 0.0,
        "training_duration": 0.0,
        "evaluation_duration": 0.0,
        "data_prep_duration": 0.0,
        "h2d_duration": 0.0,
        "forward_duration": 0.0,
        "backward_duration": 0.0,
        "optimizer_duration": 0.0,
        "mcts_duration": 0.0,
        "nn_inference_duration": 0.0,
        "save_duration": 0.0,
        "inference_requests": 0.0,
        "inference_batches": 0.0,
        "inference_batch_items": 0.0,
    }


def _monitor_summary(
    metrics_path: Path, iteration: int
) -> tuple[float, float, float]:
    phase = f"full_training:iteration_{iteration}"
    if not metrics_path.exists():
        return float("nan"), float("nan"), float("nan")
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    selected = [record for record in records if record["phase"] == phase]
    if not selected:
        return float("nan"), float("nan"), float("nan")
    gpu = [float(record["gpu_util_percent"]) for record in selected]
    vram = [float(record["gpu_memory_used_mb"]) for record in selected]
    cpu = [float(record["cpu_percent"]) for record in selected]
    return sum(gpu) / len(gpu), max(vram), sum(cpu) / len(cpu)


def _iteration_config(iteration: int, args: argparse.Namespace) -> dict[str, Any]:
    weight, rollouts = _schedule_entry(iteration)
    return {
        "validation_only": False,
        "batch_size": 20,
        "gamma": 0.95,
        "max_steps": 50,
        "opponent_weight": weight,
        "opponent_rollouts": rollouts,
        "learning_rate": 0.00025,
        "adam_epsilon": 1e-7,
        "global_clipnorm": 1.0,
        "target_mean_reward": args.target_mean_reward,
        "promote_on_cap": args.promote_on_cap,
        "min_self_play_games": args.min_games_per_iteration,
        "max_self_play_games": args.self_play_games_per_iteration,
        "replay_buffer_games": args.replay_buffer_games,
        "training_updates_per_batch": args.training_updates_per_batch,
        "replay_sample_games": args.replay_sample_games,
        "evaluation_every_batches": args.evaluation_every_batches,
        "evaluation_games": args.evaluation_games,
        "parallel_workers": args.parallel_workers,
        "parallel_inference_batch_size": args.parallel_inference_batch_size,
        "parallel_min_rollouts": args.parallel_min_rollouts,
        "parallel_pin_memory": args.parallel_pin_memory,
    }


def _save_progress(
    path: Path,
    trainer: AlphaZeroTrainer,
    *,
    iteration: int,
    in_iteration: bool,
    episodes: int,
    batches: int,
    running_rewards: deque[float],
    history: list[dict[str, Any]],
    accumulators: dict[str, float],
    replay_buffer: deque[dict[str, Any]],
    total_games: int,
    total_positions: int,
    total_optimizer_steps: int,
    reference_optimizer_state_dict: dict[str, Any],
    config: dict[str, Any],
    complete: bool = False,
) -> None:
    _atomic_torch_save(
        _state_payload(
            trainer,
            iteration=iteration,
            in_iteration=in_iteration,
            episodes=episodes,
            batches=batches,
            running_rewards=running_rewards,
            history=history,
            accumulators=accumulators,
            replay_buffer=replay_buffer,
            total_games=total_games,
            total_positions=total_positions,
            total_optimizer_steps=total_optimizer_steps,
            reference_optimizer_state_dict=reference_optimizer_state_dict,
            config=config,
            complete=complete,
        ),
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(allow_cpu=False)
    model = PolicyGradientNet()
    initialize_like_keras(model)
    trainer = AlphaZeroTrainer(
        model,
        device=device,
        seed=args.seed,
        opponent_weight=BASE_SCHEDULE[0][0],
        opponent_rollouts=BASE_SCHEDULE[0][1],
        max_steps=50,
    )
    config = {
        "seed": args.seed,
        "device": str(device),
        "profile": args.profile,
        "max_iterations": args.max_iterations,
        "schedule": [
            {
                "iteration": i,
                "weight": _schedule_entry(i)[0],
                "rollouts": _schedule_entry(i)[1],
            }
            for i in range(args.max_iterations)
        ],
        "limits": {
            "min_games_per_iteration": args.min_games_per_iteration,
            "self_play_games_per_iteration": args.self_play_games_per_iteration,
            "max_total_games": args.max_total_games,
            "max_optimizer_steps": args.max_optimizer_steps,
            "target_mean_reward": args.target_mean_reward,
            "promote_on_cap": args.promote_on_cap,
        },
        "replay": {
            "buffer_games": args.replay_buffer_games,
            "sample_games": args.replay_sample_games,
            "training_updates_per_batch": args.training_updates_per_batch,
        },
        "evaluation": {
            "every_batches": args.evaluation_every_batches,
            "games": args.evaluation_games,
        },
        "checkpoint_every_batches": args.checkpoint_every_batches,
        "tensorboard_logdir": str(args.tensorboard_logdir),
        "production_configuration": {
            "parallel_workers": args.parallel_workers,
            "parallel_inference_batch_size": args.parallel_inference_batch_size,
            "parallel_min_rollouts": args.parallel_min_rollouts,
            "parallel_pin_memory": args.parallel_pin_memory,
            "float_precision": "float32",
            "amp": False,
            "torch_compile": False,
        },
        "source": "ch21AlphaZeroUnsolvedGames.ipynb",
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    resume: dict[str, Any] | None = None
    if args.resume and args.state_checkpoint.exists():
        resume = torch.load(
            args.state_checkpoint, map_location=device, weights_only=False
        )
        if resume.get("complete") and int(resume["current_iteration"]) >= args.max_iterations:
            return {
                "status": "already_complete",
                "final_checkpoint": str(args.final_checkpoint),
                "state_checkpoint": str(args.state_checkpoint),
            }
        trainer.model.load_state_dict(resume["model_state_dict"])
        trainer.optimizer.load_state_dict(resume["optimizer_state_dict"])
        assert_finite_module(trainer.model, label="resumed candidate")
        assert_finite_optimizer(trainer.optimizer, label="resumed candidate")
        saved_seed = resume.get("config", {}).get("seed")
        if saved_seed is not None and int(saved_seed) != args.seed:
            raise ValueError(
                f"resume seed mismatch: checkpoint={saved_seed}, requested={args.seed}"
            )
        if not resume["in_iteration"]:
            _restore_rng_state(trainer, resume["rng_state"])

    current_iteration = int(resume["current_iteration"]) if resume else 0
    replay_buffer: deque[dict[str, Any]] = deque(
        resume.get("replay_buffer", []) if resume else [],
        maxlen=args.replay_buffer_games,
    )
    total_games = int(resume.get("total_games", 0)) if resume else 0
    total_positions = int(resume.get("total_positions", 0)) if resume else 0
    total_optimizer_steps = (
        int(resume.get("total_optimizer_steps", 0)) if resume else 0
    )
    reference_optimizer_state = deepcopy(trainer.optimizer.state_dict())
    if resume and "reference_optimizer_state_dict" in resume:
        reference_optimizer_state = deepcopy(
            resume["reference_optimizer_state_dict"]
        )
    elif resume and resume.get("in_iteration"):
        # Legacy state files did not preserve the optimizer at the generation
        # boundary.  Empty moments are safer than reusing rejected-candidate
        # moments if this generation later fails its promotion gate.
        reference_optimizer_state["state"] = {}
    if resume and not any((total_games, total_positions, total_optimizer_steps)):
        # Version-1 checkpoints did not store global counters.  Recover them
        # from the append-only iteration CSV when available.
        if args.iterations_output.exists():
            with args.iterations_output.open(newline="", encoding="utf-8") as stream:
                legacy_rows = list(csv.DictReader(stream))
            total_games = sum(int(row["games_generated"]) for row in legacy_rows)
            total_positions = sum(int(row["training_samples"]) for row in legacy_rows)
            total_optimizer_steps = sum(
                int(row.get("optimizer_steps") or row.get("epochs") or 0)
                for row in legacy_rows
            )
    process_batches = 0
    next_iteration = current_iteration
    status = LiveStatus()
    started_all = perf_counter()
    purge_step = current_iteration * 30_000 + (
        int(resume["episodes"])
        if resume and resume["in_iteration"]
        else 0
    )
    with ExitStack() as stack:
        writer = stack.enter_context(
            SummaryWriter(
                log_dir=str(args.tensorboard_logdir), purge_step=purge_step
            )
        )
        monitor = stack.enter_context(
            PerformanceMonitor(
                args.metrics_output,
                status,
                interval_seconds=1.0,
                run_label="full_training",
            )
        )
        writer.add_text(
            "configuration/json",
            f"```json\n{json.dumps(config, indent=2)}\n```",
            global_step=purge_step,
        )
        for iteration in range(current_iteration, args.max_iterations):
            weight, rollouts = _schedule_entry(iteration)
            resuming_iteration = bool(
                resume
                and iteration == current_iteration
                and resume["in_iteration"]
            )
            if resuming_iteration:
                trainer.set_opponent_snapshot(weight=weight, num_rollouts=rollouts)
                trainer.opponent_model.load_state_dict(
                    resume["opponent_model_state_dict"]
                )
                assert_finite_module(
                    trainer.opponent_model, label="resumed reference"
                )
                episodes = int(resume["episodes"])
                batches = int(resume["batches"])
                running_rewards = deque(resume["running_rewards"], maxlen=1000)
                history = list(resume["iteration_history"])
                accumulators = {
                    key: float(value)
                    for key, value in resume["accumulators"].items()
                }
                for key in (
                    "positions_trained",
                    "optimizer_steps",
                    "data_prep_duration",
                    "h2d_duration",
                    "forward_duration",
                    "backward_duration",
                    "optimizer_duration",
                    "mcts_duration",
                    "nn_inference_duration",
                    "save_duration",
                    "inference_requests",
                    "inference_batches",
                    "inference_batch_items",
                ):
                    accumulators.setdefault(key, 0.0)
                _restore_rng_state(trainer, resume["rng_state"])
                print(
                    f"resuming iteration {iteration} at {episodes} games / {batches} batches",
                    flush=True,
                )
            else:
                if iteration > 0:
                    trainer.set_opponent_snapshot(
                        weight=weight, num_rollouts=rollouts
                    )
                episodes = 0
                batches = 0
                running_rewards = deque(maxlen=1000)
                history: list[dict[str, Any]] = []
                accumulators = _empty_accumulators()
                reference_optimizer_state = deepcopy(
                    trainer.optimizer.state_dict()
                )
            resume = None
            status.update(f"iteration_{iteration}")
            use_parallel = rollouts >= args.parallel_min_rollouts
            pool: ParallelSelfPlayPool | None = None
            if use_parallel:
                pool_started = perf_counter()
                pool = ParallelSelfPlayPool(
                    trainer.model,
                    trainer.opponent_model,
                    config=ParallelSelfPlayConfig(
                        workers=args.parallel_workers,
                        inference_batch_size=args.parallel_inference_batch_size,
                        inference_wait_ms=1.0,
                        opponent_weight=weight,
                        opponent_rollouts=rollouts,
                        gamma=0.95,
                        max_steps=50,
                        pin_memory=args.parallel_pin_memory,
                    ),
                    device=device,
                )
                accumulators["self_play_duration"] += perf_counter() - pool_started
            stop_reason = ""
            evaluation_win_rate = float("nan")
            arena_summary: dict[str, Any] | None = None
            try:
                while True:
                    if (
                        args.max_optimizer_steps is not None
                        and total_optimizer_steps >= args.max_optimizer_steps
                    ):
                        stop_reason = "max_optimizer_steps"
                        break
                    side = "red" if batches % 2 == 0 else "yellow"
                    remaining_iteration_games = (
                        args.self_play_games_per_iteration - episodes
                    )
                    remaining_total_games = (
                        args.max_total_games - total_games
                        if args.max_total_games is not None
                        else 20
                    )
                    batch_games = min(20, remaining_iteration_games, remaining_total_games)
                    if batch_games <= 0:
                        stop_reason = (
                            "max_total_games"
                            if args.max_total_games is not None
                            and total_games >= args.max_total_games
                            else "self_play_games_per_iteration"
                        )
                        break
                    generation_started = perf_counter()
                    if pool is None:
                        trajectories = [
                            trainer.collect_episode(learner_side=side)
                            for _ in range(batch_games)
                        ]
                        generation = {
                            "games": batch_games,
                            "completed_games": sum(
                                int(item["completed"]) for item in trajectories
                            ),
                            "policy_steps": sum(
                                item["policy_steps"] for item in trajectories
                            ),
                            "game_moves": sum(
                                int(item["game_length"]) for item in trajectories
                            ),
                            "inference_requests": sum(
                                int(item["profile"]["nn_requests"])
                                for item in trajectories
                            ),
                            "inference_batches": sum(
                                int(item["profile"]["nn_requests"])
                                for item in trajectories
                            ),
                            "model_inference_batches": sum(
                                int(item["profile"]["nn_requests"])
                                for item in trajectories
                            ),
                            "average_model_batch_size": 1.0,
                            "mcts_seconds": sum(
                                float(item["profile"]["mcts_seconds"])
                                for item in trajectories
                            ),
                            "mcts_rollouts": sum(
                                int(item["profile"]["mcts_rollouts"])
                                for item in trajectories
                            ),
                            "worker_inference_wait_seconds": sum(
                                float(item["profile"]["nn_wait_seconds"])
                                for item in trajectories
                            ),
                            "data_prep_seconds": 0.0,
                            "h2d_seconds": 0.0,
                            "forward_seconds": sum(
                                float(item["profile"]["nn_wait_seconds"])
                                for item in trajectories
                            ),
                        }
                    else:
                        trajectories, generation = pool.generate(
                            batch_games,
                            seed=args.seed + iteration * 1_000_000 + episodes,
                            learner_side=side,
                        )
                        generation["game_moves"] = sum(
                            int(item["game_length"]) for item in trajectories
                        )
                    batch_self_play_duration = perf_counter() - generation_started
                    accumulators["self_play_duration"] += batch_self_play_duration

                    for trajectory in trajectories:
                        trajectory["generation"] = iteration
                        replay_buffer.append(trajectory)
                    training_started = perf_counter()
                    updates: list[dict[str, float | int]] = []
                    for update_index in range(args.training_updates_per_batch):
                        if (
                            args.max_optimizer_steps is not None
                            and total_optimizer_steps >= args.max_optimizer_steps
                        ):
                            break
                        if update_index == 0:
                            update_trajectories = trajectories
                        else:
                            sample_count = min(
                                args.replay_sample_games, len(replay_buffer)
                            )
                            update_trajectories = random.sample(
                                list(replay_buffer), sample_count
                            )
                        updates.append(
                            optimize_trajectories(
                                trainer.model,
                                trainer.optimizer,
                                update_trajectories,
                                device=device,
                            )
                        )
                        total_optimizer_steps += 1
                    batch_training_duration = perf_counter() - training_started
                    accumulators["training_duration"] += batch_training_duration
                    episode_rewards = [
                        float(item["episode_reward"]) for item in trajectories
                    ]
                    episodes += batch_games
                    total_games += batch_games
                    generated_positions = int(generation["policy_steps"])
                    total_positions += generated_positions
                    batches += 1
                    process_batches += 1
                    accumulators["samples"] += generated_positions
                    accumulators["mcts_duration"] += float(generation["mcts_seconds"])
                    accumulators["nn_inference_duration"] += float(
                        generation["forward_seconds"]
                    )
                    accumulators["data_prep_duration"] += float(
                        generation["data_prep_seconds"]
                    )
                    accumulators["h2d_duration"] += float(
                        generation["h2d_seconds"]
                    )
                    accumulators["forward_duration"] += float(
                        generation["forward_seconds"]
                    )
                    accumulators["inference_requests"] += int(
                        generation["inference_requests"]
                    )
                    accumulators["inference_batches"] += int(
                        generation["model_inference_batches"]
                    )
                    accumulators["inference_batch_items"] += (
                        float(generation["average_model_batch_size"])
                        * int(generation["model_inference_batches"])
                    )
                    for update in updates:
                        accumulators["loss_sum"] += float(update["loss"])
                        accumulators["positions_trained"] += int(update["samples"])
                        accumulators["optimizer_steps"] += 1
                        accumulators["data_prep_duration"] += float(
                            update["data_prep_seconds"]
                        )
                        accumulators["h2d_duration"] += float(update["h2d_seconds"])
                        accumulators["forward_duration"] += float(
                            update["forward_seconds"]
                        )
                        accumulators["backward_duration"] += float(
                            update["backward_seconds"]
                        )
                        accumulators["optimizer_duration"] += float(
                            update["optimizer_seconds"]
                        )

                    evaluation_started = perf_counter()
                    running_rewards.extend(episode_rewards)
                    running_reward = float(np.mean(running_rewards))
                    evaluation_win_rate = float("nan")
                    evaluation_mean_reward = float("nan")
                    if (
                        args.evaluation_every_batches > 0
                        and args.evaluation_games > 0
                        and batches % args.evaluation_every_batches == 0
                    ):
                        evaluation_results: list[tuple[str, float]] = []
                        red_games = args.evaluation_games // 2
                        for eval_side, eval_games in (
                            ("red", red_games),
                            ("yellow", args.evaluation_games - red_games),
                        ):
                            if pool is None:
                                evaluated, _ = (
                                    [
                                        trainer.collect_episode(learner_side=eval_side)
                                        for _ in range(eval_games)
                                    ],
                                    {},
                                )
                            else:
                                evaluated, _ = pool.generate(
                                    eval_games,
                                    seed=(
                                        args.seed
                                        + 900_000_000
                                        + iteration * 1_000_000
                                        + episodes
                                        + (0 if eval_side == "red" else red_games)
                                    ),
                                    learner_side=eval_side,
                                )
                            evaluation_results.extend(
                                (eval_side, float(item["episode_reward"]))
                                for item in evaluated
                            )
                        arena_summary = summarize_arena_results(evaluation_results)
                        evaluation_rewards = [
                            result for _, result in evaluation_results
                        ]
                        evaluation_mean_reward = float(np.mean(evaluation_rewards))
                        evaluation_win_rate = float(arena_summary["win_rate"])
                    if (
                        episodes >= args.min_games_per_iteration
                        and promotion_threshold_met(
                            games=len(running_rewards),
                            score=running_reward,
                            minimum_games=min(
                                1000, args.min_games_per_iteration
                            ),
                            threshold=args.target_mean_reward,
                        )
                    ):
                        stop_reason = "promotion_target"
                    elif episodes >= args.self_play_games_per_iteration:
                        stop_reason = "self_play_games_per_iteration"
                    elif (
                        args.max_total_games is not None
                        and total_games >= args.max_total_games
                    ):
                        stop_reason = "max_total_games"
                    elif (
                        args.max_optimizer_steps is not None
                        and total_optimizer_steps >= args.max_optimizer_steps
                    ):
                        stop_reason = "max_optimizer_steps"
                    batch_evaluation_duration = perf_counter() - evaluation_started
                    accumulators["evaluation_duration"] += batch_evaluation_duration
                    metric = {
                        "iteration": iteration,
                        "episodes": episodes,
                        "batch": batches,
                        "side": side,
                        "loss": float(updates[-1]["loss"]) if updates else float("nan"),
                        "policy_steps": int(generation["policy_steps"]),
                        "episode_rewards": episode_rewards,
                        "completed_games": int(generation["completed_games"]),
                        "running_reward": running_reward,
                        "evaluation_mean_reward": evaluation_mean_reward,
                        "evaluation_win_rate": evaluation_win_rate,
                        "optimizer_steps": len(updates),
                        "positions_trained": sum(
                            int(item["samples"]) for item in updates
                        ),
                        "replay_buffer_games": len(replay_buffer),
                        "opponent_weight": weight,
                        "opponent_rollouts": rollouts,
                    }
                    history.append(metric)
                    elapsed_iteration = (
                        accumulators["self_play_duration"]
                        + accumulators["training_duration"]
                        + accumulators["evaluation_duration"]
                    )
                    status.update(
                        f"iteration_{iteration}",
                        games_per_sec=episodes / elapsed_iteration,
                        positions_per_sec=accumulators["samples"] / elapsed_iteration,
                        training_steps_per_sec=(
                            accumulators["positions_trained"] / elapsed_iteration
                        ),
                    )
                    global_step = iteration * 30_000 + episodes
                    writer.add_scalar(
                        "batch/loss", float(updates[-1]["loss"]), global_step
                    )
                    writer.add_scalar(
                        "batch/policy_loss",
                        float(updates[-1]["loss"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "batch/running_reward", running_reward, global_step
                    )
                    writer.add_scalar(
                        "batch/mean_episode_reward",
                        float(np.mean(episode_rewards)),
                        global_step,
                    )
                    writer.add_scalar(
                        "batch/policy_steps",
                        int(generation["policy_steps"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "batch/gradient_norm_before_clip",
                        float(updates[-1]["gradient_norm_before_clip"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "batch/optimizer_steps", len(updates), global_step
                    )
                    writer.add_scalar(
                        "batch/replay_buffer_games", len(replay_buffer), global_step
                    )
                    # Stable namespaces keep training, self-play, arena, and
                    # pipeline telemetry separate.  Legacy tags above remain
                    # for compatibility with existing runs.
                    writer.add_scalar(
                        "train/total_loss", float(updates[-1]["loss"]), global_step
                    )
                    writer.add_scalar(
                        "train/policy_loss", float(updates[-1]["loss"]), global_step
                    )
                    writer.add_scalar(
                        "train/learning_rate",
                        float(trainer.optimizer.param_groups[0]["lr"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "selfplay/games_per_sec",
                        batch_games / max(batch_self_play_duration, 1e-12),
                        global_step,
                    )
                    writer.add_scalar(
                        "selfplay/positions_per_sec",
                        generated_positions / max(batch_self_play_duration, 1e-12),
                        global_step,
                    )
                    writer.add_scalar(
                        "selfplay/avg_game_length",
                        float(generation["game_moves"]) / batch_games,
                        global_step,
                    )
                    writer.add_scalar(
                        "system/replay_buffer_size", len(replay_buffer), global_step
                    )
                    writer.add_scalar(
                        "system/inference_batch_size",
                        float(generation["average_model_batch_size"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "system/inference_batches_per_sec",
                        float(generation["model_inference_batches"])
                        / max(batch_self_play_duration, 1e-12),
                        global_step,
                    )
                    writer.add_scalar(
                        "system/training_steps_per_sec",
                        len(updates) / max(batch_training_duration, 1e-12),
                        global_step,
                    )
                    if evaluation_win_rate == evaluation_win_rate:
                        writer.add_scalar(
                            "evaluation/win_rate", evaluation_win_rate, global_step
                        )
                        writer.add_scalar(
                            "evaluation/mean_reward",
                            evaluation_mean_reward,
                            global_step,
                        )
                        assert arena_summary is not None
                        add_arena_scalars(writer, arena_summary, global_step)
                    writer.add_scalar(
                        "throughput/games_per_sec",
                        episodes / elapsed_iteration,
                        global_step,
                    )
                    writer.add_scalar(
                        "throughput/training_samples_per_sec",
                        accumulators["positions_trained"] / elapsed_iteration,
                        global_step,
                    )
                    writer.add_scalar(
                        "throughput/optimizer_steps_per_sec",
                        accumulators["optimizer_steps"] / elapsed_iteration,
                        global_step,
                    )
                    writer.add_scalar(
                        "duration/self_play_batch_seconds",
                        batch_self_play_duration,
                        global_step,
                    )
                    writer.add_scalar(
                        "duration/training_batch_seconds",
                        batch_training_duration,
                        global_step,
                    )
                    writer.add_scalar(
                        "duration/evaluation_batch_seconds",
                        batch_evaluation_duration,
                        global_step,
                    )
                    for profile_key, profile_value in (
                        ("data_prep", sum(float(u["data_prep_seconds"]) for u in updates)),
                        ("h2d", sum(float(u["h2d_seconds"]) for u in updates)),
                        ("forward", sum(float(u["forward_seconds"]) for u in updates)),
                        ("backward", sum(float(u["backward_seconds"]) for u in updates)),
                        ("optimizer", sum(float(u["optimizer_seconds"]) for u in updates)),
                        ("mcts_cpu", float(generation["mcts_seconds"])),
                        ("nn_inference", float(generation["forward_seconds"])),
                    ):
                        writer.add_scalar(
                            f"profile/{profile_key}_batch_seconds",
                            profile_value,
                            global_step,
                        )
                    profiled_wall = (
                        batch_self_play_duration
                        + batch_training_duration
                        + batch_evaluation_duration
                    )
                    batch_profile = {
                        "event": "training_batch",
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "iteration": iteration,
                        "batch": batches,
                        "total_games": total_games,
                        "total_positions": total_positions,
                        "total_optimizer_steps": total_optimizer_steps,
                        "buffer_games": len(replay_buffer),
                        "batch_size_games": batch_games,
                        "positions_generated": generated_positions,
                        "positions_trained": sum(int(u["samples"]) for u in updates),
                        "policy_loss": float(updates[-1]["loss"]),
                        "value_loss": None,
                        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                        "average_inference_batch_size": float(
                            generation["average_model_batch_size"]
                        ),
                        "inference_requests": int(generation["inference_requests"]),
                        "games_per_second": batch_games
                        / max(profiled_wall, 1e-12),
                        "positions_per_second": generated_positions
                        / max(profiled_wall, 1e-12),
                        "optimizer_steps_per_second": len(updates)
                        / max(profiled_wall, 1e-12),
                        "mcts_rollouts_per_second": int(generation["mcts_rollouts"])
                        / max(batch_self_play_duration, 1e-12),
                        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device)
                        / (1024**2),
                        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device)
                        / (1024**2),
                        "seconds": {
                            "self_play": batch_self_play_duration,
                            "mcts_cpu_aggregate": float(generation["mcts_seconds"]),
                            "nn_inference": float(generation["forward_seconds"]),
                            "data_prep": sum(
                                float(u["data_prep_seconds"]) for u in updates
                            ),
                            "h2d": sum(float(u["h2d_seconds"]) for u in updates),
                            "forward": sum(
                                float(u["forward_seconds"]) for u in updates
                            ),
                            "backward": sum(
                                float(u["backward_seconds"]) for u in updates
                            ),
                            "optimizer": sum(
                                float(u["optimizer_seconds"]) for u in updates
                            ),
                            "training_total": batch_training_duration,
                            "evaluation": batch_evaluation_duration,
                        },
                        "percent_of_profiled_wall": {
                            "self_play": 100.0
                            * batch_self_play_duration
                            / max(profiled_wall, 1e-12),
                            "training": 100.0
                            * batch_training_duration
                            / max(profiled_wall, 1e-12),
                            "evaluation": 100.0
                            * batch_evaluation_duration
                            / max(profiled_wall, 1e-12),
                        },
                    }
                    _append_jsonl(args.profile_output, batch_profile)
                    if monitor.records:
                        latest_resource = monitor.records[-1]
                        for source, tag in (
                            ("gpu_util_percent", "resource/gpu_util_percent"),
                            ("gpu_memory_used_mb", "resource/gpu_memory_used_mb"),
                            ("cpu_percent", "resource/cpu_util_percent"),
                            ("ram_used_gb", "resource/ram_used_gb"),
                        ):
                            writer.add_scalar(
                                tag, float(latest_resource[source]), global_step
                            )
                    if batches % args.progress_every_batches == 0 or stop_reason:
                        self_play_share = 100.0 * batch_self_play_duration / max(
                            batch_self_play_duration
                            + batch_training_duration
                            + batch_evaluation_duration,
                            1e-12,
                        )
                        print(
                            f"iteration={iteration} games={episodes} batches={batches} "
                            f"running_reward={running_reward:.4f} "
                            f"games_per_sec={episodes / elapsed_iteration:.2f} "
                            f"self_play={batch_self_play_duration:.3f}s({self_play_share:.1f}%) "
                            f"train={batch_training_duration:.3f}s "
                            f"eval={batch_evaluation_duration:.3f}s "
                            f"avg_infer_batch={generation['average_model_batch_size']:.2f}",
                            flush=True,
                        )
                    if (
                        batches % args.checkpoint_every_batches == 0
                        or stop_reason
                        or (
                            args.stop_after_batches is not None
                            and process_batches >= args.stop_after_batches
                        )
                    ):
                        save_started = perf_counter()
                        _save_progress(
                            args.state_checkpoint,
                            trainer,
                            iteration=iteration,
                            in_iteration=True,
                            episodes=episodes,
                            batches=batches,
                            running_rewards=running_rewards,
                            history=history,
                            accumulators=accumulators,
                            replay_buffer=replay_buffer,
                            total_games=total_games,
                            total_positions=total_positions,
                            total_optimizer_steps=total_optimizer_steps,
                            reference_optimizer_state_dict=reference_optimizer_state,
                            config=config,
                        )
                        save_seconds = perf_counter() - save_started
                        accumulators["save_duration"] += save_seconds
                        _append_jsonl(
                            args.profile_output,
                            {
                                "event": "checkpoint",
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "iteration": iteration,
                                "batch": batches,
                                "save_seconds": save_seconds,
                                "path": str(args.state_checkpoint),
                            },
                        )
                        writer.flush()
                    if (
                        args.stop_after_batches is not None
                        and process_batches >= args.stop_after_batches
                    ):
                        return {
                            "status": "interrupted_for_resume_validation",
                            "iteration": iteration,
                            "episodes": episodes,
                            "state_checkpoint": str(args.state_checkpoint),
                        }
                    if stop_reason:
                        break
            finally:
                if pool is not None:
                    close_started = perf_counter()
                    pool.close()
                    accumulators["self_play_duration"] += (
                        perf_counter() - close_started
                    )

            promoted = stop_reason == "promotion_target" or (
                args.promote_on_cap
                and stop_reason == "self_play_games_per_iteration"
            )
            suffix = "" if promoted else "_rejected"
            checkpoint_path = (
                args.output_dir / f"connzero_iter_{iteration:03d}{suffix}.pt"
            )
            iteration_config = _iteration_config(iteration, args)
            iteration_config.update(
                {
                    "promoted": promoted,
                    "promotion_score": float(np.mean(running_rewards)),
                    "promotion_window_games": len(running_rewards),
                    "promotion_reason": stop_reason,
                }
            )
            save_started = perf_counter()
            save_training_checkpoint(
                checkpoint_path,
                trainer,
                iteration=iteration,
                config=iteration_config,
                history=history,
            )
            accumulators["save_duration"] += perf_counter() - save_started
            gpu_mean, peak_vram, cpu_mean = _monitor_summary(
                args.metrics_output, iteration
            )
            total_duration = (
                accumulators["self_play_duration"]
                + accumulators["training_duration"]
                + accumulators["evaluation_duration"]
            )
            row = {
                "iteration": iteration,
                "games_generated": episodes,
                "training_samples": int(accumulators["samples"]),
                # Kept only for compatibility with the existing CSV header.
                # This project has no dataset epoch; the value is optimizer steps.
                "epochs": int(accumulators["optimizer_steps"]),
                "optimizer_steps": int(accumulators["optimizer_steps"]),
                "positions_trained": int(accumulators["positions_trained"]),
                "training_updates_per_position": (
                    accumulators["positions_trained"]
                    / max(accumulators["samples"], 1.0)
                ),
                "replay_buffer_games": len(replay_buffer),
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "loss": accumulators["loss_sum"]
                / max(accumulators["optimizer_steps"], 1.0),
                "policy_loss": accumulators["loss_sum"]
                / max(accumulators["optimizer_steps"], 1.0),
                "value_loss": "N/A",
                "self_play_duration": accumulators["self_play_duration"],
                "training_duration": accumulators["training_duration"],
                "evaluation_duration": accumulators["evaluation_duration"],
                "data_prep_duration": accumulators["data_prep_duration"],
                "h2d_duration": accumulators["h2d_duration"],
                "forward_duration": accumulators["forward_duration"],
                "backward_duration": accumulators["backward_duration"],
                "optimizer_duration": accumulators["optimizer_duration"],
                "mcts_duration": accumulators["mcts_duration"],
                "nn_inference_duration": accumulators["nn_inference_duration"],
                "save_duration": accumulators["save_duration"],
                "gpu_mean_utilization": gpu_mean,
                "gpu_peak_vram_mb": peak_vram,
                "cpu_mean_utilization": cpu_mean,
                "games_per_sec": episodes / total_duration,
                "positions_per_sec": accumulators["samples"] / total_duration,
                "optimizer_steps_per_sec": (
                    accumulators["optimizer_steps"] / total_duration
                ),
                "average_inference_batch_size": (
                    accumulators["inference_batch_items"]
                    / max(accumulators["inference_batches"], 1.0)
                ),
                "opponent_weight": weight,
                "opponent_rollouts": rollouts,
                "running_reward": float(np.mean(running_rewards)),
                "evaluation_win_rate": evaluation_win_rate,
                "arena_score": (
                    float(arena_summary["score"])
                    if arena_summary is not None
                    else float("nan")
                ),
                "promoted": int(promoted),
                "stop_reason": stop_reason,
                "checkpoint": str(checkpoint_path),
            }
            _write_iteration_row(args.iterations_output, row)
            for key in (
                "games_generated",
                "training_samples",
                "epochs",
                "optimizer_steps",
                "positions_trained",
                "training_updates_per_position",
                "replay_buffer_games",
                "loss",
                "self_play_duration",
                "training_duration",
                "evaluation_duration",
                "data_prep_duration",
                "h2d_duration",
                "forward_duration",
                "backward_duration",
                "optimizer_duration",
                "mcts_duration",
                "nn_inference_duration",
                "save_duration",
                "gpu_mean_utilization",
                "gpu_peak_vram_mb",
                "cpu_mean_utilization",
                "games_per_sec",
                "positions_per_sec",
                "optimizer_steps_per_sec",
                "average_inference_batch_size",
                "running_reward",
                "evaluation_win_rate",
                "arena_score",
                "promoted",
            ):
                writer.add_scalar(f"iteration/{key}", float(row[key]), iteration)
            generation_global_step = iteration * 30_000 + episodes
            if arena_summary is not None:
                add_arena_scalars(
                    writer,
                    arena_summary,
                    generation_global_step,
                    promoted=promoted,
                )
            else:
                writer.add_scalar(
                    "arena/promoted", float(promoted), generation_global_step
                )
            writer.add_text(
                "iteration/checkpoint",
                str(checkpoint_path),
                global_step=iteration,
            )
            writer.flush()
            if not promoted:
                # Keep the rejected candidate for diagnosis, but restore the
                # frozen reference before writing resumable state.  Samples
                # produced by the rejected generation must not leak forward.
                trainer.model.load_state_dict(trainer.opponent_model.state_dict())
                trainer.optimizer.load_state_dict(reference_optimizer_state)
                replay_buffer = deque(
                    (
                        item
                        for item in replay_buffer
                        if item.get("generation") != iteration
                    ),
                    maxlen=args.replay_buffer_games,
                )
                _save_progress(
                    args.state_checkpoint,
                    trainer,
                    iteration=iteration,
                    in_iteration=False,
                    episodes=0,
                    batches=0,
                    running_rewards=deque(maxlen=1000),
                    history=[],
                    accumulators=_empty_accumulators(),
                    replay_buffer=replay_buffer,
                    total_games=total_games,
                    total_positions=total_positions,
                    total_optimizer_steps=total_optimizer_steps,
                    reference_optimizer_state_dict=reference_optimizer_state,
                    config=config,
                    complete=False,
                )
                status.update("promotion_rejected")
                writer.add_text(
                    "training/status",
                    "promotion_rejected",
                    global_step=generation_global_step,
                )
                writer.flush()
                return {
                    "status": "promotion_rejected",
                    "iteration": iteration,
                    "score": float(np.mean(running_rewards)),
                    "candidate_checkpoint": str(checkpoint_path),
                    "state_checkpoint": str(args.state_checkpoint),
                    "total_games": total_games,
                    "total_positions": total_positions,
                    "total_optimizer_steps": total_optimizer_steps,
                }
            _save_progress(
                args.state_checkpoint,
                trainer,
                iteration=iteration + 1,
                in_iteration=False,
                episodes=0,
                batches=0,
                running_rewards=deque(maxlen=1000),
                history=[],
                accumulators=_empty_accumulators(),
                replay_buffer=replay_buffer,
                total_games=total_games,
                total_positions=total_positions,
                total_optimizer_steps=total_optimizer_steps,
                reference_optimizer_state_dict=reference_optimizer_state,
                config=config,
                complete=False,
            )
            next_iteration = iteration + 1
            if stop_reason in ("max_total_games", "max_optimizer_steps"):
                break

        save_training_checkpoint(
            args.final_checkpoint,
            trainer,
            iteration=next_iteration - 1,
            config={**config, "completed": True},
            history=[],
        )
        _save_progress(
            args.state_checkpoint,
            trainer,
            iteration=next_iteration,
            in_iteration=False,
            episodes=0,
            batches=0,
            running_rewards=deque(maxlen=1000),
            history=[],
            accumulators=_empty_accumulators(),
            replay_buffer=replay_buffer,
            total_games=total_games,
            total_positions=total_positions,
            total_optimizer_steps=total_optimizer_steps,
            reference_optimizer_state_dict=reference_optimizer_state,
            config=config,
            complete=True,
        )
        status.update("complete")
        writer.add_text(
            "training/status", "complete", global_step=next_iteration
        )
        writer.flush()
    return {
        "status": "complete",
        "elapsed_seconds": perf_counter() - started_all,
        "final_checkpoint": str(args.final_checkpoint),
        "state_checkpoint": str(args.state_checkpoint),
        "iterations_completed": next_iteration,
        "total_games": total_games,
        "total_positions": total_positions,
        "total_optimizer_steps": total_optimizer_steps,
        "monitor": monitor.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="full_train",
        help="Named duration/update profile; explicit limit flags override it.",
    )
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--min-games-per-iteration", type=int)
    parser.add_argument("--self-play-games-per-iteration", type=int)
    parser.add_argument("--max-total-games", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--target-mean-reward", type=float, default=0.1)
    parser.add_argument(
        "--promote-on-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow an unevaluated/under-threshold candidate to advance at the "
            "per-generation game cap. Disabled by default for checkpoint safety."
        ),
    )
    parser.add_argument("--replay-buffer-games", type=int)
    parser.add_argument("--replay-sample-games", type=int, default=20)
    parser.add_argument("--training-updates-per-batch", type=int)
    parser.add_argument("--evaluation-every-batches", type=int)
    parser.add_argument("--evaluation-games", type=int)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--parallel-inference-batch-size", type=int, default=2)
    parser.add_argument("--parallel-min-rollouts", type=int, default=100)
    parser.add_argument(
        "--parallel-pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=10)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--stop-after-batches", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/training")
    )
    parser.add_argument(
        "--state-checkpoint",
        type=Path,
        default=Path("checkpoints/training/connzero_training_state.pt"),
    )
    parser.add_argument(
        "--final-checkpoint",
        type=Path,
        default=Path("checkpoints/training/connzero_final_trained.pt"),
    )
    parser.add_argument(
        "--iterations-output",
        type=Path,
        default=Path("results/training_iterations.csv"),
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("results/gpu_metrics.csv")
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("results/training_profile.jsonl"),
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("results/final_training_config.json"),
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=Path("runs/alphazero_full"),
    )
    args = parser.parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.checkpoint_every_batches < 1 or args.progress_every_batches < 1:
        parser.error("checkpoint/progress intervals must be positive")
    positive_fields = (
        "max_iterations",
        "min_games_per_iteration",
        "self_play_games_per_iteration",
        "replay_buffer_games",
        "replay_sample_games",
        "training_updates_per_batch",
    )
    if any(getattr(args, field) < 1 for field in positive_fields):
        parser.error("iteration, game, replay, and update limits must be positive")
    if args.min_games_per_iteration > args.self_play_games_per_iteration:
        parser.error("min games per iteration cannot exceed its game cap")
    if args.max_total_games is not None and args.max_total_games < 1:
        parser.error("max total games must be positive")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps < 1:
        parser.error("max optimizer steps must be positive")
    if args.evaluation_every_batches < 0:
        parser.error("evaluation frequency cannot be negative")
    if args.evaluation_games < 0 or (
        args.evaluation_games > 0 and args.evaluation_games % 2 != 0
    ):
        parser.error("evaluation games must be zero or a positive even number")
    result = run(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
