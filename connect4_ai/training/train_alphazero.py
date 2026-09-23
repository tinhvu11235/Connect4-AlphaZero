"""Chapter 21 Connect Four AlphaZero training and checkpoint migration.

The default mode is a deliberately short CUDA validation. Use ``--mode full``
only when intentionally starting the notebook's multi-day iterative training.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connect4_ai.agents.alphazero import AlphaZeroAgent  # noqa: E402
from connect4_ai.env import ConnectFourEnv  # noqa: E402
from connect4_ai.models.policy_net import (  # noqa: E402
    PolicyGradientNet,
    board_from_player_perspective,
    load_keras_policy_gradient_weights,
    policy_board_to_tensor,
)
from connect4_ai.training.parallel_self_play import (  # noqa: E402
    ParallelSelfPlayConfig,
    ParallelSelfPlayPool,
    optimize_trajectories,
)
from connect4_ai.utils.device import CudaRequiredError, resolve_device  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def initialize_like_keras(model: nn.Module) -> None:
    """Apply the source model's Glorot-uniform/zero-bias initialization."""

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def assert_finite_module(model: nn.Module, *, label: str) -> None:
    """Reject corrupted model state before inference or checkpointing."""

    for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"non-finite {label} tensor: {name}")


def assert_finite_optimizer(
    optimizer: torch.optim.Optimizer, *, label: str
) -> None:
    """Reject non-finite optimizer moments when loading/saving state."""

    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                raise FloatingPointError(
                    f"non-finite {label} optimizer state {name} "
                    f"for parameter shape {tuple(parameter.shape)}"
                )


def _policy_distribution(
    model: PolicyGradientNet, tensor: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.forward_logits(tensor)[0]
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError(f"non-finite policy logits during {context}")
    log_probabilities = F.log_softmax(logits, dim=0)
    probabilities = log_probabilities.exp()
    if not torch.isfinite(probabilities).all().item():
        raise FloatingPointError(f"non-finite policy probabilities during {context}")
    if not (probabilities.sum() > 0).item():
        raise FloatingPointError(f"zero-sum policy probabilities during {context}")
    return probabilities, log_probabilities


@dataclass
class EpisodeTrace:
    log_probabilities: list[torch.Tensor]
    combined_returns: list[float]
    episode_reward: float
    completed: bool
    policy_steps: int


class AlphaZeroTrainer:
    """Faithful alternating-side REINFORCE trainer from Chapter 21."""

    def __init__(
        self,
        model: PolicyGradientNet,
        *,
        device: torch.device | str | None = None,
        seed: int = 20260923,
        opponent_weight: float = 0.05,
        opponent_rollouts: int = 50,
        gamma: float = 0.95,
        max_steps: int = 50,
    ) -> None:
        selected_device = resolve_device(allow_cpu=False) if device is None else torch.device(device)
        if selected_device.type != "cuda":
            raise CudaRequiredError("AlphaZero training requires CUDA")
        self.device = selected_device
        self.seed = seed
        self.gamma = gamma
        self.max_steps = max_steps
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.00025, eps=1e-7
        )
        self.sample_generator = torch.Generator(device=self.device).manual_seed(seed)
        self._opponent_seed = seed + 1
        self.opponent_model: PolicyGradientNet
        self.opponent: AlphaZeroAgent
        self.set_opponent_snapshot(
            weight=opponent_weight, num_rollouts=opponent_rollouts
        )

    def set_opponent_snapshot(self, *, weight: float, num_rollouts: int) -> None:
        self.opponent_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.opponent_model.parameters():
            parameter.requires_grad_(False)
        self.opponent = AlphaZeroAgent(
            self.opponent_model,
            weight=weight,
            num_rollouts=num_rollouts,
            seed=self._opponent_seed,
            device=self.device,
        )
        self._opponent_seed += 1

    def discounted_returns(
        self, rewards: list[float], wrong_moves: list[int]
    ) -> list[float]:
        discounted = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for index in reversed(range(len(rewards))):
            if wrong_moves[index] == 0:
                running = self.gamma * running + rewards[index]
                discounted[index] = running
        return (discounted + np.asarray(wrong_moves, dtype=np.float32)).tolist()

    def _sample_policy_action(
        self, env: ConnectFourEnv
    ) -> tuple[int, torch.Tensor]:
        board = board_from_player_perspective(env.state, env.turn)
        tensor = policy_board_to_tensor(board, device=self.device)
        probabilities, log_probabilities = _policy_distribution(
            self.model, tensor, context="gradient-tracked self-play"
        )
        index = int(
            torch.multinomial(
                probabilities, 1, generator=self.sample_generator
            ).item()
        )
        return index + 1, log_probabilities[index]

    def play_episode(self, *, learner_side: str) -> EpisodeTrace:
        if learner_side not in ("red", "yellow"):
            raise ValueError("learner_side must be 'red' or 'yellow'")
        env = ConnectFourEnv()
        env.reset()
        log_probabilities: list[torch.Tensor] = []
        wrong_moves: list[int] = []
        rewards: list[float] = []
        episode_reward = 0.0

        if learner_side == "yellow":
            _, _, done, _ = env.step(self.opponent.select_action(env))
            if done:
                raise RuntimeError("Connect Four cannot terminate after one move")

        completed = False
        for _ in range(self.max_steps):
            action, log_probability = self._sample_policy_action(env)
            log_probabilities.append(log_probability)
            if action not in env.validinputs:
                rewards.append(0.0)
                wrong_moves.append(-1)
                continue

            _, reward, done, _ = env.step(action)
            perspective_reward = reward if learner_side == "red" else -reward
            wrong_moves.append(0)
            if done:
                rewards.append(float(perspective_reward))
                episode_reward += float(perspective_reward)
                completed = True
                break

            _, reward, done, _ = env.step(self.opponent.select_action(env))
            perspective_reward = reward if learner_side == "red" else -reward
            rewards.append(float(perspective_reward))
            episode_reward += float(perspective_reward)
            if done:
                completed = True
                break

        return EpisodeTrace(
            log_probabilities=log_probabilities,
            combined_returns=self.discounted_returns(rewards, wrong_moves),
            episode_reward=episode_reward,
            completed=completed,
            policy_steps=len(log_probabilities),
        )

    def train_batch(self, *, learner_side: str, batch_size: int) -> dict[str, Any]:
        self.model.train()
        logs: list[torch.Tensor] = []
        returns: list[float] = []
        episode_rewards: list[float] = []
        completed_games = 0
        policy_steps = 0
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        for _ in range(batch_size):
            trace = self.play_episode(learner_side=learner_side)
            logs.extend(trace.log_probabilities)
            returns.extend(trace.combined_returns)
            episode_rewards.append(trace.episode_reward)
            completed_games += int(trace.completed)
            policy_steps += trace.policy_steps

        self.optimizer.zero_grad(set_to_none=True)
        return_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        loss = -torch.sum(torch.stack(logs) * return_tensor)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite policy-gradient loss")
        loss.backward()
        unclipped_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        torch.cuda.synchronize(self.device)
        assert_finite_module(self.model, label="candidate after optimizer step")

        change_squared = torch.zeros((), device=self.device)
        for old, new in zip(before, self.model.parameters()):
            change_squared += torch.sum((old - new.detach()) ** 2)
        return {
            "side": learner_side,
            "batch_size": batch_size,
            "loss": float(loss.detach().item()),
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "episode_rewards": episode_rewards,
            "completed_games": completed_games,
            "policy_steps": policy_steps,
            "gradient_norm_before_clip": float(unclipped_gradient_norm.item()),
            "parameter_change_l2": float(torch.sqrt(change_squared).item()),
            "device": str(self.device),
        }

    def collect_episode(self, *, learner_side: str) -> dict[str, Any]:
        """Collect one equivalent trajectory without retaining per-move graphs."""

        if learner_side not in ("red", "yellow"):
            raise ValueError("learner_side must be 'red' or 'yellow'")
        env = ConnectFourEnv()
        env.reset()
        boards: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        wrong_moves: list[int] = []
        episode_reward = 0.0
        profile = {
            "mcts_seconds": 0.0,
            "mcts_rollouts": 0,
            "nn_wait_seconds": 0.0,
            "nn_requests": 0,
        }

        if learner_side == "yellow":
            env.step(self.opponent.select_action(env))
            profile["mcts_seconds"] += self.opponent.metrics.rollout_time
            profile["mcts_rollouts"] += self.opponent.metrics.rollouts
            profile["nn_wait_seconds"] += self.opponent.metrics.policy_inference_time
            profile["nn_requests"] += self.opponent.metrics.policy_inference_calls

        completed = False
        self.model.eval()
        for _ in range(self.max_steps):
            board = board_from_player_perspective(env.state, env.turn)
            tensor = policy_board_to_tensor(board, device=self.device)
            inference_started = perf_counter()
            with torch.inference_mode():
                probabilities, _ = _policy_distribution(
                    self.model, tensor, context="detached self-play"
                )
            torch.cuda.synchronize(self.device)
            profile["nn_wait_seconds"] += perf_counter() - inference_started
            profile["nn_requests"] += 1
            index = int(
                torch.multinomial(
                    probabilities, 1, generator=self.sample_generator
                ).item()
            )
            boards.append(np.asarray(board, dtype=np.float32).copy())
            actions.append(index)
            action = index + 1
            if action not in env.validinputs:
                rewards.append(0.0)
                wrong_moves.append(-1)
                continue

            _, reward, done, _ = env.step(action)
            perspective_reward = reward if learner_side == "red" else -reward
            wrong_moves.append(0)
            if done:
                rewards.append(float(perspective_reward))
                episode_reward += float(perspective_reward)
                completed = True
                break
            _, reward, done, _ = env.step(self.opponent.select_action(env))
            profile["mcts_seconds"] += self.opponent.metrics.rollout_time
            profile["mcts_rollouts"] += self.opponent.metrics.rollouts
            profile["nn_wait_seconds"] += self.opponent.metrics.policy_inference_time
            profile["nn_requests"] += self.opponent.metrics.policy_inference_calls
            perspective_reward = reward if learner_side == "red" else -reward
            rewards.append(float(perspective_reward))
            episode_reward += float(perspective_reward)
            if done:
                completed = True
                break

        return {
            "boards": np.asarray(boards, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "returns": np.asarray(
                self.discounted_returns(rewards, wrong_moves), dtype=np.float32
            ),
            "episode_reward": episode_reward,
            "completed": completed,
            "policy_steps": len(actions),
            "game_length": int(np.count_nonzero(env.state)),
            "profile": profile,
        }

    def train_batch_vectorized(
        self, *, learner_side: str, batch_size: int
    ) -> dict[str, Any]:
        """Collect source-equivalent games, then recompute one batched loss."""

        trajectories = [
            self.collect_episode(learner_side=learner_side)
            for _ in range(batch_size)
        ]
        update = optimize_trajectories(
            self.model, self.optimizer, trajectories, device=self.device
        )
        return {
            "side": learner_side,
            "batch_size": batch_size,
            "loss": update["loss"],
            "mean_episode_reward": float(
                np.mean([item["episode_reward"] for item in trajectories])
            ),
            "episode_rewards": [
                item["episode_reward"] for item in trajectories
            ],
            "completed_games": sum(
                int(item["completed"]) for item in trajectories
            ),
            "policy_steps": sum(item["policy_steps"] for item in trajectories),
            "gradient_norm_before_clip": update["gradient_norm_before_clip"],
            "device": str(self.device),
            "vectorized_update": True,
        }


def save_training_checkpoint(
    path: Path,
    trainer: AlphaZeroTrainer,
    *,
    iteration: int | str,
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    assert_finite_module(trainer.model, label="candidate")
    assert_finite_module(trainer.opponent_model, label="reference")
    assert_finite_optimizer(trainer.optimizer, label="candidate")
    checkpoint = {
        "format_version": 1,
        "model_name": "Chapter21PolicyNet",
        "model_state_dict": trainer.model.state_dict(),
        "opponent_model_state_dict": trainer.opponent_model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
        "seed": trainer.seed,
        "training_metadata": {
            "timestamp": datetime.now().astimezone().isoformat(),
            "device": str(trainer.device),
            "history": history,
            "source_notebook": "ch21AlphaZeroUnsolvedGames.ipynb",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_final_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    model = load_keras_policy_gradient_weights(PolicyGradientNet(), source).eval()
    checkpoint = {
        "format_version": 1,
        "model_name": "Chapter21PolicyNet",
        "model_state_dict": model.state_dict(),
        "opponent_model_state_dict": None,
        "optimizer_state_dict": None,
        "iteration": 4,
        "config": {
            "board_shape": [7, 6],
            "raw_input_features": 42,
            "conv_filters": 128,
            "conv_kernel": [4, 4],
            "hidden_units": [256, 64],
            "output_actions": 7,
            "source_optimizer": "Adam",
            "source_learning_rate": 0.00025,
            "source_global_clipnorm": 1.0,
        },
        "seed": None,
        "training_metadata": {
            "source_notebook": "ch21AlphaZeroUnsolvedGames.ipynb",
            "source_model": source.as_posix(),
            "source_sha256": sha256(source),
            "conversion_method": "direct HDF5 weight transpose; no retraining",
            "conversion_timestamp": datetime.now().astimezone().isoformat(),
            "missing_predecessors": [
                "archive/book/artifacts/files/CONNzero0.h5",
                "archive/book/artifacts/files/CONNzero1.h5",
                "archive/book/artifacts/files/CONNzero2.h5",
                "archive/book/artifacts/files/CONNzero3.h5",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "source": str(source),
        "output": str(output),
        "source_sha256": checkpoint["training_metadata"]["source_sha256"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_validation(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed_everything(args.seed)
    device = resolve_device(allow_cpu=False)
    model = PolicyGradientNet()
    initialize_like_keras(model)
    trainer = AlphaZeroTrainer(
        model,
        device=device,
        seed=args.seed,
        opponent_weight=args.validation_weight,
        opponent_rollouts=args.validation_rollouts,
        max_steps=args.max_steps,
    )
    history: list[dict[str, Any]] = []
    for batch in range(args.validation_batches):
        side = "red" if batch % 2 == 0 else "yellow"
        metric = trainer.train_batch(
            learner_side=side, batch_size=args.validation_batch_size
        )
        metric["batch"] = batch + 1
        history.append(metric)
    save_training_checkpoint(
        args.output,
        trainer,
        iteration="short_validation",
        config={
            "validation_only": True,
            "batches": args.validation_batches,
            "batch_size": args.validation_batch_size,
            "gamma": 0.95,
            "max_steps": args.max_steps,
            "opponent_weight": args.validation_weight,
            "opponent_rollouts": args.validation_rollouts,
            "learning_rate": 0.00025,
            "global_clipnorm": 1.0,
        },
        history=history,
    )
    return history


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed_everything(args.seed)
    device = resolve_device(allow_cpu=False)
    model = PolicyGradientNet()
    initialize_like_keras(model)
    trainer = AlphaZeroTrainer(
        model,
        device=device,
        seed=args.seed,
        opponent_weight=0.05,
        opponent_rollouts=50,
        max_steps=50,
    )
    all_history: list[dict[str, Any]] = []
    schedule = [(0.05, 50), (0.3, 100), (0.5, 150), (0.7, 200), (0.9, 250)]
    for iteration, (weight, rollouts) in enumerate(schedule):
        if iteration > 0:
            trainer.set_opponent_snapshot(weight=weight, num_rollouts=rollouts)
        running_rewards: deque[float] = deque(maxlen=1000)
        episodes = 0
        batches = 0
        iteration_history: list[dict[str, Any]] = []
        use_parallel = rollouts >= args.parallel_min_rollouts
        pool_context: Any
        if use_parallel:
            pool_context = ParallelSelfPlayPool(
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
        else:
            pool_context = None
        try:
            while True:
                side = "red" if batches % 2 == 0 else "yellow"
                if pool_context is None:
                    metric = trainer.train_batch_vectorized(
                        learner_side=side, batch_size=20
                    )
                else:
                    trajectories, generation = pool_context.generate(
                        20,
                        seed=args.seed + iteration * 1_000_000 + episodes,
                        learner_side=side,
                    )
                    update = optimize_trajectories(
                        trainer.model,
                        trainer.optimizer,
                        trajectories,
                        device=device,
                    )
                    metric = {
                        "side": side,
                        "batch_size": 20,
                        "loss": update["loss"],
                        "mean_episode_reward": float(
                            np.mean(
                                [item["episode_reward"] for item in trajectories]
                            )
                        ),
                        "episode_rewards": [
                            item["episode_reward"] for item in trajectories
                        ],
                        "completed_games": generation["completed_games"],
                        "policy_steps": generation["policy_steps"],
                        "gradient_norm_before_clip": update[
                            "gradient_norm_before_clip"
                        ],
                        "device": str(device),
                        "parallel_generation": generation,
                    }
                episodes += 20
                batches += 1
                running_rewards.extend(metric["episode_rewards"])
                metric.update(
                    {
                        "iteration": iteration,
                        "episodes": episodes,
                        "running_reward": float(np.mean(running_rewards)),
                        "opponent_weight": weight,
                        "opponent_rollouts": rollouts,
                    }
                )
                iteration_history.append(metric)
                all_history.append(metric)
                if metric["running_reward"] >= 0.1 and episodes > 1000:
                    break
                if episodes > 25000:
                    break
        finally:
            if pool_context is not None:
                pool_context.close()
        promoted = (
            len(running_rewards) == running_rewards.maxlen
            and float(np.mean(running_rewards)) >= 0.1
            and episodes > 1000
        )
        suffix = "" if promoted else "_rejected"
        save_training_checkpoint(
            args.output_dir / f"connzero_iter_{iteration:03d}{suffix}.pt",
            trainer,
            iteration=iteration,
            config={
                "validation_only": False,
                "batch_size": 20,
                "gamma": 0.95,
                "max_steps": 50,
                "opponent_weight": weight,
                "opponent_rollouts": rollouts,
                "learning_rate": 0.00025,
                "global_clipnorm": 1.0,
                "stop_running_reward": 0.1,
                "stop_min_episodes": 1000,
                "max_episodes": 25000,
                "promoted": promoted,
            },
            history=iteration_history,
        )
        if not promoted:
            # Do not allow a safety-cap candidate to become the next
            # generation's frozen reference.
            trainer.model.load_state_dict(trainer.opponent_model.state_dict())
            break
    return all_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("validate", "full", "convert-final"), default="validate"
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/training/connzero_short_validation.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/alphazero_short_training.json"),
    )
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--validation-weight", type=float, default=0.05)
    parser.add_argument("--validation-rollouts", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--parallel-inference-batch-size", type=int, default=2)
    parser.add_argument("--parallel-min-rollouts", type=int, default=100)
    parser.add_argument(
        "--parallel-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=Path("archive/book/artifacts/files/CONNzero4.h5"),
    )
    args = parser.parse_args()

    if args.mode == "convert-final":
        output = args.output
        if output == Path("checkpoints/training/connzero_short_validation.pt"):
            output = Path("checkpoints/pretrained/connzero_book_iter_004.pt")
        result: Any = convert_final_checkpoint(args.source_h5, output)
    elif args.mode == "full":
        result = run_full(args)
    else:
        result = run_validation(args)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
