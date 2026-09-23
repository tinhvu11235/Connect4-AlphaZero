"""CUDA training entry points for the two Chapter 11/15 policy networks.

The fast-policy dataset is absent from this repository. The supervised mode
therefore fails clearly unless the Chapter 11-generated pickle is supplied.
The policy-gradient mode reproduces Chapter 15's online gameplay method.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import pickle
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connect4_ai.env import ConnectFourEnv  # noqa: E402
from connect4_ai.models.policy_net import (  # noqa: E402
    FastPolicyNet,
    PolicyGradientNet,
    board_from_player_perspective,
    load_policy_checkpoint,
    policy_board_to_tensor,
)
from connect4_ai.utils.device import resolve_device  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_chapter11_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chapter 11 generated dataset is missing: {path}. Recreate it "
            "with the notebook's 10,000-game MiniMax/MCTS self-play pipeline; "
            "do not substitute unrelated or fabricated labels."
        )
    with path.open("rb") as stream:
        games = pickle.load(stream)
    boards: list[np.ndarray] = []
    actions: list[int] = []
    for reward, history in games:
        if reward > 0:
            for state, action, turn in history:
                if turn == "red":
                    boards.append(np.asarray(state))
                    actions.append(int(action) - 1)
        elif reward < 0:
            for state, action, turn in history:
                if turn == "yellow":
                    boards.append(-np.asarray(state))
                    actions.append(int(action) - 1)
    return (
        policy_board_to_tensor(np.asarray(boards).reshape(-1, 7, 6)),
        torch.tensor(actions, dtype=torch.long),
    )


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int | None,
    config: dict[str, Any],
    seed: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "seed": seed,
        "training_metadata": metadata,
    }


def train_fast(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    inputs, targets = load_chapter11_dataset(args.dataset)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = FastPolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = perf_counter()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for boards, labels in loader:
            boards = boards.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(boards)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            count = labels.shape[0]
            loss_sum += float(loss.detach().item()) * count
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += count
        torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / seen,
                "accuracy": correct / seen,
                "samples_per_second": seen / elapsed,
            }
        )
    payload = checkpoint_payload(
        model,
        optimizer,
        epoch=args.epochs,
        config={
            "mode": "fast_supervised",
            "dataset": str(args.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 0.001,
            "validation_split": None,
        },
        seed=args.seed,
        metadata={
            "timestamp": datetime.now().astimezone().isoformat(),
            "device": str(device),
            "samples": len(dataset),
            "history": history,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return history


class PolicyGradientTrainer:
    """Chapter 15's alternating red/yellow REINFORCE loop."""

    def __init__(
        self,
        model: PolicyGradientNet,
        opponent: FastPolicyNet,
        *,
        device: torch.device,
        seed: int,
        gamma: float = 0.95,
        max_steps: int = 50,
    ) -> None:
        self.model = model
        self.opponent_model = opponent.eval()
        self.device = device
        self.rng = random.Random(seed)
        self.gamma = gamma
        self.max_steps = max_steps
        self.all_states: list[list[np.ndarray]] = []
        self.all_outcomes: list[int] = []

    def opponent_action(self, env: ConnectFourEnv) -> int:
        board = board_from_player_perspective(env.state, env.turn)
        tensor = policy_board_to_tensor(board, device=self.device)
        with torch.inference_mode():
            probabilities = self.opponent_model(tensor)[0]
        legal = sorted(env.validinputs)
        weights = [float(probabilities[action - 1].item()) for action in legal]
        return self.rng.choices(legal, weights=weights, k=1)[0]

    def policy_action(
        self, env: ConnectFourEnv
    ) -> tuple[int, torch.Tensor]:
        board = board_from_player_perspective(env.state, env.turn)
        tensor = policy_board_to_tensor(board, device=self.device)
        probabilities = self.model(tensor)[0]
        action_index = int(torch.multinomial(probabilities, 1).item())
        return action_index + 1, torch.log(probabilities[action_index])

    def discounted_returns(
        self, rewards: list[float], wrong_moves: list[int]
    ) -> list[float]:
        discounted = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for index in reversed(range(len(rewards))):
            if wrong_moves[index] == 0:
                running = self.gamma * running + rewards[index]
                discounted[index] = running
        return (discounted + np.asarray(wrong_moves)).tolist()

    def play(self, *, yellow: bool) -> tuple[list[torch.Tensor], list[float], float]:
        env = ConnectFourEnv()
        state = env.reset()
        log_probabilities: list[torch.Tensor] = []
        wrong_moves: list[int] = []
        rewards: list[float] = []
        states: list[np.ndarray] = []
        episode_reward = 0.0
        reward = 0

        if yellow:
            state, reward, done, _ = env.step(self.opponent_action(env))
            if done:
                self.all_states.append(states)
                self.all_outcomes.append(-reward)
                return log_probabilities, rewards, float(-reward)

        for _ in range(self.max_steps):
            action, log_probability = self.policy_action(env)
            log_probabilities.append(log_probability)
            if action not in env.validinputs:
                rewards.append(0.0)
                wrong_moves.append(-1)
                continue

            state, reward, done, _ = env.step(action)
            states.append(-state if yellow else state)
            perspective_reward = -reward if yellow else reward
            wrong_moves.append(0)
            rewards.append(float(perspective_reward))
            episode_reward += float(perspective_reward)
            if done:
                break

            state, reward, done, _ = env.step(self.opponent_action(env))
            perspective_reward = -reward if yellow else reward
            rewards[-1] = float(perspective_reward)
            episode_reward += float(perspective_reward)
            if done:
                break

        outcome = -reward if yellow else reward
        self.all_states.append(states)
        self.all_outcomes.append(int(outcome))
        return (
            log_probabilities,
            self.discounted_returns(rewards, wrong_moves),
            episode_reward,
        )


def train_policy_gradient(
    args: argparse.Namespace, device: torch.device
) -> list[dict[str, Any]]:
    opponent, _ = load_policy_checkpoint(
        args.fast_checkpoint, model_type="fast", device=device
    )
    model = PolicyGradientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    trainer = PolicyGradientTrainer(
        model,
        opponent,  # type: ignore[arg-type]
        device=device,
        seed=args.seed,
    )
    running_rewards: deque[float] = deque(maxlen=100)
    history: list[dict[str, Any]] = []
    episodes = 0
    batches = 0
    while args.max_episodes == 0 or episodes < args.max_episodes:
        model.train()
        logs: list[torch.Tensor] = []
        returns: list[float] = []
        batch_rewards: list[float] = []
        yellow = batches % 2 == 1
        for _ in range(10):
            game_logs, game_returns, episode_reward = trainer.play(yellow=yellow)
            logs.extend(game_logs)
            returns.extend(game_returns)
            batch_rewards.append(episode_reward)
        optimizer.zero_grad(set_to_none=True)
        loss = -torch.sum(
            torch.stack(logs)
            * torch.tensor(returns, dtype=torch.float32, device=device)
        )
        loss.backward()
        optimizer.step()
        episodes += 10
        batches += 1
        running_rewards.extend(batch_rewards)
        running_reward = float(np.mean(running_rewards))
        history.append(
            {
                "episodes": episodes,
                "batch": batches,
                "loss": float(loss.detach().item()),
                "running_reward": running_reward,
                "side": "yellow" if yellow else "red",
            }
        )
        if running_reward >= 0.9 and episodes > 100:
            break

    args.pg_dataset_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pg_dataset_output.open("wb") as stream:
        pickle.dump((trainer.all_states, trainer.all_outcomes), stream)
    payload = checkpoint_payload(
        model,
        optimizer,
        epoch=None,
        config={
            "mode": "policy_gradient",
            "learning_rate": 0.00025,
            "gamma": 0.95,
            "batch_games": 10,
            "max_steps": 50,
            "stop_running_reward": 0.9,
            "fast_checkpoint": str(args.fast_checkpoint),
        },
        seed=args.seed,
        metadata={
            "timestamp": datetime.now().astimezone().isoformat(),
            "device": str(device),
            "episodes": episodes,
            "history": history,
            "generated_value_dataset": str(args.pg_dataset_output),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fast-supervised", "policy-gradient"),
        default="fast-supervised",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("archive/book/artifacts/files/games_conn.p"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/pretrained/policy_conn.pt")
    )
    parser.add_argument(
        "--fast-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrained/policy_conn.pt"),
    )
    parser.add_argument(
        "--pg-dataset-output",
        type=Path,
        default=Path("archive/book/artifacts/files/PG_games_conn.p"),
    )
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(allow_cpu=False)
    if args.mode == "fast-supervised":
        history = train_fast(args, device)
    else:
        if args.output == Path("checkpoints/pretrained/policy_conn.pt"):
            args.output = Path("checkpoints/pretrained/policy_gradient_conn.pt")
        history = train_policy_gradient(args, device)
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
