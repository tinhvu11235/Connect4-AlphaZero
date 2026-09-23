"""CUDA trainer for Chapter 15's Connect Four value network."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import random
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connect4_ai.models.value_net import ValueNet, board_to_tensor  # noqa: E402
from connect4_ai.utils.device import resolve_device  # noqa: E402


def load_chapter15_value_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chapter 15 generated dataset is missing: {path}. Recreate it "
            "with the original alternating policy-gradient gameplay pipeline; "
            "do not substitute fabricated labels."
        )
    with path.open("rb") as stream:
        histories, outcomes = pickle.load(stream)
    boards: list[np.ndarray] = []
    labels: list[int] = []
    class_for_outcome = {0: 0, 1: 1, -1: 2}
    for states, outcome in zip(histories, outcomes):
        for state in states:
            boards.append(np.asarray(state))
            labels.append(class_for_outcome[int(outcome)])
    return (
        board_to_tensor(np.asarray(boards).reshape(-1, 7, 6)),
        torch.tensor(labels, dtype=torch.long),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("archive/book/artifacts/files/PG_games_conn.p"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/pretrained/value_conn.pt")
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/value_training_metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(allow_cpu=False)
    inputs, targets = load_chapter15_value_dataset(args.dataset)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = ValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, eps=1e-7)
    history: list[dict[str, float | int | None]] = []
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
                "train_loss": loss_sum / seen,
                "train_accuracy": correct / seen,
                "validation_loss": None,
                "validation_accuracy": None,
                "samples_per_second": seen / elapsed,
            }
        )

    checkpoint = {
        "format_version": 1,
        "model_name": "ValueNet",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "config": {
            "dataset": str(args.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 0.0005,
            "validation_split": None,
        },
        "seed": args.seed,
        "training_metadata": {
            "timestamp": datetime.now().astimezone().isoformat(),
            "device": str(device),
            "samples": len(dataset),
            "history": history,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
