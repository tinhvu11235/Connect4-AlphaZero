"""Single-GPU batched inference service with CPU self-play workers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from connect4_ai.models.policy_net import PolicyGradientNet, policy_board_to_tensor
from connect4_ai.training.self_play_worker import self_play_worker
from connect4_ai.utils.device import CudaRequiredError


@dataclass(frozen=True)
class ParallelSelfPlayConfig:
    workers: int = 8
    inference_batch_size: int = 32
    inference_wait_ms: float = 1.0
    opponent_weight: float = 0.05
    opponent_rollouts: int = 50
    gamma: float = 0.95
    max_steps: int = 50
    pin_memory: bool = True


class ParallelSelfPlayPool:
    """Persistent Windows-safe workers sharing one parent-owned CUDA service."""

    def __init__(
        self,
        learner_model: PolicyGradientNet,
        opponent_model: PolicyGradientNet,
        *,
        config: ParallelSelfPlayConfig,
        device: torch.device | str = "cuda",
        opponent_device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise CudaRequiredError("parallel self-play inference requires CUDA")
        self.opponent_device = torch.device(opponent_device or self.device)
        if self.opponent_device.type != "cuda":
            raise CudaRequiredError("parallel self-play inference requires CUDA")
        if config.workers < 1 or config.inference_batch_size < 1:
            raise ValueError("workers and inference_batch_size must be positive")
        self.learner_model = learner_model.to(self.device)
        # Keep the trainer's frozen snapshot on the primary GPU intact.  A
        # separate copy on GPU 1 lets learner and opponent inference overlap.
        self.opponent_model = (
            opponent_model.to(self.device).eval()
            if self.opponent_device == self.device
            else deepcopy(opponent_model).to(self.opponent_device).eval()
        )
        self.config = config
        buffer_shape = (config.inference_batch_size, 1, 7, 6)
        self._devices = {
            "learner": self.device,
            "opponent": self.opponent_device,
        }
        self._host_buffers = {
            kind: (
                torch.empty(buffer_shape, dtype=torch.float32, pin_memory=True)
                if config.pin_memory
                else None
            )
            for kind in self._devices
        }
        self._device_buffers = {
            kind: (
                torch.empty(buffer_shape, dtype=torch.float32, device=target)
                if config.pin_memory
                else None
            )
            for kind, target in self._devices.items()
        }
        self._inference_executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="cuda-inference")
            if self.opponent_device != self.device
            else None
        )
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue()
        self.inference_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.response_queues = [self.context.Queue() for _ in range(config.workers)]
        self.processes: list[mp.Process] = []
        self.closed = False
        for worker_id in range(config.workers):
            process = self.context.Process(
                target=self_play_worker,
                args=(
                    worker_id,
                    self.task_queue,
                    self.inference_queue,
                    self.response_queues[worker_id],
                    self.result_queue,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

    def __enter__(self) -> ParallelSelfPlayPool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if self._inference_executor is not None:
            self._inference_executor.shutdown(wait=True)
        for queue in (
            self.task_queue,
            self.inference_queue,
            self.result_queue,
            *self.response_queues,
        ):
            queue.close()
            queue.join_thread()

    def generate(
        self,
        games: int,
        *,
        seed: int,
        first_side: str = "red",
        learner_side: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.closed:
            raise RuntimeError("parallel self-play pool is closed")
        if games < 1:
            raise ValueError("games must be positive")
        if learner_side not in (None, "red", "yellow"):
            raise ValueError("learner_side must be None, 'red', or 'yellow'")
        for episode_id in range(games):
            if learner_side is not None:
                side = learner_side
            elif first_side == "red":
                side = "red" if episode_id % 2 == 0 else "yellow"
            else:
                side = "yellow" if episode_id % 2 == 0 else "red"
            self.task_queue.put(
                {
                    "episode_id": episode_id,
                    "learner_side": side,
                    "seed": seed + episode_id,
                    "opponent_weight": self.config.opponent_weight,
                    "opponent_rollouts": self.config.opponent_rollouts,
                    "gamma": self.config.gamma,
                    "max_steps": self.config.max_steps,
                }
            )

        started = perf_counter()
        results: list[dict[str, Any]] = []
        inference_requests = 0
        inference_batches = 0
        observed_batch_sizes: list[int] = []
        model_batch_sizes: list[int] = []
        service_profile = {
            "data_prep_seconds": 0.0,
            "h2d_seconds": 0.0,
            "forward_seconds": 0.0,
            "response_seconds": 0.0,
        }
        self.learner_model.eval()
        self.opponent_model.eval()
        while len(results) < games:
            results.extend(self._drain_results())
            if len(results) >= games:
                break
            try:
                first = self.inference_queue.get(timeout=0.1)
            except Empty:
                dead = [p.pid for p in self.processes if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"self-play workers exited early: {dead}")
                continue
            requests = [first]
            deadline = perf_counter() + self.config.inference_wait_ms / 1000.0
            while len(requests) < self.config.inference_batch_size:
                try:
                    requests.append(self.inference_queue.get_nowait())
                except Empty:
                    if perf_counter() >= deadline:
                        break
            served = self._serve_requests(requests)
            for key in service_profile:
                service_profile[key] += float(served[key])
            model_batch_sizes.extend(served["model_batch_sizes"])
            inference_requests += len(requests)
            inference_batches += 1
            observed_batch_sizes.append(len(requests))

        elapsed = perf_counter() - started
        results.sort(key=lambda item: item["episode_id"])
        return results, {
            "games": games,
            "elapsed_seconds": elapsed,
            "games_per_second": games / elapsed,
            "policy_steps": sum(item["policy_steps"] for item in results),
            "completed_games": sum(int(item["completed"]) for item in results),
            "inference_requests": inference_requests,
            "inference_batches": inference_batches,
            "average_inference_batch_size": (
                sum(observed_batch_sizes) / len(observed_batch_sizes)
            ),
            "maximum_inference_batch_size": max(observed_batch_sizes),
            # A mixed learner/opponent queue batch can require two model
            # forwards.  Report the actual forward batch size as well as the
            # aggregate queue batch size above.
            "average_model_batch_size": (
                sum(model_batch_sizes) / len(model_batch_sizes)
            ),
            "maximum_model_batch_size": max(model_batch_sizes),
            "model_inference_batches": len(model_batch_sizes),
            "mcts_seconds": sum(
                float(item["profile"]["mcts_seconds"]) for item in results
            ),
            "mcts_rollouts": sum(
                int(item["profile"]["mcts_rollouts"]) for item in results
            ),
            "worker_inference_wait_seconds": sum(
                float(item["profile"]["nn_wait_seconds"]) for item in results
            ),
            "worker_inference_requests": sum(
                int(item["profile"]["nn_requests"]) for item in results
            ),
            **service_profile,
        }

    def _drain_results(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        while True:
            try:
                kind, payload = self.result_queue.get_nowait()
            except Empty:
                return output
            if kind == "error":
                raise RuntimeError(f"self-play worker failed: {payload}")
            output.append(payload)

    def _serve_requests(self, requests: list[tuple[Any, ...]]) -> dict[str, Any]:
        grouped = {
            model_kind: [request for request in requests if request[2] == model_kind]
            for model_kind in ("learner", "opponent")
        }
        active = [(kind, selected) for kind, selected in grouped.items() if selected]
        if self._inference_executor is not None and len(active) == 2:
            profiles = list(
                self._inference_executor.map(
                    lambda item: self._serve_model_requests(*item), active
                )
            )
        else:
            profiles = [self._serve_model_requests(*item) for item in active]

        combined: dict[str, Any] = {
            "data_prep_seconds": 0.0,
            "h2d_seconds": 0.0,
            "forward_seconds": 0.0,
            "response_seconds": 0.0,
            "model_batch_sizes": [],
        }
        for profile in profiles:
            for key in (
                "data_prep_seconds",
                "h2d_seconds",
                "forward_seconds",
                "response_seconds",
            ):
                combined[key] += profile[key]
            combined["model_batch_sizes"].extend(profile["model_batch_sizes"])
        return combined

    def _serve_model_requests(
        self, model_kind: str, selected: list[tuple[Any, ...]]
    ) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "data_prep_seconds": 0.0,
            "h2d_seconds": 0.0,
            "forward_seconds": 0.0,
            "response_seconds": 0.0,
            "model_batch_sizes": [len(selected)],
        }
        target_device = self._devices[model_kind]
        model = (
            self.learner_model if model_kind == "learner" else self.opponent_model
        )
        with torch.cuda.device(target_device):
            data_started = perf_counter()
            boards_np = np.stack([request[3] for request in selected])
            cpu_tensor = policy_board_to_tensor(boards_np)
            profile["data_prep_seconds"] += perf_counter() - data_started
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            if self.config.pin_memory:
                count = len(selected)
                host_buffer = self._host_buffers[model_kind]
                device_buffer = self._device_buffers[model_kind]
                assert host_buffer is not None
                assert device_buffer is not None
                host_view = host_buffer[:count]
                device_tensor = device_buffer[:count]
                host_view.copy_(cpu_tensor)
                device_tensor.copy_(host_view, non_blocking=True)
            else:
                device_tensor = cpu_tensor.to(target_device)
            h2d_end.record()
            forward_start.record()
            with torch.inference_mode():
                probability_tensor = model(device_tensor).float()
                if not torch.isfinite(probability_tensor).all().item():
                    raise FloatingPointError(
                        f"non-finite {model_kind} policy probabilities during inference"
                    )
                row_sums = probability_tensor.sum(dim=1)
                if not (row_sums > 0).all().item():
                    raise FloatingPointError(
                        f"zero-sum {model_kind} policy probabilities during inference"
                    )
                probabilities = probability_tensor.cpu().numpy()
            forward_end.record()
            # The D2H copy above synchronizes the result.  Synchronize once
            # explicitly so CUDA-event phase timings are valid.
            forward_end.synchronize()
            profile["h2d_seconds"] += h2d_start.elapsed_time(h2d_end) / 1000.0
            profile["forward_seconds"] += (
                forward_start.elapsed_time(forward_end) / 1000.0
            )
            response_started = perf_counter()
            for request, result in zip(selected, probabilities):
                worker_id, request_id = request[0], request[1]
                self.response_queues[worker_id].put((request_id, result))
            profile["response_seconds"] += perf_counter() - response_started
        return profile


def policy_gradient_loss_from_logits(
    logits: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable REINFORCE loss for selected actions.

    Computing ``log(softmax(logits))`` explicitly can overflow its backward
    pass when a selected probability is subnormal.  ``log_softmax`` keeps the
    equivalent calculation in log space.
    """

    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("non-finite policy logits before loss")
    log_probabilities = F.log_softmax(logits, dim=1)
    selected_log_probabilities = log_probabilities[
        torch.arange(len(actions), device=logits.device), actions
    ]
    loss = -torch.sum(selected_log_probabilities * returns)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("non-finite policy-gradient loss")
    return loss


def optimize_trajectories(
    model: PolicyGradientNet,
    optimizer: torch.optim.Optimizer,
    trajectories: list[dict[str, Any]],
    *,
    device: torch.device | str = "cuda",
) -> dict[str, float | int]:
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise CudaRequiredError("parallel trajectory optimization requires CUDA")
    data_started = perf_counter()
    boards = np.concatenate([item["boards"] for item in trajectories])
    actions = np.concatenate([item["actions"] for item in trajectories])
    returns = np.concatenate([item["returns"] for item in trajectories])
    inputs_cpu = policy_board_to_tensor(boards)
    action_cpu = torch.as_tensor(actions, dtype=torch.long)
    return_cpu = torch.as_tensor(returns, dtype=torch.float32)
    data_prep_seconds = perf_counter() - data_started

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    inputs = inputs_cpu.to(selected_device)
    action_tensor = action_cpu.to(selected_device)
    return_tensor = return_cpu.to(selected_device)
    h2d_end.record()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    forward_start.record()
    logits = model.forward_logits(inputs)
    loss = policy_gradient_loss_from_logits(logits, action_tensor, return_tensor)
    forward_end.record()
    backward_start.record()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    backward_end.record()
    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    torch.cuda.synchronize(selected_device)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(
                f"optimizer produced non-finite candidate parameter: {name}"
            )
    return {
        "samples": len(actions),
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clip": float(gradient_norm.item()),
        "data_prep_seconds": data_prep_seconds,
        "h2d_seconds": h2d_start.elapsed_time(h2d_end) / 1000.0,
        "forward_seconds": forward_start.elapsed_time(forward_end) / 1000.0,
        "backward_seconds": backward_start.elapsed_time(backward_end) / 1000.0,
        "optimizer_seconds": optimizer_start.elapsed_time(optimizer_end) / 1000.0,
    }


__all__ = [
    "ParallelSelfPlayConfig",
    "ParallelSelfPlayPool",
    "optimize_trajectories",
    "policy_gradient_loss_from_logits",
]
