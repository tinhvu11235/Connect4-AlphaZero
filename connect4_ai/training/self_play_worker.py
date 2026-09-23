"""CPU-only worker for batched Chapter 21 self-play inference."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any

import numpy as np

from connect4_ai.env import ConnectFourEnv


def _request_policy(
    inference_queue: Any,
    response_queue: Any,
    worker_id: int,
    request_id: int,
    model_kind: str,
    env: ConnectFourEnv,
    profile: dict[str, float | int],
) -> tuple[np.ndarray, int]:
    board = env.state if env.turn == "red" else -env.state
    started = perf_counter()
    inference_queue.put(
        (worker_id, request_id, model_kind, np.asarray(board, dtype=np.float32))
    )
    returned_id, probabilities = response_queue.get()
    profile["nn_wait_seconds"] += perf_counter() - started
    profile["nn_requests"] += 1
    if returned_id != request_id:
        raise RuntimeError(
            f"worker {worker_id} expected response {request_id}, got {returned_id}"
        )
    output = np.asarray(probabilities, dtype=np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError(
            f"worker {worker_id} received non-finite {model_kind} probabilities"
        )
    total = float(output.sum())
    if total <= 0.0:
        raise FloatingPointError(
            f"worker {worker_id} received zero-sum {model_kind} probabilities"
        )
    return output / total, request_id + 1


def _select_root(
    priors: np.ndarray,
    legal: tuple[int, ...],
    results: dict[int, list[float]],
    weight: float,
) -> int:
    scores: dict[int, float] = {}
    for move in legal:
        outcomes = results[move]
        value = 0.0 if not outcomes else sum(outcomes) / len(outcomes)
        prior = float(priors[move - 1]) / (1 + len(outcomes))
        scores[move] = weight * prior + (1.0 - weight) * value
    return max(scores, key=scores.get)


def _opponent_action(
    env: ConnectFourEnv,
    *,
    inference_queue: Any,
    response_queue: Any,
    worker_id: int,
    request_id: int,
    weight: float,
    num_rollouts: int,
    rollout_rng: random.Random,
    profile: dict[str, float | int],
) -> tuple[int, int]:
    legal = tuple(env.validinputs)
    if len(legal) == 1:
        return legal[0], request_id
    priors, request_id = _request_policy(
        inference_queue,
        response_queue,
        worker_id,
        request_id,
        "opponent",
        env,
        profile,
    )
    results = {move: [] for move in legal}
    mcts_started = perf_counter()
    for _ in range(num_rollouts):
        move = _select_root(priors, legal, results, weight)
        child = env.copy()
        _, reward, done, _ = child.step(move)
        while not done:
            _, reward, done, _ = child.step(
                rollout_rng.choice(child.validinputs)
            )
        root_reward = float(reward if env.turn == "red" else -reward)
        results[move].append(root_reward)
    profile["mcts_seconds"] += perf_counter() - mcts_started
    profile["mcts_rollouts"] += num_rollouts
    visits = {move: len(outcomes) for move, outcomes in results.items()}
    return max(visits, key=visits.get), request_id


def _discounted_returns(
    rewards: list[float], wrong_moves: list[int], gamma: float
) -> list[float]:
    discounted = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        if wrong_moves[index] == 0:
            running = gamma * running + rewards[index]
            discounted[index] = running
    return (discounted + np.asarray(wrong_moves, dtype=np.float32)).tolist()


def _play_episode(
    task: dict[str, Any],
    *,
    worker_id: int,
    inference_queue: Any,
    response_queue: Any,
) -> dict[str, Any]:
    side = task["learner_side"]
    sampling_rng = np.random.default_rng(task["seed"])
    rollout_rng = random.Random(task["seed"] + 1_000_003)
    env = ConnectFourEnv()
    env.reset()
    boards: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    wrong_moves: list[int] = []
    episode_reward = 0.0
    request_id = 0
    profile: dict[str, float | int] = {
        "mcts_seconds": 0.0,
        "mcts_rollouts": 0,
        "nn_wait_seconds": 0.0,
        "nn_requests": 0,
    }

    if side == "yellow":
        action, request_id = _opponent_action(
            env,
            inference_queue=inference_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            request_id=request_id,
            weight=task["opponent_weight"],
            num_rollouts=task["opponent_rollouts"],
            rollout_rng=rollout_rng,
            profile=profile,
        )
        env.step(action)

    completed = False
    for _ in range(task["max_steps"]):
        perspective_board = env.state if env.turn == "red" else -env.state
        probabilities, request_id = _request_policy(
            inference_queue,
            response_queue,
            worker_id,
            request_id,
            "learner",
            env,
            profile,
        )
        action_index = int(sampling_rng.choice(7, p=probabilities / probabilities.sum()))
        boards.append(np.asarray(perspective_board, dtype=np.float32).copy())
        actions.append(action_index)
        action = action_index + 1
        if action not in env.validinputs:
            rewards.append(0.0)
            wrong_moves.append(-1)
            continue

        _, reward, done, _ = env.step(action)
        perspective_reward = reward if side == "red" else -reward
        wrong_moves.append(0)
        if done:
            rewards.append(float(perspective_reward))
            episode_reward += float(perspective_reward)
            completed = True
            break

        opponent_move, request_id = _opponent_action(
            env,
            inference_queue=inference_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            request_id=request_id,
            weight=task["opponent_weight"],
            num_rollouts=task["opponent_rollouts"],
            rollout_rng=rollout_rng,
            profile=profile,
        )
        _, reward, done, _ = env.step(opponent_move)
        perspective_reward = reward if side == "red" else -reward
        rewards.append(float(perspective_reward))
        episode_reward += float(perspective_reward)
        if done:
            completed = True
            break

    return {
        "episode_id": task["episode_id"],
        "learner_side": side,
        "boards": np.asarray(boards, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "returns": np.asarray(
            _discounted_returns(rewards, wrong_moves, task["gamma"]),
            dtype=np.float32,
        ),
        "episode_reward": episode_reward,
        "completed": completed,
        "policy_steps": len(actions),
        "game_length": int(np.count_nonzero(env.state)),
        "profile": profile,
    }


def self_play_worker(
    worker_id: int,
    task_queue: Any,
    inference_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            result_queue.put(
                (
                    "result",
                    _play_episode(
                        task,
                        worker_id=worker_id,
                        inference_queue=inference_queue,
                        response_queue=response_queue,
                    ),
                )
            )
        except Exception as error:
            result_queue.put(
                (
                    "error",
                    {
                        "worker_id": worker_id,
                        "episode_id": task.get("episode_id"),
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            )


__all__ = ["self_play_worker"]
