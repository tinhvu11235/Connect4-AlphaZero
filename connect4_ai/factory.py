"""Single entry point for constructing any of the eight Connect Four agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from connect4_ai.agents import (
    AlphaBetaAgent,
    AlphaGoAgent,
    AlphaZeroAgent,
    DepthMinimaxAgent,
    MCTSAgent,
    PositionEvalAgent,
    RandomAgent,
    RuleBasedThink3Agent,
)
from connect4_ai.config import AGENT_CHECKPOINTS


AGENT_NAMES = (
    "RandomAgent",
    "RuleBasedThink3Agent",
    "DepthMinimaxAgent",
    "AlphaBetaAgent",
    "PositionEvalAgent",
    "MCTSAgent",
    "AlphaGoAgent",
    "AlphaZeroAgent",
)


def _reject_unknown(options: dict[str, Any]) -> None:
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"unsupported agent options: {names}")


def create_agent(
    name: str,
    *,
    seed: int | None = None,
    device: str = "cuda",
    allow_cpu: bool = False,
    **options: Any,
) -> Any:
    """Create one configured agent by its public class name."""

    normalized = name.strip().lower()
    if normalized == "randomagent":
        _reject_unknown(options)
        return RandomAgent(seed=seed)
    if normalized == "rulebasedthink3agent":
        _reject_unknown(options)
        return RuleBasedThink3Agent(seed=seed)
    if normalized == "depthminimaxagent":
        depth = int(options.pop("depth", 3))
        _reject_unknown(options)
        return DepthMinimaxAgent(depth=depth, seed=seed)
    if normalized == "alphabetaagent":
        depth = int(options.pop("depth", 3))
        _reject_unknown(options)
        return AlphaBetaAgent(depth=depth, seed=seed)
    if normalized == "positionevalagent":
        checkpoint = Path(
            options.pop("checkpoint", AGENT_CHECKPOINTS.position_eval)
        )
        depth = int(options.pop("depth", 3))
        _reject_unknown(options)
        return PositionEvalAgent.from_checkpoint(
            checkpoint,
            depth=depth,
            device=device,
            allow_cpu=allow_cpu,
        )
    if normalized == "mctsagent":
        rollouts = int(options.pop("num_rollouts", 100))
        exploration = float(options.pop("exploration_constant", 1.4))
        _reject_unknown(options)
        return MCTSAgent(
            num_rollouts=rollouts,
            exploration_constant=exploration,
            seed=seed,
        )
    if normalized == "alphagoagent":
        policy = Path(
            options.pop("policy_checkpoint", AGENT_CHECKPOINTS.policy_gradient)
        )
        value = Path(options.pop("value_checkpoint", AGENT_CHECKPOINTS.value))
        rollout_policy = Path(
            options.pop(
                "rollout_policy_checkpoint", AGENT_CHECKPOINTS.fast_policy
            )
        )
        weight = float(options.pop("weight", 0.75))
        depth = int(options.pop("depth", 45))
        policy_rollout = bool(options.pop("policy_rollout", False))
        rollouts = int(options.pop("num_rollouts", 584))
        _reject_unknown(options)
        return AlphaGoAgent.from_checkpoints(
            policy_checkpoint=policy,
            value_checkpoint=value,
            rollout_policy_checkpoint=rollout_policy,
            weight=weight,
            depth=depth,
            policy_rollout=policy_rollout,
            num_rollouts=rollouts,
            seed=seed,
            device=device,
            allow_cpu=allow_cpu,
        )
    if normalized == "alphazeroagent":
        checkpoint = Path(
            options.pop("checkpoint", AGENT_CHECKPOINTS.alphazero_book)
        )
        weight = float(options.pop("weight", 0.75))
        rollouts = int(options.pop("num_rollouts", 584))
        _reject_unknown(options)
        return AlphaZeroAgent.from_checkpoint(
            checkpoint,
            weight=weight,
            num_rollouts=rollouts,
            seed=seed,
            device=device,
            allow_cpu=allow_cpu,
        )
    raise ValueError(
        f"unknown agent {name!r}; expected one of: {', '.join(AGENT_NAMES)}"
    )


__all__ = ["AGENT_NAMES", "create_agent"]
