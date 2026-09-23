"""Small, literal, nonterminal Connect Four regression corpus."""

from __future__ import annotations

from connect4_ai.env import ConnectFourEnv


TACTICAL_POSITIONS = {
    "empty": [],
    "immediate_vertical_win": [1, 4, 1, 4, 1, 5],
    "forced_vertical_block": [1, 4, 2, 4, 3, 4],
    "full_first_column": [1, 1, 1, 1, 1, 1],
    "fallback_after_center_opening": [4],
}


REGRESSION_ACTION_SEQUENCES = [
    [],
    [2],
    [3, 3],
    [1, 1, 5],
    [3, 5, 3, 5],
    [2, 2, 5, 5, 6],
    [1, 2, 5, 5, 5, 2],
    [2, 7, 3, 1, 5, 2, 6],
    [3, 3, 3, 4, 3, 6, 5, 3],
    [2, 1, 1, 3, 7, 3, 5, 5, 5],
    [5, 2, 6, 2, 3, 1, 5, 1, 6, 6],
    [6, 1, 3, 3, 6, 2, 5, 2, 1, 2, 6],
    [1, 6, 4, 2, 7, 5, 4, 1, 2, 4, 3, 1],
    [7, 6, 1, 5, 4, 1, 5, 5, 7, 2, 3, 5, 4, 2],
    [5, 3, 6, 1, 3, 7, 3, 5, 7, 2, 6, 7, 1, 1, 7, 1],
    [3, 4, 6, 3, 7, 1, 7, 4, 2, 5, 2, 1, 4, 7, 6, 6, 6, 6],
    [4, 7, 6, 1, 6, 5, 7, 1, 7, 2, 1, 7, 3, 2, 2, 7, 1, 3, 1, 3],
    [6, 5, 3, 4, 6, 6, 1, 4, 4, 2, 2, 1, 5, 5, 4, 3, 6, 3, 7, 1, 5, 1],
    [6, 7, 5, 2, 3, 2, 5, 6, 2, 3, 7, 2, 3, 6, 1, 2, 1, 4, 3, 7, 2, 4, 7, 7],
    [4, 3, 4, 1, 1, 6, 4, 2, 1, 6, 6, 5, 6, 3, 2, 4, 6, 7, 3, 5, 1, 4, 7, 2, 2, 3],
    [2, 2, 7, 6, 6, 1, 3, 1, 2, 5, 1, 6, 5, 7, 6, 4, 1, 1, 3, 5, 3, 2, 7, 4, 6, 6, 2, 1],
    [1, 6, 4, 3, 4, 2, 7, 5, 6, 6, 2, 7, 4, 3, 3, 5, 6, 4, 1, 4, 7, 6, 6, 3, 3, 2, 2, 7, 5, 7],
]


def make_env(actions: list[int], *, seed: int = 0) -> ConnectFourEnv:
    env = ConnectFourEnv(seed=seed)
    for action in actions:
        env.step(action)
    if env.done:
        raise ValueError(f"regression position is terminal: {actions}")
    return env
