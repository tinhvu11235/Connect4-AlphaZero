"""Canonical Connect Four environment derived from ``utils/conn_simple_env.py``.

The representation deliberately follows the book rather than the more common
row-major convention:

* board shape: ``(7, 6)`` and indexing: ``state[column, row]``
* bottom row: row index 0
* legal actions: integers 1 through 7
* red pieces: +1; yellow pieces: -1
* rewards: red win +1, yellow win -1, draw 0

The terminal player is intentionally not switched after a win or draw. This
matches the original environment and is relied on by the book's search code.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
import random
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


Board: TypeAlias = NDArray[np.int64]


class InvalidActionError(ValueError):
    """Raised when an action is outside the current legal-move set."""


class GameAlreadyFinishedError(RuntimeError):
    """Raised when a move is attempted after a terminal state."""


@dataclass(frozen=True)
class ActionSpace:
    """Small compatibility object matching the original ``action_space``."""

    n: int


@dataclass(frozen=True, init=False)
class ObservationSpace:
    """Small compatibility object matching the original ``observation_space``."""

    shape: tuple[int, int]

    def __init__(
        self, row: int | tuple[int, int], col: int | None = None
    ) -> None:
        if isinstance(row, tuple):
            shape = row
        elif col is not None:
            shape = (row, col)
        else:
            raise TypeError("col is required when row is an integer")
        object.__setattr__(self, "shape", shape)


class ConnectFourEnv:
    """Headless, copyable Connect Four environment with validated actions."""

    COLUMNS = 7
    ROWS = 6
    RED = "red"
    YELLOW = "yellow"

    def __init__(self, seed: int | None = None) -> None:
        self.action_space = ActionSpace(self.COLUMNS)
        self.observation_space = ObservationSpace((self.COLUMNS, self.ROWS))
        self.info = ""
        self._rng = random.Random(seed)
        self.game_piece: list[int | str] | None = None
        self.reset()

    def seed(self, seed: int | None) -> None:
        """Seed this environment's legal-move sampler."""

        self._rng.seed(seed)

    def sample(self) -> int:
        """Return a uniformly sampled legal action."""

        if not self.validinputs:
            raise InvalidActionError("cannot sample an action: no legal moves remain")
        return self._rng.choice(self.validinputs)

    def reset(self, *, seed: int | None = None) -> Board:
        """Reset the game and return the canonical ``(7, 6)`` board."""

        if seed is not None:
            self.seed(seed)
        self.turn = self.RED
        self.validinputs = list(range(1, self.COLUMNS + 1))
        self.occupied: list[list[str]] = [[] for _ in range(self.COLUMNS)]
        # NumPy's default integer dtype matches the original implementation.
        self.state = np.zeros((self.COLUMNS, self.ROWS), dtype=np.int64)
        self.done = False
        self.reward = 0
        self.game_piece = None
        return self.state

    def copy(self) -> ConnectFourEnv:
        """Return an independent environment including sampler state."""

        return deepcopy(self)

    def step(self, action: int) -> tuple[Board, int, bool, str]:
        """Apply one legal action, rejecting invalid input before mutation."""

        action = self._validate_action(action)
        column = action - 1
        row = len(self.occupied[column])
        piece_value = 1 if self.turn == self.RED else -1

        self.game_piece = [column, row, self.turn]
        self.state[column, row] = piece_value
        self.occupied[column].append(self.turn)

        if len(self.occupied[column]) == self.ROWS:
            self.validinputs.remove(action)

        if self.win_game(action):
            self.done = True
            self.reward = piece_value
            self.validinputs = []
        elif not self.validinputs:
            self.done = True
            self.reward = 0
        else:
            self.turn = self.YELLOW if self.turn == self.RED else self.RED

        return self.state, self.reward, self.done, self.info

    def _validate_action(self, action: int) -> int:
        if self.done:
            raise GameAlreadyFinishedError("cannot play a move: the game is finished")
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise InvalidActionError(
                f"action must be an integer in 1..{self.COLUMNS}; got {action!r}"
            )
        action = int(action)
        if action < 1 or action > self.COLUMNS:
            raise InvalidActionError(
                f"action must be in 1..{self.COLUMNS}; got {action}"
            )
        if action not in self.validinputs:
            raise InvalidActionError(f"column {action} is full and is not a legal move")
        return action

    def _line_length(self, x: int, y: int, dx: int, dy: int) -> int:
        piece = self.state[x, y]
        if piece == 0:
            return 0

        count = 1
        for direction in (-1, 1):
            cx = x + direction * dx
            cy = y + direction * dy
            while (
                0 <= cx < self.COLUMNS
                and 0 <= cy < self.ROWS
                and self.state[cx, cy] == piece
            ):
                count += 1
                cx += direction * dx
                cy += direction * dy
        return count

    def horizontal4(self, x: int, y: int) -> bool:
        return self._line_length(x, y, 1, 0) >= 4

    def vertical4(self, x: int, y: int) -> bool:
        return self._line_length(x, y, 0, 1) >= 4

    def forward4(self, x: int, y: int) -> bool:
        """Check a diagonal rising from lower-left to upper-right (``/``)."""

        return self._line_length(x, y, 1, 1) >= 4

    def back4(self, x: int, y: int) -> bool:
        """Check a diagonal falling from upper-left to lower-right (``\\``)."""

        return self._line_length(x, y, 1, -1) >= 4

    def win_game(self, action: int) -> bool:
        """Return whether the most recent piece in ``action`` completed four."""

        column = action - 1
        row = len(self.occupied[column]) - 1
        return any(
            check(column, row)
            for check in (
                self.vertical4,
                self.horizontal4,
                self.forward4,
                self.back4,
            )
        )


# Compatibility names used by the original notebooks and helper modules.
action_space = ActionSpace
observation_space = ObservationSpace
conn = ConnectFourEnv


__all__ = [
    "ActionSpace",
    "Board",
    "ConnectFourEnv",
    "GameAlreadyFinishedError",
    "InvalidActionError",
    "ObservationSpace",
    "action_space",
    "conn",
    "observation_space",
]
