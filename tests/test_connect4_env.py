from __future__ import annotations

import unittest

import numpy as np

from connect4_ai.env.connect4 import (
    ConnectFourEnv,
    GameAlreadyFinishedError,
    InvalidActionError,
)


class ConnectFourEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ConnectFourEnv(seed=123)

    def play(self, actions: list[int]) -> ConnectFourEnv:
        for action in actions:
            self.env.step(action)
        return self.env

    def test_initial_state_preserves_book_representation(self) -> None:
        self.assertEqual((7, 6), self.env.state.shape)
        self.assertEqual(np.dtype(np.int64), self.env.state.dtype)
        self.assertEqual((7, 6), self.env.observation_space.shape)
        self.assertEqual(7, self.env.action_space.n)
        self.assertTrue(np.array_equal(np.zeros((7, 6)), self.env.state))
        self.assertEqual("red", self.env.turn)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], self.env.validinputs)
        self.assertEqual(0, self.env.reward)
        self.assertFalse(self.env.done)

    def test_action_zero_is_rejected_without_mutation(self) -> None:
        before = self.env.state.copy()
        with self.assertRaisesRegex(InvalidActionError, "1..7"):
            self.env.step(0)
        self.assertTrue(np.array_equal(before, self.env.state))
        self.assertEqual("red", self.env.turn)

    def test_action_eight_is_rejected_without_mutation(self) -> None:
        before = self.env.state.copy()
        with self.assertRaisesRegex(InvalidActionError, "1..7"):
            self.env.step(8)
        self.assertTrue(np.array_equal(before, self.env.state))

    def test_full_column_is_removed_and_rejected(self) -> None:
        self.play([1, 1, 1, 1, 1, 1])
        self.assertNotIn(1, self.env.validinputs)
        before = self.env.state.copy()
        with self.assertRaisesRegex(InvalidActionError, "column 1 is full"):
            self.env.step(1)
        self.assertTrue(np.array_equal(before, self.env.state))

    def test_player_switching(self) -> None:
        self.env.step(4)
        self.assertEqual("yellow", self.env.turn)
        self.env.step(5)
        self.assertEqual("red", self.env.turn)
        self.assertEqual(1, self.env.state[3, 0])
        self.assertEqual(-1, self.env.state[4, 0])

    def test_red_vertical_win(self) -> None:
        self.play([1, 2, 1, 2, 1, 2, 1])
        self.assertTrue(self.env.done)
        self.assertEqual(1, self.env.reward)
        self.assertEqual("red", self.env.turn)
        self.assertEqual([], self.env.validinputs)

    def test_yellow_vertical_win_and_reward_sign(self) -> None:
        self.play([1, 2, 1, 2, 3, 2, 3, 2])
        self.assertTrue(self.env.done)
        self.assertEqual(-1, self.env.reward)
        self.assertEqual("yellow", self.env.turn)

    def test_horizontal_win(self) -> None:
        self.play([1, 1, 2, 2, 3, 3, 4])
        self.assertTrue(self.env.done)
        self.assertEqual(1, self.env.reward)
        self.assertTrue(np.array_equal([1, 1, 1, 1], self.env.state[0:4, 0]))

    def test_forward_diagonal_win(self) -> None:
        self.play([1, 2, 2, 3, 7, 3, 3, 4, 7, 4, 7, 4, 4])
        self.assertTrue(self.env.done)
        self.assertEqual(1, self.env.reward)
        self.assertEqual([1, 1, 1, 1], [self.env.state[i, i] for i in range(4)])

    def test_back_diagonal_win(self) -> None:
        self.play([7, 6, 6, 5, 1, 5, 5, 4, 1, 4, 1, 4, 4])
        self.assertTrue(self.env.done)
        self.assertEqual(1, self.env.reward)
        self.assertEqual(
            [1, 1, 1, 1],
            [self.env.state[column, 6 - column] for column in range(3, 7)],
        )

    def test_draw(self) -> None:
        draw_actions = [
            2, 3, 6, 4, 1, 2, 6, 3, 5, 5, 6, 5, 4, 6,
            1, 7, 3, 4, 1, 7, 6, 6, 2, 1, 7, 1, 1, 7,
            4, 2, 5, 4, 5, 5, 2, 2, 7, 7, 4, 3, 3, 3,
        ]
        self.play(draw_actions)
        self.assertTrue(self.env.done)
        self.assertEqual(0, self.env.reward)
        self.assertEqual([], self.env.validinputs)
        self.assertEqual(42, int(np.count_nonzero(self.env.state)))

    def test_reset_restores_initial_state(self) -> None:
        self.play([4, 5, 4])
        state = self.env.reset(seed=456)
        self.assertIs(state, self.env.state)
        self.assertTrue(np.array_equal(np.zeros((7, 6)), state))
        self.assertEqual("red", self.env.turn)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], self.env.validinputs)
        self.assertIsNone(self.env.game_piece)
        self.assertFalse(self.env.done)

    def test_copy_is_independent_and_preserves_rng_state(self) -> None:
        self.env.step(4)
        copied = self.env.copy()
        self.assertEqual(self.env.sample(), copied.sample())
        copied.step(5)
        self.assertEqual(0, self.env.state[4, 0])
        self.assertEqual(-1, copied.state[4, 0])

    def test_move_after_terminal_state_is_rejected(self) -> None:
        self.play([1, 2, 1, 2, 1, 2, 1])
        with self.assertRaises(GameAlreadyFinishedError):
            self.env.step(3)


if __name__ == "__main__":
    unittest.main()
