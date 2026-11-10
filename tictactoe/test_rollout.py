"""Tests for configurable rollout move-selection policies."""

import unittest

from .constants import PLAYER_O, PLAYER_X
from .game import TicTacToeGame
from .rollout import heuristic_rollout, random_rollout
from .ai_controller import AIController


class RolloutPolicyTests(unittest.TestCase):
    def test_random_rollout_returns_a_legal_move(self):
        game = TicTacToeGame()
        move = random_rollout(game)
        self.assertIn(move, game.get_valid_moves())

    def test_heuristic_rollout_takes_an_immediate_win(self):
        game = TicTacToeGame()
        game.board[7][3:7] = [PLAYER_X] * 4
        game.moves_count = 4
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 2), (7, 7)})
        self.assertTrue(all(value == PLAYER_X for value in game.board[7][3:7]))

    def test_heuristic_rollout_blocks_an_immediate_win(self):
        game = TicTacToeGame()
        game.board[7][3:7] = [PLAYER_O] * 4
        game.moves_count = 4
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 2), (7, 7)})
        self.assertTrue(all(value == PLAYER_O for value in game.board[7][3:7]))

    def test_heuristic_rollout_returns_none_when_no_moves(self):
        game = TicTacToeGame(board_size=5, win_length=5)
        for row in range(5):
            for col in range(5):
                game.board[row][col] = PLAYER_X if (row + col) % 2 == 0 else PLAYER_O
        game.moves_count = 25

        self.assertIsNone(heuristic_rollout(game))

    def test_controller_configures_heuristic_as_mcts_rollout(self):
        controller = AIController(algorithm="mcts", rollout_policy="heuristic")
        self.assertEqual(controller.algorithm, "mcts")
        self.assertIs(controller.rollout_policy, heuristic_rollout)

        self.assertIsNone(controller.get_move(TicTacToeGame()))
        self.assertIn("selected rollout policy: heuristic", controller.last_status)


if __name__ == "__main__":
    unittest.main()
