"""Unit tests for the TicTacToe base game logic."""

import unittest

try:
    from .constants import PLAYER_O, PLAYER_X
    from .game import TicTacToeGame
except ImportError:  # pragma: no cover
    from constants import PLAYER_O, PLAYER_X
    from game import TicTacToeGame


class TicTacToeGameTests(unittest.TestCase):
    def test_initial_board(self):
        game = TicTacToeGame()
        self.assertEqual(game.board_size, 15)
        self.assertEqual(game.win_length, 5)
        self.assertEqual(game.current_player, PLAYER_X)
        self.assertIsNone(game.winner)
        self.assertEqual(game.moves_count, 0)
        self.assertIsNone(game.last_move)

    def test_valid_moves(self):
        game = TicTacToeGame()
        moves = game.get_valid_moves()
        self.assertIn((0, 0), moves)
        self.assertIn((7, 7), moves)
        self.assertEqual(len(moves), 225)

    def test_invalid_move_outside_board(self):
        game = TicTacToeGame()
        self.assertFalse(game.is_valid_move((-1, 0)))
        self.assertFalse(game.is_valid_move((15, 15)))

    def test_invalid_move_on_occupied_cell(self):
        game = TicTacToeGame()
        game.make_move((0, 0))
        self.assertFalse(game.is_valid_move((0, 0)))

    def test_turns_alternate(self):
        game = TicTacToeGame()
        game.make_move((1, 1))
        self.assertEqual(game.current_player, PLAYER_O)
        game.make_move((2, 2))
        self.assertEqual(game.current_player, PLAYER_X)

    def test_horizontal_win(self):
        game = TicTacToeGame()
        for col in range(5):
            game.make_move((0, col))
            if col < 4:
                game.make_move((1, col))
        self.assertEqual(game.get_result(), PLAYER_X)

    def test_vertical_win(self):
        game = TicTacToeGame()
        for row in range(5):
            game.make_move((row, 0))
            if row < 4:
                game.make_move((row, 1))
        self.assertEqual(game.get_result(), PLAYER_X)

    def test_diagonal_down_win(self):
        game = TicTacToeGame()
        sequence = [(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (0, 3), (3, 3), (0, 4), (4, 4)]
        for index, (row, col) in enumerate(sequence):
            player = PLAYER_X if index % 2 == 0 else PLAYER_O
            self.assertTrue(game.make_move((row, col), player=player))
        self.assertEqual(game.get_result(), PLAYER_X)

    def test_diagonal_up_win(self):
        game = TicTacToeGame()
        sequence = [(4, 0), (0, 0), (3, 1), (0, 1), (2, 2), (0, 2), (1, 3), (0, 3), (0, 4)]
        for index, (row, col) in enumerate(sequence):
            player = PLAYER_X if index % 2 == 0 else PLAYER_O
            self.assertTrue(game.make_move((row, col), player=player))
        self.assertEqual(game.get_result(), PLAYER_X)

    def test_no_winner_yet(self):
        game = TicTacToeGame()
        sequence = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        for row, col in sequence:
            game.make_move((row, col))
        self.assertIsNone(game.get_result())

    def test_draw_game(self):
        game = TicTacToeGame()
        board = [[0 for _ in range(game.board_size)] for _ in range(game.board_size)]
        for row in range(game.board_size):
            for col in range(game.board_size):
                board[row][col] = PLAYER_X if (row + col) % 2 == 0 else PLAYER_O
        game.board = board
        game.moves_count = game.board_size * game.board_size
        self.assertTrue(game.is_draw())
        self.assertEqual(game.get_result(), 0)

    def test_cannot_play_after_terminal_state(self):
        game = TicTacToeGame()
        for index in range(5):
            game.make_move((0, index))
            if index < 4:
                game.make_move((1, index))
        self.assertTrue(game.is_terminal())
        self.assertFalse(game.is_valid_move((0, 5)))

    def test_copy_independent_state(self):
        game = TicTacToeGame()
        game.make_move((0, 0))
        copy_game = game.copy()
        copy_game.make_move((0, 1))
        self.assertEqual(game.board[0][1], 0)
        self.assertEqual(copy_game.board[0][1], PLAYER_O)

    def test_many_consecutive_stones(self):
        game = TicTacToeGame()
        sequence = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5)]
        for row, col in sequence:
            game.make_move((row, col))
        self.assertEqual(game.winner, PLAYER_X)


if __name__ == "__main__":
    unittest.main()
