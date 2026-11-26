from __future__ import annotations

import unittest
from unittest.mock import patch

from .ai_controller import AIController
from .constants import PLAYER_O, PLAYER_X
from .game import TicTacToeGame
from . import minimax as minimax_module
from .minimax import MinimaxAI


class TestMinimax(unittest.TestCase):
    def test_immediate_win_is_selected(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((0, 0), PLAYER_X)
        game.make_move((1, 0), PLAYER_O)
        game.make_move((0, 1), PLAYER_X)
        game.make_move((1, 1), PLAYER_O)

        move = MinimaxAI(depth=6).get_move(game)
        self.assertEqual(move, (0, 2))

    def test_blocks_immediate_opponent_win(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((0, 0), PLAYER_X)
        game.make_move((1, 0), PLAYER_O)
        game.make_move((2, 2), PLAYER_X)
        game.make_move((1, 1), PLAYER_O)

        move = MinimaxAI(depth=6).get_move(game)
        self.assertEqual(move, (1, 2))

    def test_search_does_not_modify_game(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((1, 1), PLAYER_X)
        board_before = [row[:] for row in game.board]
        player_before = game.current_player

        MinimaxAI(depth=6).get_move(game)

        self.assertEqual(game.board, board_before)
        self.assertEqual(game.current_player, player_before)

    def test_terminal_game_returns_none(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((0, 0), PLAYER_X)
        game.make_move((1, 0), PLAYER_O)
        game.make_move((0, 1), PLAYER_X)
        game.make_move((1, 1), PLAYER_O)
        game.make_move((0, 2), PLAYER_X)

        self.assertIsNone(MinimaxAI(depth=6).get_move(game))

    def test_controller_integrates_minimax(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        controller = AIController(algorithm="minimax", depth=6)
        move = controller.get_move(game)
        self.assertIn(move, game.get_valid_moves())
        self.assertIsInstance(controller.last_agent, MinimaxAI)
        self.assertIn("Minimax chose", controller.last_status)

    def test_pruning_methods_are_callable(self):
        """Regression: the bool attribute must not shadow the alpha_beta() method."""
        game = TicTacToeGame(board_size=3, win_length=3)
        ai = MinimaxAI()
        for name in ("alpha_beta", "alphabeta", "_alphabeta", "minimax", "_minimax"):
            self.assertIsInstance(getattr(ai, name)(game, 2), (int, float), name)

    def test_node_counts_full_3x3(self):
        """Pure minimax visits the whole tree; alpha-beta visits far fewer nodes."""
        game = TicTacToeGame(board_size=3, win_length=3)
        plain = MinimaxAI(alpha_beta=False)
        pruned = MinimaxAI(alpha_beta=True)
        plain.get_move(game)
        pruned.get_move(game)
        self.assertFalse(plain._search_aborted)
        self.assertEqual(plain.nodes_searched, 549945)
        self.assertLess(pruned.nodes_searched, plain.nodes_searched // 50)
        self.assertGreater(pruned.pruned_nodes, 0)

    def test_default_search_terminates_on_5x5(self):
        """Regression: depth=None used to mean depth 25 with no limit (never ends)."""
        game = TicTacToeGame(board_size=5, win_length=5)
        ai = MinimaxAI()
        move = ai.get_move(game)
        self.assertIn(move, game.get_valid_moves())
        self.assertFalse(ai._search_aborted)
        self.assertLess(ai.nodes_searched, 100_000)

    def test_auto_depth_policy(self):
        game = TicTacToeGame(board_size=5, win_length=5)
        self.assertEqual(MinimaxAI()._choose_depth(game), 3)
        self.assertEqual(MinimaxAI(depth=2)._choose_depth(game), 2)
        for move in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0),
                     (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (4, 4), (4, 0)]:
            game.make_move(move)  # 10 empties left
        self.assertEqual(MinimaxAI()._choose_depth(game), 10)
        self.assertEqual(MinimaxAI(alpha_beta=False)._choose_depth(game), 3)
        game.make_move((4, 1))  # 9 empties: still depth 3 for plain minimax on 5x5
        self.assertEqual(MinimaxAI(alpha_beta=False)._choose_depth(game), 3)
        game.make_move((4, 2))  # 8 empties: exhaustive
        self.assertEqual(MinimaxAI(alpha_beta=False)._choose_depth(game), 8)
        game3 = TicTacToeGame(board_size=3, win_length=3)
        self.assertEqual(MinimaxAI()._choose_depth(game3), 9)
        self.assertEqual(MinimaxAI(alpha_beta=False)._choose_depth(game3), 9)

    def test_direct_call_side_to_move(self):
        """maximizing=True must treat the side to move as the maximizer."""
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((0, 0))  # X moved; O to move
        ai = MinimaxAI()
        ai.alpha_beta(game, 2, maximizing=True)
        self.assertEqual(ai._ai_player, PLAYER_O)
        ai.alpha_beta(game, 2, maximizing=False)
        self.assertEqual(ai._ai_player, PLAYER_X)

    def test_auto_mode_is_bounded_on_heavy_5x5_position(self):
        """Regression: an unlucky 12-empty 5x5 position took ~510k nodes (~7 s) uncapped."""
        game = TicTacToeGame(board_size=5, win_length=5)
        for move in [(4, 2), (2, 2), (3, 4), (1, 0), (1, 2), (1, 1), (1, 4),
                     (1, 3), (0, 2), (2, 0), (4, 1), (4, 0), (2, 1)]:
            self.assertTrue(game.make_move(move))
        ai = MinimaxAI()
        # Force the exhaustive branch (12 empties) to exercise the node-budget fallback.
        with patch.object(minimax_module, "FULL_SEARCH_MAX_EMPTY_AB", 12):
            move = ai.get_move(game)
        self.assertIn(move, game.get_valid_moves())
        self.assertFalse(ai._search_aborted)
        self.assertLess(ai.nodes_searched, 110_000)
        self.assertTrue(ai.last_stats["warnings"])  # reports the depth-3 fallback


if __name__ == "__main__":
    unittest.main()
