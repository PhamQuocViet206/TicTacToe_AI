"""Lightweight GUI tests for the Tkinter interface."""

import os
import unittest
from unittest.mock import patch

try:
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None

try:
    from .gui import TicTacToeGUI
except ImportError:  # pragma: no cover
    from gui import TicTacToeGUI


@unittest.skipIf(tk is None, "Tkinter is not available.")
class TicTacToeGUITests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
            self.skipTest("Tkinter GUI requires an X display in non-Windows environments.")
        self.root = tk.Tk()
        self.root.withdraw()
        self.gui = TicTacToeGUI(self.root)

    def tearDown(self):
        self.gui._stop_ai_matches()
        self.gui._cancel_scheduled_callback()
        self.root.destroy()

    def test_initial_board_ready(self):
        self.assertEqual(self.gui.game.board_size, 15)
        self.assertEqual(self.gui.game.current_player, 1)

    def test_click_updates_board(self):
        self.gui._handle_cell_click(3, 4)
        self.assertEqual(self.gui.game.board[3][4], 1)
        self.assertEqual(self.gui.game.current_player, -1)

    def test_reset_game(self):
        self.gui._handle_cell_click(0, 0)
        self.gui._reset_game()
        self.assertEqual(self.gui.game.moves_count, 0)
        self.assertIsNone(self.gui.game.last_move)

    def test_selects_an_ai_for_each_player(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.ai_var.set("random")
        self.gui.ai_o_var.set("mcts")
        self.gui.game.current_player = 1
        x_controller = self.gui._controller_for_current_player()
        self.gui.game.current_player = -1
        o_controller = self.gui._controller_for_current_player()

        self.assertEqual(x_controller.algorithm, "random")
        self.assertEqual(o_controller.algorithm, "mcts")

    def test_completed_ai_match_updates_score(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.ai_var.set("random")
        self.gui.ai_o_var.set("random")
        self.gui.game.board[7][3:7] = [1] * 4
        self.gui.game.moves_count = 8
        self.gui.game.current_player = 1
        self.gui.batch_active = True
        self.gui.batch_target = 1

        self.gui._apply_move((7, 7))

        self.assertEqual(self.gui.batch_completed, 1)
        self.assertEqual(self.gui.batch_x_wins, 1)
        self.assertFalse(self.gui.batch_active)
        self.assertIn("100.0%", self.gui.batch_stats_var.get())

    def test_start_ai_matches_uses_requested_game_count(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.batch_games_var.set("3")

        with patch.object(self.gui, "_start_next_ai_match") as start_match:
            self.gui.start_ai_matches()

        self.assertTrue(self.gui.batch_active)
        self.assertEqual(self.gui.batch_target, 3)
        start_match.assert_called_once_with()

    def test_start_ai_matches_rejects_non_positive_count(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.batch_games_var.set("0")

        self.gui.start_ai_matches()

        self.assertFalse(self.gui.batch_active)
        self.assertIn("positive", self.gui.status_var.get())

    def test_ai_match_schedules_the_next_ai_turn(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.batch_active = True

        with patch.object(self.gui, "_schedule_ai_turn") as schedule_next:
            self.gui._apply_move((7, 7))

        schedule_next.assert_called_once_with()

    def test_winning_line_detection(self):
        for col in range(5):
            self.gui.game.board[7][col] = 1
        self.gui.game.winner = 1
        line = self.gui._get_winning_line()
        self.assertIsNotNone(line)
        self.assertEqual(len(line), 5)
        self.assertEqual(line, [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4)])

    def test_hover_motion_and_leave(self):
        from unittest.mock import Mock

        # Hover over an empty cell
        event = Mock(x=self.gui.margin + 4 * self.gui.cell_size + 10, y=self.gui.margin + 4 * self.gui.cell_size + 10)
        self.gui._on_canvas_motion(event)
        self.assertEqual(self.gui.hover_cell, (4, 4))

        # Leaving clears hover
        self.gui._on_canvas_leave()
        self.assertIsNone(self.gui.hover_cell)

    def test_toggle_advanced_settings(self):
        initial = self.gui.advanced_expanded
        self.gui._toggle_advanced_settings()
        self.assertNotEqual(self.gui.advanced_expanded, initial)
        self.gui._toggle_advanced_settings()
        self.assertEqual(self.gui.advanced_expanded, initial)

    def test_separate_advanced_ai_settings_for_x_and_o(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.ai_var.set("mcts")
        self.gui.ai_iterations_var.set("500")
        self.gui.uct_var.set("1.25")
        self.gui.rollout_var.set("random")

        self.gui.ai_o_var.set("mcts")
        self.gui.ai_o_iterations_var.set("2500")
        self.gui.uct_o_var.set("2.10")
        self.gui.rollout_o_var.set("heuristic")

        # Check Player X controller
        self.gui.game.current_player = 1
        ctrl_x = self.gui._controller_for_current_player()
        self.assertEqual(ctrl_x.iterations, 500)
        self.assertAlmostEqual(ctrl_x.uct_c, 1.25)
        self.assertEqual(ctrl_x.rollout_policy_name, "random")

        # Check Player O controller
        self.gui.game.current_player = -1
        ctrl_o = self.gui._controller_for_current_player()
        self.assertEqual(ctrl_o.iterations, 2500)
        self.assertAlmostEqual(ctrl_o.uct_c, 2.10)
        self.assertEqual(ctrl_o.rollout_policy_name, "heuristic")

    def test_ai_move_timing_and_average_calculation(self):
        self.gui.mode_var.set("AI vs AI")
        self.gui.ai_var.set("random")
        self.gui.ai_o_var.set("random")
        self.gui.batch_active = True
        self.gui.game.current_player = 1

        self.gui._run_ai_turn()

        # AI X move duration should have been recorded
        self.assertEqual(len(self.gui.ai_x_move_times), 1)
        self.assertGreater(self.gui.ai_x_move_times[0], 0.0)

        avg_x, avg_o = self.gui.get_average_move_times()
        self.assertGreater(avg_x, 0.0)
        self.assertEqual(avg_o, 0.0)

        # Result stats should include the average time
        stats = self.gui.batch_stats_var.get()
        self.assertIn("Thời gian TB", stats)

    def test_separate_advanced_toggles_for_x_and_o(self):
        self.gui.mode_var.set("AI vs AI")
        self.assertFalse(self.gui.advanced_x_expanded)
        self.assertFalse(self.gui.advanced_o_expanded)

        self.gui._toggle_advanced_x()
        self.assertTrue(self.gui.advanced_x_expanded)
        self.assertFalse(self.gui.advanced_o_expanded)

        self.gui._toggle_advanced_o()
        self.assertTrue(self.gui.advanced_x_expanded)
        self.assertTrue(self.gui.advanced_o_expanded)

        self.gui._toggle_advanced_x()
        self.assertFalse(self.gui.advanced_x_expanded)
        self.assertTrue(self.gui.advanced_o_expanded)


if __name__ == "__main__":
    unittest.main()
