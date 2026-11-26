"""Tests for configurable rollout move-selection policies."""

import unittest

from .ai_controller import AIController
from .constants import PLAYER_O, PLAYER_X
from .game import TicTacToeGame
from .mcts import MCTSAgent
from .rollout import (
    _analyze_direction,
    _evaluate_move,
    _get_candidate_moves,
    _is_winning_move,
    _score_direction,
    heuristic_rollout,
    random_rollout,
)


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

        # MCTS is integrated: on an empty 15x15 board the only candidate is the centre (MC-AC8.3).
        self.assertEqual(controller.get_move(TicTacToeGame()), (7, 7))
        self.assertIn("rollout=heuristic", controller.last_status)

    def test_prefers_own_immediate_win_over_blocking_opponent(self):
        game = TicTacToeGame()
        # X has 4-in-a-row on row 7: (7, 3)..(7, 6) -> wins at (7, 2) or (7, 7)
        game.board[7][3:7] = [PLAYER_X] * 4
        # O also has 4-in-a-row on row 5: (5, 3)..(5, 6) -> threatens (5, 2) and (5, 7)
        game.board[5][3:7] = [PLAYER_O] * 4
        game.moves_count = 8
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 2), (7, 7)})

    def test_multiple_opponent_threats_chooses_highest_value_block(self):
        game = TicTacToeGame()
        # O has an open-4 at row 7: (7, 4)..(7, 7) -> two winning threats: (7, 3) and (7, 8)
        game.board[7][4:8] = [PLAYER_O] * 4
        # X has two stones near (7, 3) at (5, 3) and (6, 3), making (7, 3) also build an X vertical chain of 3
        game.board[5][3] = PLAYER_X
        game.board[6][3] = PLAYER_X
        game.moves_count = 6
        game.current_player = PLAYER_X

        self.assertEqual(heuristic_rollout(game), (7, 3))

    def test_extends_open_three_to_open_four(self):
        game = TicTacToeGame()
        # X has an open 3 at (7, 5), (7, 6), (7, 7)
        for col in (5, 6, 7):
            game.board[7][col] = PLAYER_X
        game.board[2][2] = PLAYER_O
        game.moves_count = 4
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 4), (7, 8)})

    def test_blocks_opponent_open_three(self):
        game = TicTacToeGame()
        # O has an open 3 at (7, 5), (7, 6), (7, 7); X only has an isolated stone
        for col in (5, 6, 7):
            game.board[7][col] = PLAYER_O
        game.board[2][2] = PLAYER_X
        game.moves_count = 4
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 4), (7, 8)})

    def test_prioritizes_double_three_fork_intersection(self):
        game = TicTacToeGame()
        # Intersection at (7, 7):
        # Horizontal X stones at (7, 5), (7, 6) and vertical X stones at (5, 7), (6, 7)
        game.board[7][5] = PLAYER_X
        game.board[7][6] = PLAYER_X
        game.board[5][7] = PLAYER_X
        game.board[6][7] = PLAYER_X
        game.moves_count = 4
        game.current_player = PLAYER_X

        # Playing (7, 7) creates two open-3 lines simultaneously (1000 + 1000 = 2000 attack points)
        self.assertEqual(heuristic_rollout(game), (7, 7))

    def test_dead_corridor_shorter_than_win_length_scores_zero(self):
        game = TicTacToeGame()
        # Corridor on row 7 between O at (7, 2) and O at (7, 7) has only 4 cells: (7, 3), (7, 4), (7, 5), (7, 6)
        game.board[7][2] = PLAYER_O
        game.board[7][3] = PLAYER_X
        game.board[7][4] = PLAYER_X
        game.board[7][7] = PLAYER_O
        game.moves_count = 4

        # Placing X at (7, 5) makes 3-in-a-row with (7, 6) empty, but total corridor length is only 4 (< 5)
        count, open_ends = _analyze_direction(game, 7, 5, 0, 1, PLAYER_X)
        self.assertEqual(count, 3)
        self.assertEqual(open_ends, 0)
        self.assertEqual(_score_direction(count, open_ends, game.win_length, is_attack=True), 0)

    def test_heuristic_rollout_does_not_mutate_game(self):
        game = TicTacToeGame()
        for move in ((7, 7), (7, 8), (6, 7), (8, 8), (5, 7)):
            game.make_move(move)
        board_before = [row[:] for row in game.board]
        player_before = game.current_player
        moves_before = game.moves_count
        last_before = game.last_move

        chosen = heuristic_rollout(game)
        self.assertIn(chosen, game.get_valid_moves())
        self.assertEqual(game.board, board_before)
        self.assertEqual(game.current_player, player_before)
        self.assertEqual(game.moves_count, moves_before)
        self.assertEqual(game.last_move, last_before)

    def test_3x3_board_win_and_block(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((0, 0))  # X
        game.make_move((1, 0))  # O
        game.make_move((0, 1))  # X
        # O to move: must block (0, 2)
        self.assertEqual(heuristic_rollout(game), (0, 2))

        game.make_move((2, 0))  # O plays elsewhere; X to move can win at (0, 2)
        self.assertEqual(heuristic_rollout(game), (0, 2))

    def test_empty_large_board_chooses_center(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        self.assertEqual(_get_candidate_moves(game), [(7, 7)])
        self.assertEqual(heuristic_rollout(game), (7, 7))

    def test_mcts_agent_works_with_both_rollouts(self):
        game = TicTacToeGame(board_size=5, win_length=5)
        game.make_move((2, 2))
        game.make_move((2, 3))

        for policy in (random_rollout, heuristic_rollout):
            agent = MCTSAgent(iterations=30, uct_c=1.4, rollout_policy=policy, seed=42)
            move = agent.get_move(game)
            self.assertIn(move, game.get_valid_moves())
            self.assertEqual(agent.last_stats["iterations"], 30)

    def test_benchmark_rollout_modes(self):
        from .benchmark_rollout import run_iteration_matchup, run_time_budget_matchup

        eq_row = run_iteration_matchup(3, 3, 20, 20, games=2, seed=10)
        self.assertEqual(eq_row["games"], 2)
        self.assertEqual(eq_row["heur_W"] + eq_row["draws"] + eq_row["heur_L"], 2)

        tb_row = run_time_budget_matchup(5, 5, 0.01, games=2, seed=10)
        self.assertEqual(tb_row["games"], 2)
        self.assertEqual(tb_row["heur_W"] + tb_row["draws"] + tb_row["heur_L"], 2)
        self.assertGreater(tb_row["rand_iter"], 0)
        self.assertGreater(tb_row["heur_iter"], 0)


if __name__ == "__main__":
    unittest.main()
