"""Tests for configurable rollout move-selection policies."""

import random
import unittest

from .ai_controller import AIController
from .constants import PLAYER_O, PLAYER_X
from .game import TicTacToeGame
from .mcts import MCTSAgent
from .rollout import (
    ATTACK_WEIGHTS,
    DEFENSE_WEIGHTS,
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

    def test_weight_tables_have_exact_six_patterns(self):
        expected_attack = {
            (4, 2): 10_000,
            (4, 1): 2_000,
            (3, 2): 800,
            (3, 1): 200,
            (2, 2): 60,
            (2, 1): 15,
        }
        expected_defense = {
            (4, 2): 8_000,
            (4, 1): 1_500,
            (3, 2): 600,
            (3, 1): 120,
            (2, 2): 30,
            (2, 1): 10,
        }
        self.assertEqual(ATTACK_WEIGHTS, expected_attack)
        self.assertEqual(DEFENSE_WEIGHTS, expected_defense)
        self.assertEqual(_score_direction(1, 2, is_attack=True), 0)
        self.assertEqual(_score_direction(3, 0, is_attack=True), 0)

    # Tình huống 1: Thắng ngay (luật 5 quân và luật 3 quân)
    def test_heuristic_rollout_takes_an_immediate_win(self):
        game = TicTacToeGame()
        game.board[7][3:7] = [PLAYER_X] * 4
        game.moves_count = 4
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 2), (7, 7)})
        self.assertTrue(all(value == PLAYER_X for value in game.board[7][3:7]))

    def test_prefers_own_immediate_win_over_blocking_opponent(self):
        game = TicTacToeGame()
        # X has 4-in-a-row on row 7: (7, 3)..(7, 6) -> wins at (7, 2) or (7, 7)
        game.board[7][3:7] = [PLAYER_X] * 4
        # O also has 4-in-a-row on row 5: (5, 3)..(5, 6) -> threatens (5, 2) and (5, 7)
        game.board[5][3:7] = [PLAYER_O] * 4
        game.moves_count = 8
        game.current_player = PLAYER_X

        self.assertIn(heuristic_rollout(game), {(7, 2), (7, 7)})

    def test_3x3_board_win_and_block(self):
        game = TicTacToeGame(board_size=3, win_length=3)
        game.make_move((0, 0))  # X
        game.make_move((1, 0))  # O
        game.make_move((0, 1))  # X
        # O to move: must block (0, 2)
        self.assertEqual(heuristic_rollout(game), (0, 2))

        game.make_move((2, 0))  # O plays elsewhere; X to move can win at (0, 2)
        self.assertEqual(heuristic_rollout(game), (0, 2))

    # Tình huống 2: Chặn đối thủ thắng ngay (1 đầu và 2 đầu)
    def test_heuristic_rollout_blocks_an_immediate_win(self):
        game = TicTacToeGame()
        # O has 4 stones blocked on left edge: (7, 0)..(7, 3) -> only (7, 4) wins for O
        game.board[7][0:4] = [PLAYER_O] * 4
        game.moves_count = 4
        game.current_player = PLAYER_X

        self.assertEqual(heuristic_rollout(game), (7, 4))

    def test_multiple_opponent_threats_chooses_valid_highest_value_block(self):
        game = TicTacToeGame()
        # O has an open-4 at row 7: (7, 4)..(7, 7) -> two winning threats: (7, 3) and (7, 8)
        game.board[7][4:8] = [PLAYER_O] * 4
        # X has two stones near (7, 3) at (5, 3) and (6, 3), making (7, 3) also build an X vertical chain of 3
        game.board[5][3] = PLAYER_X
        game.board[6][3] = PLAYER_X
        game.moves_count = 6
        game.current_player = PLAYER_X

        self.assertEqual(heuristic_rollout(game), (7, 3))

    # Tình huống 3: Phát triển chuỗi 2 -> 3 trên bàn 10x10
    def test_situation_3_extends_two_to_three_over_opponent_cluster_on_10x10(self):
        game = TicTacToeGame(board_size=10, win_length=5)
        # X has two contiguous stones at (5, 3) and (5, 4), blocked on left at (5, 2) by O
        game.board[5][2] = PLAYER_O
        game.board[5][3] = PLAYER_X
        game.board[5][4] = PLAYER_X
        # Separate cluster around (3, 7): O at (3, 6) and (4, 7), X at (2, 7)
        game.board[3][6] = PLAYER_O
        game.board[4][7] = PLAYER_O
        game.board[2][7] = PLAYER_X
        game.moves_count = 6
        game.current_player = PLAYER_X

        # (5, 5) creates (3, 1) = 200 points
        self.assertEqual(_analyze_direction(game, 5, 5, 0, 1, PLAYER_X), (3, 1))
        self.assertEqual(_evaluate_move(game, (5, 5), PLAYER_X), 200)

        # (3, 7) has vertical Attack (2, 1) = 15, vertical Defense (2, 1) = 10, horizontal Defense (2, 2) = 30 -> total = 55
        self.assertEqual(_analyze_direction(game, 3, 7, 1, 0, PLAYER_X), (2, 1))
        self.assertEqual(_analyze_direction(game, 3, 7, 1, 0, PLAYER_O), (2, 1))
        self.assertEqual(_analyze_direction(game, 3, 7, 0, 1, PLAYER_O), (2, 2))
        self.assertEqual(_evaluate_move(game, (3, 7), PLAYER_X), 55)
        self.assertGreater(_evaluate_move(game, (5, 5), PLAYER_X), _evaluate_move(game, (3, 7), PLAYER_X))

        # When O's stones around (3, 7) do not form an unblocked open-2 diagonal of their own
        # (e.g. only O at (3, 6) or (5, 3)-(5, 4) is open at both ends), heuristic_rollout selects (5, 5)
        game_single_o = TicTacToeGame(board_size=10, win_length=5)
        game_single_o.board[5][2] = PLAYER_O
        game_single_o.board[5][3] = PLAYER_X
        game_single_o.board[5][4] = PLAYER_X
        game_single_o.board[3][6] = PLAYER_O
        game_single_o.board[2][7] = PLAYER_X
        game_single_o.moves_count = 5
        game_single_o.current_player = PLAYER_X
        self.assertEqual(heuristic_rollout(game_single_o), (5, 5))

    # Tình huống 4: Phát triển chuỗi 3 -> 4
    def test_situation_4_extends_three_to_four_over_open_three(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        # Row 7: X has half-open 3 at (7, 0), (7, 1), (7, 2) -> playing (7, 3) forms (4, 1) = 2,000
        game.board[7][0] = PLAYER_X
        game.board[7][1] = PLAYER_X
        game.board[7][2] = PLAYER_X
        # Row 10: X has 2 open stones at (10, 5), (10, 6) -> playing (10, 4) or (10, 7) forms (3, 2) = 800
        game.board[10][5] = PLAYER_X
        game.board[10][6] = PLAYER_X
        game.moves_count = 5
        game.current_player = PLAYER_X

        self.assertEqual(_evaluate_move(game, (7, 3), PLAYER_X), 2_000)
        self.assertEqual(_evaluate_move(game, (10, 7), PLAYER_X), 800)
        self.assertEqual(heuristic_rollout(game), (7, 3))

        # When 3 stones are open on both ends at (4, 5), (4, 6), (4, 7) -> playing (4, 4) or (4, 8) forms (4, 2) = 10,000
        game2 = TicTacToeGame(board_size=15, win_length=5)
        for col in (5, 6, 7):
            game2.board[4][col] = PLAYER_X
        game2.board[10][5] = PLAYER_X
        game2.board[10][6] = PLAYER_X
        game2.moves_count = 5
        game2.current_player = PLAYER_X

        self.assertEqual(_evaluate_move(game2, (4, 4), PLAYER_X), 10_000)
        self.assertEqual(_evaluate_move(game2, (4, 8), PLAYER_X), 10_000)
        self.assertIn(heuristic_rollout(game2), {(4, 4), (4, 8)})

    # Tình huống 5: Cân bằng Attack và Defense trên bàn 10x10
    def test_situation_5_attack_defense_balance_on_10x10(self):
        game = TicTacToeGame(board_size=10, win_length=5)
        # X has two stones at (4, 4), (4, 5) -> extending at (4, 3) or (4, 6) forms (3, 2) = 800
        game.board[4][4] = PLAYER_X
        game.board[4][5] = PLAYER_X
        # O has three stones at (6, 4), (6, 5), (6, 6), flanked by X at (6, 2) and (6, 8)
        game.board[6][2] = PLAYER_X
        game.board[6][4] = PLAYER_O
        game.board[6][5] = PLAYER_O
        game.board[6][6] = PLAYER_O
        game.board[6][8] = PLAYER_X
        game.moves_count = 7
        game.current_player = PLAYER_X

        self.assertEqual(_evaluate_move(game, (4, 3), PLAYER_X), 800)
        self.assertEqual(_evaluate_move(game, (4, 6), PLAYER_X), 800)

        # At (6, 3) and (6, 7): Defense forms (4, 1) = 1,500; Attack forms (2, 1) = 15 -> total = 1,515
        self.assertEqual(_analyze_direction(game, 6, 3, 0, 1, PLAYER_O), (4, 1))
        self.assertEqual(_analyze_direction(game, 6, 3, 0, 1, PLAYER_X), (2, 1))
        self.assertEqual(_evaluate_move(game, (6, 3), PLAYER_X), 1_515)
        self.assertEqual(_evaluate_move(game, (6, 7), PLAYER_X), 1_515)

        self.assertIn(heuristic_rollout(game), {(6, 3), (6, 7)})

    # Tình huống 6: Ví dụ bàn cờ 5x5
    def test_situation_6_5x5_board_example_scores_and_choice(self):
        game = TicTacToeGame(board_size=5, win_length=5)
        game.board[2][1] = PLAYER_X
        game.board[2][2] = PLAYER_X
        game.board[3][2] = PLAYER_O
        game.board[3][3] = PLAYER_O
        game.moves_count = 4
        game.current_player = PLAYER_X

        # Candidate A = (2, 3): 800 (Attack horiz (3, 2)) + 30 (Defense vert (2, 2)) + 30 (Defense anti-diag (2, 2)) = 860
        self.assertEqual(_evaluate_move(game, (2, 3), PLAYER_X), 860)
        # Candidate B = (3, 1): 600 (Defense horiz (3, 2)) + 60 (Attack vert (2, 2)) + 60 (Attack anti-diag (2, 2)) = 720
        self.assertEqual(_evaluate_move(game, (3, 1), PLAYER_X), 720)
        self.assertEqual(heuristic_rollout(game), (2, 3))

        # When O also has a stone at (3, 1):
        game.board[3][1] = PLAYER_O
        game.moves_count = 5
        # C = (3, 0) scores 1,500 (Defense horiz (4, 1)) + 15 (Attack anti-diag (2, 1)) = 1,515
        self.assertEqual(_analyze_direction(game, 3, 0, 0, 1, PLAYER_O), (4, 1))
        self.assertEqual(_evaluate_move(game, (3, 0), PLAYER_X), 1_515)
        # D = (3, 4) scores 1,500 (Defense horiz (4, 1))
        self.assertEqual(_analyze_direction(game, 3, 4, 0, 1, PLAYER_O), (4, 1))
        self.assertEqual(_evaluate_move(game, (3, 4), PLAYER_X), 1_500)
        self.assertEqual(heuristic_rollout(game), (3, 0))

    # Tình huống 7: Tie-breaking ngẫu nhiên khi nhiều nước đồng điểm cao nhất
    def test_situation_7_tie_breaking_among_equal_best_moves(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        # Symmetric open 2-stone chain of X at (7, 6) and (7, 7)
        game.board[7][6] = PLAYER_X
        game.board[7][7] = PLAYER_X
        game.moves_count = 2
        game.current_player = PLAYER_X

        score_left = _evaluate_move(game, (7, 5), PLAYER_X)
        score_right = _evaluate_move(game, (7, 8), PLAYER_X)
        self.assertEqual(score_left, 800)
        self.assertEqual(score_right, 800)

        rng_state = random.getstate()
        try:
            random.seed(42)
            chosen_moves = {heuristic_rollout(game) for _ in range(30)}
        finally:
            random.setstate(rng_state)

        self.assertEqual(chosen_moves, {(7, 5), (7, 8)})

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

        self.assertEqual(controller.get_move(TicTacToeGame()), (7, 7))
        self.assertIn("rollout=heuristic", controller.last_status)

    def test_prioritizes_double_three_fork_intersection(self):
        game = TicTacToeGame()
        game.board[7][5] = PLAYER_X
        game.board[7][6] = PLAYER_X
        game.board[5][7] = PLAYER_X
        game.board[6][7] = PLAYER_X
        game.moves_count = 4
        game.current_player = PLAYER_X

        # Playing (7, 7) creates two open-3 lines simultaneously (800 + 800 = 1,600 attack points)
        self.assertEqual(_evaluate_move(game, (7, 7), PLAYER_X), 1_600)
        self.assertEqual(heuristic_rollout(game), (7, 7))

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

    def test_mcts_root_prioritizes_immediate_win_even_with_low_iterations(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        game.board[7][3:7] = [PLAYER_X] * 4
        game.board[5][3:7] = [PLAYER_O] * 4
        game.moves_count = 8
        game.current_player = PLAYER_X

        agent = MCTSAgent(iterations=2, uct_c=1.4, rollout_policy=heuristic_rollout, seed=1)
        move = agent.get_move(game)
        self.assertIn(move, {(7, 2), (7, 7)})

    def test_mcts_root_prioritizes_single_immediate_block_with_low_iterations(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        game.board[7][0:4] = [PLAYER_O] * 4
        game.board[2][2] = PLAYER_X
        game.moves_count = 5
        game.current_player = PLAYER_X

        agent = MCTSAgent(iterations=1, uct_c=1.4, rollout_policy=heuristic_rollout, seed=1)
        self.assertEqual(agent.get_move(game), (7, 4))

    def test_mcts_root_handles_multiple_opponent_threats(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        game.board[7][4:8] = [PLAYER_O] * 4
        game.board[5][3] = PLAYER_X
        game.board[6][3] = PLAYER_X
        game.moves_count = 6
        game.current_player = PLAYER_X

        agent = MCTSAgent(iterations=10, uct_c=1.4, rollout_policy=heuristic_rollout, seed=1)
        move = agent.get_move(game)
        self.assertIn(move, {(7, 3), (7, 8)})

    def test_mcts_root_detects_win_and_block_with_candidate_radius_zero(self):
        game = TicTacToeGame(board_size=15, win_length=5)
        game.board[7][0:4] = [PLAYER_O] * 4
        game.board[2][2] = PLAYER_X
        game.moves_count = 5
        game.current_player = PLAYER_X

        agent = MCTSAgent(
            iterations=2,
            uct_c=1.4,
            rollout_policy=heuristic_rollout,
            seed=1,
            candidate_radius=0,
        )
        self.assertEqual(agent.get_move(game), (7, 4))


if __name__ == "__main__":
    unittest.main()
