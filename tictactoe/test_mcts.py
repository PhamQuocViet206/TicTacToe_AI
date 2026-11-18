"""Unit tests for MCTS + UCT (requirement IDs MC-* in 03-mcts-uct v2.0).

Fast tests run by default. Slow tests (> 5 s) run only when RUN_SLOW=1:

    $env:PYTHONUTF8 = "1"; $env:RUN_SLOW = "1"
    python -m unittest discover -s tictactoe -p "test_*.py" -t .
"""

import math
import os
import random
import time
import unittest

from .ai_controller import AIController
from .constants import PLAYER_O, PLAYER_X
from .game import TicTacToeGame
from .mcts import MCTSAgent, Node, candidate_moves, uct_score
from .rollout import heuristic_rollout, random_rollout
from .verify_mcts import (
    OpponentUnavailable,
    PerfectTicTacToe,
    RandomPlayer,
    make_opponent,
    play_match,
    run_series,
    threshold_sweep,
)

RUN_SLOW = os.environ.get("RUN_SLOW") == "1"
slow = unittest.skipUnless(RUN_SLOW, "slow test: set RUN_SLOW=1")


def ttt(*moves):
    """3x3 Tic-Tac-Toe after playing ``moves`` alternately from X."""
    game = TicTacToeGame(board_size=3, win_length=3)
    for move in moves:
        assert game.make_move(move), move
    return game


def seeded_agent(seed, **kwargs):
    """Agent + global seed, so rollouts (global ``random``) are reproducible too."""
    random.seed(seed)
    return MCTSAgent(seed=seed, **kwargs)


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


class UCTScoreTests(unittest.TestCase):
    def test_value_1_0249(self):
        """MC-AC2.1"""
        self.assertAlmostEqual(uct_score(30, 50, 100, 1.4), 1.0249, delta=1e-4)

    def test_unvisited_is_inf(self):
        """MC-AC2.2"""
        self.assertEqual(uct_score(0, 0, 10, 1.4), math.inf)

    def test_first_visit_ln1_zero(self):
        """MC-EC2.1: N_p = 1 -> ln 1 = 0, exploration term vanishes without error."""
        self.assertEqual(uct_score(1, 1, 1, 1.4), 1.0)

    def test_exploration_grows_with_parent_visits(self):
        """MC-AC2.5: the exploration term grows when the parent is visited more."""
        scores = [uct_score(5, 10, parent_n, 1.4) for parent_n in (10, 100, 1000)]
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_c_zero_is_pure_wn(self):
        """MC-AC2.3"""
        self.assertEqual(uct_score(3, 4, 1000, 0.0), 0.75)


class NodeTests(unittest.TestCase):
    def _parent_with_children(self, stats):
        parent = Node(move=None, parent=None, player_just_moved=PLAYER_O, untried=[])
        parent.N = sum(n for _, n in stats)
        for index, (w, n) in enumerate(stats):
            child = parent.add_child((0, index), PLAYER_X, [])
            child.W, child.N = w, n
        return parent

    def test_add_child_removes_untried(self):
        parent = Node(move=None, parent=None, player_just_moved=PLAYER_O, untried=[(0, 0), (0, 1)])
        child = parent.add_child((0, 1), PLAYER_X, [])
        self.assertEqual(parent.untried, [(0, 0)])
        self.assertIs(child.parent, parent)
        self.assertEqual(child.player_just_moved, PLAYER_X)

    def test_tie_break_uses_rng_not_first_cell(self):
        """MC-EC2.2"""
        parent = self._parent_with_children([(1, 2), (1, 2), (1, 2)])
        chosen = {parent.best_child_uct(1.4, random.Random(seed)).move for seed in range(60)}
        self.assertEqual(chosen, {(0, 0), (0, 1), (0, 2)})

    def test_c_zero_selects_highest_wn(self):
        """MC-AC2.3: with c = 0 selection is pure exploitation."""
        parent = self._parent_with_children([(1, 10), (9, 10), (5, 10)])
        self.assertEqual(parent.best_child_uct(0.0, random.Random(0)).move, (0, 1))


class TacticsTests(unittest.TestCase):
    """Written before back-propagation: these fail if W is credited to the wrong player."""

    SEEDS = 100 if RUN_SLOW else 30

    def test_takes_immediate_win(self):
        """MC-AC1.2, MC-EC1.2: X to move, (0, 2) wins at once."""
        hits = 0
        for seed in range(self.SEEDS):
            game = ttt((0, 0), (1, 0), (0, 1), (1, 1))
            hits += seeded_agent(seed, iterations=500).get_move(game) == (0, 2)
        self.assertGreaterEqual(hits / self.SEEDS, 0.99)

    def test_blocks_threat(self):
        """MC-AC1.3, MC-EC1.2: O to move must block (0, 2)."""
        hits = 0
        for seed in range(self.SEEDS):
            game = ttt((0, 0), (1, 1), (0, 1))
            hits += seeded_agent(seed, iterations=1000).get_move(game) == (0, 2)
        self.assertGreaterEqual(hits / self.SEEDS, 0.95)


class CoreTests(unittest.TestCase):
    def test_root_children_n_sum_equals_iterations(self):
        """MC-AC1.1"""
        agent = seeded_agent(1, iterations=1000)
        agent.get_move(ttt())
        self.assertEqual(agent._root.N, 1000)
        self.assertEqual(sum(child.N for child in agent._root.children), 1000)

    def test_tree_invariants(self):
        """MC-AC1.4: 0 <= W <= N and N(parent) >= sum N(children)."""
        agent = seeded_agent(2, iterations=800)
        agent.get_move(ttt((1, 1)))
        for node in walk(agent._root):
            self.assertGreaterEqual(node.W, 0.0)
            self.assertLessEqual(node.W, node.N)
            self.assertGreaterEqual(node.N, sum(child.N for child in node.children))

    def test_game_not_mutated(self):
        """MC-AC1.5"""
        game = ttt((0, 0), (1, 1))
        before = ([row[:] for row in game.board], game.current_player, game.moves_count, game.winner, game.last_move)
        seeded_agent(3, iterations=300).get_move(game)
        after = ([row[:] for row in game.board], game.current_player, game.moves_count, game.winner, game.last_move)
        self.assertEqual(before, after)

    def test_terminal_node_backprops_real_result(self):
        """MC-EC1.1: a winning terminal child gets reward 1 on every visit (W == N)."""
        agent = seeded_agent(4, iterations=500)
        agent.get_move(ttt((0, 0), (1, 0), (0, 1), (1, 1)))
        win_child = next(child for child in agent._root.children if child.move == (0, 2))
        self.assertGreater(win_child.N, 0)
        self.assertEqual(win_child.W, win_child.N)
        self.assertEqual(win_child.children, [])
        self.assertEqual(win_child.untried, [])

    def test_budget_smaller_than_moves_still_legal(self):
        """MC-EC1.3"""
        game = ttt()
        agent = seeded_agent(5, iterations=3)
        move = agent.get_move(game)
        self.assertIn(move, game.get_valid_moves())
        self.assertTrue(agent.last_stats["warnings"])
        visited = {child.move for child in agent._root.children}
        self.assertIn(move, visited)

    def test_returns_none_when_game_over(self):
        """MC-EC1.4"""
        game = ttt((0, 0), (1, 0), (0, 1), (1, 1), (0, 2))
        self.assertTrue(game.is_terminal())
        self.assertIsNone(MCTSAgent(iterations=10, seed=0).get_move(game))

    def test_policy_returning_none_is_treated_as_end(self):
        """MC-EC5.3"""
        agent = seeded_agent(6, iterations=50, rollout_policy=lambda game: None)
        move = agent.get_move(ttt())
        self.assertIn(move, ttt().get_valid_moves())
        self.assertEqual(agent._root.N, 50)


class FinalPolicyTests(unittest.TestCase):
    def _root(self, stats):
        root = Node(move=None, parent=None, player_just_moved=PLAYER_O, untried=[])
        root.N = sum(n for _, n in stats)
        for index, (w, n) in enumerate(stats):
            child = root.add_child((index, 0), PLAYER_X, [])
            child.W, child.N = w, n
        return root

    def test_example_n1_vs_n800(self):
        """MC-EC3.1, MC-AC3.1: max_n ignores the noisy 1/1 child, max_wn falls for it."""
        root = self._root([(1.0, 1), (480.0, 800)])
        by_n, by_wn = MCTSAgent(iterations=1, seed=0)._pick_final(root)
        self.assertEqual(by_n.move, (1, 0))
        self.assertEqual(by_wn.move, (0, 0))

    def test_tie_break_by_wn_then_rng(self):
        """MC-AC3.1"""
        root = self._root([(4.0, 10), (6.0, 10)])
        by_n, _ = MCTSAgent(iterations=1, seed=0)._pick_final(root)
        self.assertEqual(by_n.move, (1, 0))

        tied = self._root([(5.0, 10), (5.0, 10)])
        picks = {MCTSAgent(iterations=1, seed=seed)._pick_final(tied)[0].move for seed in range(40)}
        self.assertEqual(picks, {(0, 0), (1, 0)})

    def test_final_policy_max_wn_is_used(self):
        """MC-F3: final_policy='max_wn' returns the W/N-best child."""
        game = ttt((0, 0), (1, 0), (0, 1), (1, 1))
        agent = seeded_agent(7, iterations=300, final_policy="max_wn")
        move = agent.get_move(game)
        self.assertEqual(move, agent.last_stats["best_by_wn"])

    def test_flag_max_n_equals_max_wn(self):
        """MC-AC3.3"""
        agent = seeded_agent(8, iterations=500)
        agent.get_move(ttt((0, 0), (1, 0), (0, 1), (1, 1)))
        stats = agent.last_stats
        self.assertEqual(stats["max_n_equals_max_wn"], stats["best_by_n"] == stats["best_by_wn"])
        self.assertTrue(stats["max_n_equals_max_wn"])


class RootStatsTests(unittest.TestCase):
    def test_sum_n_equals_iterations_and_sorted(self):
        """MC-AC4.1"""
        agent = seeded_agent(9, iterations=600)
        agent.get_move(ttt((1, 1)))
        rows = agent.root_stats()
        self.assertEqual(sum(row["N"] for row in rows), 600)
        self.assertEqual([row["N"] for row in rows], sorted((row["N"] for row in rows), reverse=True))
        self.assertEqual(set(rows[0]), {"move", "W", "N", "WN", "uct"})

    def test_reproducible_with_seed(self):
        """MC-AC4.2, MC-AC5.2"""
        runs = []
        for _ in range(2):
            agent = seeded_agent(10, iterations=400)
            move = agent.get_move(ttt((0, 0)))
            runs.append((move, agent.root_stats()))
        self.assertEqual(runs[0], runs[1])

    def test_w_is_float_with_draws(self):
        """MC-EC4.2"""
        agent = seeded_agent(11, iterations=400)
        agent.get_move(ttt((1, 1)))
        self.assertTrue(all(isinstance(row["W"], float) for row in agent.root_stats()))
        self.assertTrue(any(row["W"] != int(row["W"]) for row in agent.root_stats()))

    def test_empty_before_first_search(self):
        """MC-EC4.3"""
        self.assertEqual(MCTSAgent(iterations=5).root_stats(), [])


class BudgetSeedTests(unittest.TestCase):
    def test_exact_iterations(self):
        """MC-AC5.1"""
        budgets = (100, 200, 500, 1000, 2000, 5000) if RUN_SLOW else (100, 200, 500, 1000)
        for budget in budgets:
            with self.subTest(budget=budget):
                agent = seeded_agent(12, iterations=budget)
                agent.get_move(ttt())
                self.assertEqual(agent.last_stats["iterations"], budget)
                self.assertEqual(agent._root.N, budget)

    def test_last_stats_fields(self):
        """MC-AC5.3, MC-AC2.4"""
        agent = seeded_agent(13, iterations=200, uct_c=0.5)
        move = agent.get_move(ttt())
        stats = agent.last_stats
        for key in ("iterations", "tree_size", "time_s", "c", "rollout", "final_policy",
                    "best_by_n", "best_by_wn", "max_n_equals_max_wn", "warnings"):
            self.assertIn(key, stats)
        self.assertEqual(stats["c"], 0.5)
        self.assertEqual(stats["rollout"], "random_rollout")
        self.assertEqual(stats["best_by_n"], move)
        self.assertGreaterEqual(stats["tree_size"], len(agent._root.children) + 1)
        self.assertGreater(stats["time_s"], 0.0)

    def test_all_survey_c_values_run(self):
        """MC-AC2.4: c in {0.5, 1.0, 1.4, 2.0} is accepted and recorded."""
        for c in (0.5, 1.0, 1.4, 2.0):
            agent = seeded_agent(14, iterations=100, uct_c=c)
            self.assertIn(agent.get_move(ttt()), ttt().get_valid_moves())
            self.assertEqual(agent.last_stats["c"], c)

    def test_invalid_params_raise(self):
        """MC-EC2.3, MC-EC3.2, MC-EC5.1, MC-EC8.2"""
        for kwargs in ({"iterations": 0}, {"iterations": -5}, {"iterations": 2.5}, {"iterations": True},
                       {"uct_c": -0.1}, {"final_policy": "best"}, {"candidate_radius": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MCTSAgent(**kwargs)

    def test_single_move_immediate(self):
        """MC-EC5.2"""
        game = ttt((0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0))
        agent = MCTSAgent(iterations=500, seed=0)
        self.assertEqual(agent.get_move(game), (2, 2))
        self.assertEqual(agent.last_stats["iterations"], 0)

    def test_default_rollout_is_random(self):
        self.assertIs(MCTSAgent().rollout_policy, random_rollout)


class CandidateTests(unittest.TestCase):
    def test_radius2_around_center_24_cells(self):
        """MC-AC8.1"""
        game = TicTacToeGame()
        game.make_move((7, 7))
        moves = candidate_moves(game, 2)
        self.assertEqual(len(moves), 24)
        self.assertTrue(all(max(abs(r - 7), abs(c - 7)) <= 2 for r, c in moves))

    def test_small_board_all_moves(self):
        """MC-AC8.2"""
        game = ttt((1, 1))
        agent = seeded_agent(15, iterations=50)
        agent.get_move(game)
        self.assertEqual(len(agent._root.children) + len(agent._root.untried), len(game.get_valid_moves()))

    def test_empty_board_center(self):
        """MC-AC8.3"""
        agent = MCTSAgent(iterations=50, seed=0)
        self.assertEqual(agent.get_move(TicTacToeGame()), (7, 7))
        self.assertEqual(agent.last_stats["iterations"], 0)

    def test_fallback_when_radius_full(self):
        """MC-EC8.1"""
        game = TicTacToeGame(board_size=7, win_length=5)
        game.board[0][0] = PLAYER_X
        game.board[0][1] = PLAYER_O
        game.board[1][0] = PLAYER_O
        game.board[1][1] = PLAYER_X
        game.moves_count = 4
        self.assertEqual(candidate_moves(game, 0), game.get_valid_moves())

    def test_negative_radius_raises(self):
        """MC-EC8.2"""
        with self.assertRaises(ValueError):
            candidate_moves(TicTacToeGame(), -1)

    def test_large_board_expansion_stays_near_stones(self):
        game = TicTacToeGame()
        for move in ((7, 7), (7, 8), (8, 7)):
            game.make_move(move)
        agent = seeded_agent(16, iterations=30)
        move = agent.get_move(game)
        self.assertLessEqual(min(max(abs(move[0] - r), abs(move[1] - c)) for r, c in ((7, 7), (7, 8), (8, 7))), 2)


class ControllerTests(unittest.TestCase):
    def test_mcts_returns_legal_move_15x15(self):
        """MC-AC7.1"""
        game = TicTacToeGame()
        for move in ((7, 7), (7, 8), (8, 8)):
            game.make_move(move)
        snapshot = [row[:] for row in game.board]
        random.seed(0)
        controller = AIController(algorithm="mcts", iterations=40, uct_c=1.4, rollout_policy="random")
        move = controller.get_move(game)
        self.assertIn(move, game.get_valid_moves())
        self.assertEqual(game.board, snapshot)
        self.assertIn("MCTS", controller.last_status)

    def test_heuristic_policy_used(self):
        """MC-AC7.2"""
        random.seed(0)
        controller = AIController(algorithm="mcts", iterations=60, rollout_policy="heuristic")
        controller.get_move(ttt((1, 1)))
        self.assertIs(controller.last_agent.rollout_policy, heuristic_rollout)
        self.assertEqual(controller.last_agent.last_stats["rollout"], "heuristic_rollout")

    def test_unknown_policy_returns_none(self):
        """MC-EC7.1"""
        controller = AIController(algorithm="mcts", rollout_policy="nope")
        self.assertIsNone(controller.get_move(ttt()))
        self.assertIn("nope", controller.last_status)

    def test_status_contains_c_and_iterations(self):
        """MC-AC2.4"""
        random.seed(0)
        controller = AIController(algorithm="mcts", iterations=120, uct_c=2.0)
        controller.get_move(ttt())
        self.assertIn("c=2.0", controller.last_status)
        self.assertIn("it=120", controller.last_status)

    def test_invalid_c_from_gui_reports_status(self):
        controller = AIController(algorithm="mcts", iterations=50, uct_c=-1.0)
        self.assertIsNone(controller.get_move(ttt()))
        self.assertIn("MCTS", controller.last_status)


class VerifyTests(unittest.TestCase):
    def test_perfect_player_never_loses_to_random(self):
        """MC-EC6.3 support: the reference opponent really is optimal."""
        for seed in range(20):
            perfect, rnd = PerfectTicTacToe(seed=seed), RandomPlayer(seed=seed + 100)
            x, o = (perfect, rnd) if seed % 2 == 0 else (rnd, perfect)
            result, _ = play_match(x, o, 3, 3)
            perfect_side = PLAYER_X if seed % 2 == 0 else PLAYER_O
            self.assertIn(result, (0, perfect_side))

    def test_perfect_vs_perfect_draws_and_varies(self):
        """MC-EC6.3: random tie-break among optimal moves gives varied games."""
        openings = set()
        for seed in range(10):
            game = ttt()
            first = PerfectTicTacToe(seed=seed).get_move(game)
            openings.add(first)
            result, _ = play_match(PerfectTicTacToe(seed=seed), PerfectTicTacToe(seed=seed + 1), 3, 3)
            self.assertEqual(result, 0)
        self.assertGreater(len(openings), 1)

    def test_perfect_rejects_large_board(self):
        with self.assertRaises(ValueError):
            PerfectTicTacToe(seed=0).get_move(TicTacToeGame())

    def test_series_alternates_first_player(self):
        """MC-F6: games alternate who moves first, 50/50."""
        row = run_series("random", games=4, iterations=50, seed=1)
        self.assertEqual(row["games"], 4)
        self.assertEqual(row["W"] + row["D"] + row["L"], 4)
        sides = row["by_side"]
        self.assertEqual(sum(sides["first"].values()), 2)
        self.assertEqual(sum(sides["second"].values()), 2)

    def test_series_rejects_odd_games(self):
        with self.assertRaises(ValueError):
            run_series("random", games=3, iterations=10, seed=1)

    def test_threshold_table_columns(self):
        """MC-AC6.4"""
        rows = threshold_sweep([20, 40], opponent="random", games=2, seed=3)
        self.assertEqual([row["iterations"] for row in rows], [20, 40])
        for column in ("board", "k", "opponent", "iterations", "c", "rollout", "final_policy", "games",
                       "W", "D", "L", "win_rate", "no_loss_rate", "avg_move_s", "seed"):
            self.assertIn(column, rows[0])

    def test_minimax_not_ready_message(self):
        """MC-EC6.4"""
        from .minimax import MinimaxAI

        try:
            MinimaxAI().get_move(ttt())
        except NotImplementedError:
            with self.assertRaises(OpponentUnavailable) as ctx:
                make_opponent("minimax", seed=0, board_size=3, k=3)
            self.assertIn("perfect", str(ctx.exception))
        else:
            self.skipTest("MinimaxAI is implemented; the not-ready path no longer applies.")

    @slow
    def test_no_loss_vs_perfect_50_games_5000it(self):
        """MC-AC6.1"""
        row = run_series("perfect", games=50, iterations=5000, seed=1)
        self.assertEqual(row["L"], 0, row)

    @slow
    def test_no_loss_vs_random_50_games_5000it(self):
        """MC-AC6.2"""
        row = run_series("random", games=50, iterations=5000, seed=1)
        self.assertEqual(row["L"], 0, row)
        self.assertGreater(row["W"], row["games"] / 2)

    @slow
    def test_board9_table_runs(self):
        """MC-AC6.5 (smoke): the 5-in-a-row table can be produced on a larger board."""
        row = run_series("random", games=2, iterations=100, seed=1, board_size=9, k=5)
        self.assertEqual(row["W"] + row["D"] + row["L"], 2)
        self.assertEqual((row["board"], row["k"]), (9, 5))


class PerfTests(unittest.TestCase):
    @slow
    def test_3x3_5000_it_under_2s(self):
        """MC-NFR1"""
        best = math.inf
        for seed in range(3):
            agent = seeded_agent(seed, iterations=5000)
            start = time.perf_counter()
            agent.get_move(ttt())
            best = min(best, time.perf_counter() - start)
        self.assertLessEqual(best, 2.0)

    @slow
    def test_15x15_1000_it_reports_time(self):
        """MC-NFR1: 15x15 has no hard limit; just record the time."""
        game = TicTacToeGame()
        for move in ((7, 7), (7, 8), (8, 8), (6, 6)):
            game.make_move(move)
        agent = seeded_agent(0, iterations=1000)
        move = agent.get_move(game)
        self.assertIn(move, game.get_valid_moves())
        print(f"\n[MC-NFR1] 15x15, 1000 iterations: {agent.last_stats['time_s']:.2f}s")


if __name__ == "__main__":
    unittest.main()
