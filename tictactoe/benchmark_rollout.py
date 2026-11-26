"""Benchmark comparing MCTS with Random Rollout vs Heuristic Rollout (Task 5).

Supports three experimental modes in AI vs AI:
1. Equal Iterations: Both MCTS agents use the exact same iteration count.
2. Reduced Iterations (Iteration Scaling): Heuristic Rollout at lower iterations vs
   Random Rollout at higher iterations.
3. Equal Time Budget: Both MCTS agents are given the exact same wall-clock time budget
   per move, measuring iterations completed and win rate under identical time constraints.

Fairness & Reproducibility:
- Uses paired seeds: for each pair of games (2k, 2k+1) with seed = base_seed + k,
  Heuristic plays X (first) in game 2k and O (second) in game 2k+1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .constants import PLAYER_O, PLAYER_X
    from .game import TicTacToeGame
    from .mcts import MCTSAgent, Node, candidate_moves
    from .rollout import RolloutPolicy, heuristic_rollout, random_rollout
except ImportError:  # pragma: no cover
    from constants import PLAYER_O, PLAYER_X
    from game import TicTacToeGame
    from mcts import MCTSAgent, Node, candidate_moves
    from rollout import RolloutPolicy, heuristic_rollout, random_rollout

Move = Tuple[int, int]


class TimeBoundedMCTSAgent(MCTSAgent):
    """MCTSAgent wrapper that runs iterations until a per-move time budget expires."""

    def __init__(
        self,
        time_budget_s: float = 0.1,
        uct_c: float = 1.4,
        rollout_policy: Optional[RolloutPolicy] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            iterations=1,
            uct_c=uct_c,
            rollout_policy=rollout_policy,
            seed=seed,
        )
        if time_budget_s <= 0:
            raise ValueError("time_budget_s must be > 0.")
        self.time_budget_s = float(time_budget_s)

    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        if game.is_terminal() or not game.get_valid_moves():
            self._root = None
            self.last_stats = self._stats(
                iterations=0,
                time_s=0.0,
                best_by_n=None,
                best_by_wn=None,
                warnings=["Game is over: no move to search."],
            )
            return None

        start = time.perf_counter()
        radius = self._effective_radius(game)
        root_moves = candidate_moves(game, radius)
        self._root = Node(None, None, -game.current_player, list(root_moves))
        self._tree_size = 1

        if len(root_moves) == 1:
            move = root_moves[0]
            self.last_stats = self._stats(
                iterations=0,
                time_s=time.perf_counter() - start,
                best_by_n=move,
                best_by_wn=move,
                warnings=[],
            )
            return move

        completed_iterations = 0
        deadline = start + self.time_budget_s
        while True:
            self._iterate(self._root, game, radius)
            completed_iterations += 1
            if time.perf_counter() >= deadline:
                break

        by_n, by_wn = self._pick_final(self._root)
        chosen = by_n if self.final_policy == "max_n" else by_wn
        elapsed = time.perf_counter() - start
        self.last_stats = self._stats(
            iterations=completed_iterations,
            time_s=elapsed,
            best_by_n=by_n.move,
            best_by_wn=by_wn.move,
            warnings=[],
        )
        return chosen.move


def play_head_to_head_game(
    agent_x: MCTSAgent,
    agent_o: MCTSAgent,
    board_size: int,
    win_length: int,
) -> Dict[str, object]:
    """Play a single AI vs AI game and record outcome, move times, and iteration counts."""
    game = TicTacToeGame(board_size=board_size, win_length=win_length)
    move_times: Dict[int, List[float]] = {PLAYER_X: [], PLAYER_O: []}
    move_iters: Dict[int, List[int]] = {PLAYER_X: [], PLAYER_O: []}

    game_start = time.perf_counter()
    while not game.is_terminal():
        player = game.current_player
        agent = agent_x if player == PLAYER_X else agent_o
        t0 = time.perf_counter()
        move = agent.get_move(game)
        elapsed = time.perf_counter() - t0

        if move is None or not game.make_move(move):
            raise RuntimeError(f"Agent {agent} returned invalid move {move!r}.")

        iters = int(agent.last_stats.get("iterations", 0))
        if iters > 0:
            move_times[player].append(elapsed)
            move_iters[player].append(iters)

    game_duration = time.perf_counter() - game_start
    return {
        "result": game.get_result(),
        "moves_count": game.moves_count,
        "game_time_s": game_duration,
        "move_times": move_times,
        "move_iters": move_iters,
    }


def run_iteration_matchup(
    board_size: int,
    win_length: int,
    heur_iterations: int,
    rand_iterations: int,
    games: int = 20,
    uct_c: float = 1.4,
    seed: int = 100,
) -> Dict[str, object]:
    """Compare Heuristic Rollout vs Random Rollout with specified iteration budgets."""
    if games < 2 or games % 2 != 0:
        raise ValueError("games must be an even integer >= 2 for balanced X/O alternation.")

    heur_w = heur_d = heur_l = 0
    first_stats = {"W": 0, "D": 0, "L": 0}
    second_stats = {"W": 0, "D": 0, "L": 0}

    heur_move_times: List[float] = []
    rand_move_times: List[float] = []
    game_times: List[float] = []
    game_Reference_moves: List[int] = []

    for i in range(games):
        pair_seed = seed + (i // 2)
        heur_first = (i % 2 == 0)

        random.seed(pair_seed)
        heur_agent = MCTSAgent(
            iterations=heur_iterations,
            uct_c=uct_c,
            rollout_policy=heuristic_rollout,
            seed=pair_seed,
        )
        rand_agent = MCTSAgent(
            iterations=rand_iterations,
            uct_c=uct_c,
            rollout_policy=random_rollout,
            seed=pair_seed + 10_000,
        )

        agent_x, agent_o = (heur_agent, rand_agent) if heur_first else (rand_agent, heur_agent)
        match = play_head_to_head_game(agent_x, agent_o, board_size, win_length)

        heur_player = PLAYER_X if heur_first else PLAYER_O
        rand_player = -heur_player
        result = match["result"]

        side_bucket = first_stats if heur_first else second_stats
        if result == heur_player:
            heur_w += 1
            side_bucket["W"] += 1
        elif result == rand_player:
            heur_l += 1
            side_bucket["L"] += 1
        else:
            heur_d += 1
            side_bucket["D"] += 1

        heur_move_times.extend(match["move_times"][heur_player])
        rand_move_times.extend(match["move_times"][rand_player])
        game_times.append(match["game_time_s"])
        game_Reference_moves.append(match["moves_count"])

    avg_h_move = sum(heur_move_times) / len(heur_move_times) if heur_move_times else 0.0
    avg_r_move = sum(rand_move_times) / len(rand_move_times) if rand_move_times else 0.0

    return {
        "mode": "equal_iter" if heur_iterations == rand_iterations else "scaled_iter",
        "board": f"{board_size}x{board_size}",
        "k": win_length,
        "games": games,
        "heur_iter": heur_iterations,
        "rand_iter": rand_iterations,
        "time_budget_s": None,
        "heur_W": heur_w,
        "draws": heur_d,
        "heur_L": heur_l,
        "heur_win_rate": round(heur_w / games * 100, 1),
        "rand_win_rate": round(heur_l / games * 100, 1),
        "draw_rate": round(heur_d / games * 100, 1),
        "heur_no_loss_rate": round((heur_w + heur_d) / games * 100, 1),
        "heur_avg_move_s": round(avg_h_move, 4),
        "rand_avg_move_s": round(avg_r_move, 4),
        "time_ratio": round(avg_h_move / avg_r_move, 2) if avg_r_move > 0 else 0.0,
        "avg_game_s": round(sum(game_times) / len(game_times), 3),
        "avg_moves": round(sum(game_Reference_moves) / len(game_Reference_moves), 1),
        "heur_as_X": first_stats,
        "heur_as_O": second_stats,
    }


def run_time_budget_matchup(
    board_size: int,
    win_length: int,
    time_budget_s: float,
    games: int = 20,
    uct_c: float = 1.4,
    seed: int = 100,
) -> Dict[str, object]:
    """Compare Heuristic Rollout vs Random Rollout under the exact same move time budget."""
    if games < 2 or games % 2 != 0:
        raise ValueError("games must be an even integer >= 2 for balanced X/O alternation.")

    heur_w = heur_d = heur_l = 0
    first_stats = {"W": 0, "D": 0, "L": 0}
    second_stats = {"W": 0, "D": 0, "L": 0}

    heur_move_times: List[float] = []
    rand_move_times: List[float] = []
    heur_move_iters: List[int] = []
    rand_move_iters: List[int] = []
    game_times: List[float] = []
    game_moves: List[int] = []

    for i in range(games):
        pair_seed = seed + (i // 2)
        heur_first = (i % 2 == 0)

        random.seed(pair_seed)
        heur_agent = TimeBoundedMCTSAgent(
            time_budget_s=time_budget_s,
            uct_c=uct_c,
            rollout_policy=heuristic_rollout,
            seed=pair_seed,
        )
        rand_agent = TimeBoundedMCTSAgent(
            time_budget_s=time_budget_s,
            uct_c=uct_c,
            rollout_policy=random_rollout,
            seed=pair_seed + 10_000,
        )

        agent_x, agent_o = (heur_agent, rand_agent) if heur_first else (rand_agent, heur_agent)
        match = play_head_to_head_game(agent_x, agent_o, board_size, win_length)

        heur_player = PLAYER_X if heur_first else PLAYER_O
        rand_player = -heur_player
        result = match["result"]

        side_bucket = first_stats if heur_first else second_stats
        if result == heur_player:
            heur_w += 1
            side_bucket["W"] += 1
        elif result == rand_player:
            heur_l += 1
            side_bucket["L"] += 1
        else:
            heur_d += 1
            side_bucket["D"] += 1

        heur_move_times.extend(match["move_times"][heur_player])
        rand_move_times.extend(match["move_times"][rand_player])
        heur_move_iters.extend(match["move_iters"][heur_player])
        rand_move_iters.extend(match["move_iters"][rand_player])
        game_times.append(match["game_time_s"])
        game_moves.append(match["moves_count"])

    avg_h_move = sum(heur_move_times) / len(heur_move_times) if heur_move_times else 0.0
    avg_r_move = sum(rand_move_times) / len(rand_move_times) if rand_move_times else 0.0
    avg_h_iter = sum(heur_move_iters) / len(heur_move_iters) if heur_move_iters else 0.0
    avg_r_iter = sum(rand_move_iters) / len(rand_move_iters) if rand_move_iters else 0.0

    return {
        "mode": "equal_time",
        "board": f"{board_size}x{board_size}",
        "k": win_length,
        "games": games,
        "heur_iter": round(avg_h_iter, 1),
        "rand_iter": round(avg_r_iter, 1),
        "time_budget_s": time_budget_s,
        "heur_W": heur_w,
        "draws": heur_d,
        "heur_L": heur_l,
        "heur_win_rate": round(heur_w / games * 100, 1),
        "rand_win_rate": round(heur_l / games * 100, 1),
        "draw_rate": round(heur_d / games * 100, 1),
        "heur_no_loss_rate": round((heur_w + heur_d) / games * 100, 1),
        "heur_avg_move_s": round(avg_h_move, 4),
        "rand_avg_move_s": round(avg_r_move, 4),
        "time_ratio": round(avg_h_move / avg_r_move, 2) if avg_r_move > 0 else 0.0,
        "avg_game_s": round(sum(game_times) / len(game_times), 3),
        "avg_moves": round(sum(game_moves) / len(game_moves), 1),
        "heur_as_X": first_stats,
        "heur_as_O": second_stats,
    }


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    header = (
        f"{'Mode':<11} {'Board':<6} {'Gms':>3} {'H_Iter':>7} {'R_Iter':>7} "
        f"{'H_W':>3} {'D':>3} {'H_L':>3} {'H_Win%':>6} {'R_Win%':>6} "
        f"{'H_s/mv':>7} {'R_s/mv':>7} {'s/game':>7} {'Moves':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{str(r['mode']):<11} {str(r['board']):<6} {r['games']:>3} "
            f"{str(r['heur_iter']):>7} {str(r['rand_iter']):>7} "
            f"{r['heur_W']:>3} {r['draws']:>3} {r['heur_L']:>3} "
            f"{r['heur_win_rate']:>5.1f}% {r['rand_win_rate']:>5.1f}% "
            f"{r['heur_avg_move_s']:>7.4f} {r['rand_avg_move_s']:>7.4f} "
            f"{r['avg_game_s']:>7.3f} {r['avg_moves']:>5.1f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Random vs Heuristic Rollout in MCTS.")
    parser.add_argument("--games", type=int, default=20, help="Even number of games per configuration.")
    parser.add_argument("--seed", type=int, default=100, help="Base random seed.")
    parser.add_argument("--json", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path.")
    args = parser.parse_args(argv)

    results: List[Dict[str, object]] = []

    print("=== 1. EQUAL ITERATIONS BENCHMARK (10x10 & 15x15, k=5; 3x3, k=3) ===")
    for b, k, it, g in [
        (3, 3, 50, args.games),
        (3, 3, 100, args.games),
        (3, 3, 200, args.games),
        (10, 5, 30, args.games),
        (10, 5, 50, args.games),
        (10, 5, 100, args.games),
        (15, 5, 30, max(10, args.games // 2)),
        (15, 5, 50, max(10, args.games // 2)),
    ]:
        row = run_iteration_matchup(b, k, it, it, games=g, seed=args.seed)
        results.append(row)
        print(format_table([row]).splitlines()[-1], flush=True)

    print("\n=== 2. REDUCED ITERATIONS BENCHMARK (Heuristic Low Iter vs Random High Iter, 10x10 k=5) ===")
    for h_it, r_it in [(25, 50), (25, 100), (25, 200), (50, 200)]:
        row = run_iteration_matchup(10, 5, h_it, r_it, games=args.games, seed=args.seed)
        results.append(row)
        print(format_table([row]).splitlines()[-1], flush=True)

    print("\n=== 3. EQUAL TIME BUDGET BENCHMARK (Same Per-Move Time Budget, 10x10 & 15x15 k=5) ===")
    for b, k, budget, g in [
        (10, 5, 0.05, args.games),
        (10, 5, 0.10, args.games),
        (15, 5, 0.10, max(10, args.games // 2)),
    ]:
        row = run_time_budget_matchup(b, k, budget, games=g, seed=args.seed)
        results.append(row)
        print(format_table([row]).splitlines()[-1], flush=True)

    print("\n=== FULL SUMMARY TABLE ===")
    print(format_table(results))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    if args.csv:
        fieldnames = [
            "mode", "board", "k", "games", "heur_iter", "rand_iter", "time_budget_s",
            "heur_W", "draws", "heur_L", "heur_win_rate", "rand_win_rate", "draw_rate",
            "heur_no_loss_rate", "heur_avg_move_s", "rand_avg_move_s", "time_ratio",
            "avg_game_s", "avg_moves",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
