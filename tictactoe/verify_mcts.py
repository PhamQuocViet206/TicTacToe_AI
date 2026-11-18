"""Verify MCTS + UCT: "never loses at 3x3", iteration threshold, W/D/L tables, root stats.

Examples (run from the project root):

    python -m tictactoe.verify_mcts --iterations 5000 --games 50 --opponent perfect --seed 1
    python -m tictactoe.verify_mcts --iterations 5000 --games 50 --opponent random
    python -m tictactoe.verify_mcts --threshold 100 200 500 1000 2000 5000 --games 50 --csv mcts_3x3.csv
    python -m tictactoe.verify_mcts --board 15 --k 5 --iterations 1000 --games 50 --opponent random
    python -m tictactoe.verify_mcts --final-policy both --iterations 200 --games 50
    python -m tictactoe.verify_mcts --exhaustive --iterations 5000 --seed 1
    python -m tictactoe.verify_mcts --root-stats --iterations 1000 --seed 7 --moves "0,0 1,1 0,1"

Opponents: ``perfect`` (exact 3x3 solver, random tie-break), ``random``, ``self`` (another
MCTS with the same settings), ``minimax`` (the team's ``MinimaxAI`` once it is implemented).
Each series alternates who moves first (games must be even). Before every game the global
``random`` module is seeded too, because the rollout policies in ``rollout.py`` use it.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .constants import PLAYER_O, PLAYER_X
    from .game import TicTacToeGame
    from .mcts import MCTSAgent
    from .rollout import ROLLOUT_POLICIES
except ImportError:  # pragma: no cover
    from constants import PLAYER_O, PLAYER_X
    from game import TicTacToeGame
    from mcts import MCTSAgent
    from rollout import ROLLOUT_POLICIES

Move = Tuple[int, int]

CSV_COLUMNS = [
    "board", "k", "opponent", "iterations", "c", "rollout", "final_policy", "games",
    "W", "D", "L", "win_rate", "no_loss_rate", "avg_move_s", "seed",
]
OPPONENTS = ("perfect", "random", "self", "minimax")
OPPONENT_SEED_OFFSET = 10_000


class OpponentUnavailable(RuntimeError):
    """The requested opponent cannot be used (e.g. MinimaxAI not implemented yet)."""


# ---------------------------------------------------------------- opponents
class PerfectTicTacToe:
    """Exact negamax solver for 3x3 Tic-Tac-Toe, choosing randomly among optimal moves (MC-EC6.3)."""

    _cache: Dict[Tuple[Tuple[int, ...], int], int] = {}

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.name = "perfect"

    @staticmethod
    def _key(game: TicTacToeGame) -> Tuple[Tuple[int, ...], int]:
        return tuple(cell for row in game.board for cell in row), game.current_player

    def _value(self, game: TicTacToeGame) -> int:
        """Game value (+1 win / 0 draw / -1 loss) for the player to move in a non-terminal ``game``."""
        key = self._key(game)
        if key not in self._cache:
            self._cache[key] = max(self._move_value(game, move) for move in game.get_valid_moves())
        return self._cache[key]

    def _move_value(self, game: TicTacToeGame, move: Move) -> int:
        player = game.current_player
        child = game.copy()
        child.make_move(move)
        if child.winner == player:
            return 1
        if child.is_terminal():
            return 0
        return -self._value(child)

    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        if game.board_size != 3:
            raise ValueError("PerfectTicTacToe only supports the 3x3 board.")
        moves = game.get_valid_moves()
        if not moves:
            return None
        values = {move: self._move_value(game, move) for move in moves}
        best = max(values.values())
        return self.rng.choice([move for move, value in values.items() if value == best])


class RandomPlayer:
    """Uniformly random legal moves with its own RNG."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.name = "random"

    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        moves = game.get_valid_moves()
        return self.rng.choice(moves) if moves else None


def make_opponent(name: str, seed: int, board_size: int = 3, k: int = 3, iterations: int = 1000,
                  uct_c: float = 1.4, rollout: str = "random", final_policy: str = "max_n"):
    """Build an opponent; raises OpponentUnavailable with a clear message when it cannot be used."""
    if name == "perfect":
        if board_size != 3:
            raise OpponentUnavailable("Opponent 'perfect' only supports --board 3. Use 'random' or 'minimax'.")
        return PerfectTicTacToe(seed=seed)
    if name == "random":
        return RandomPlayer(seed=seed)
    if name == "self":
        return MCTSAgent(iterations=iterations, uct_c=uct_c, rollout_policy=ROLLOUT_POLICIES[rollout],
                         final_policy=final_policy, seed=seed)
    if name == "minimax":
        try:
            from .minimax import MinimaxAI
        except ImportError:  # pragma: no cover
            from minimax import MinimaxAI
        player = MinimaxAI()
        try:
            player.get_move(TicTacToeGame(board_size=board_size, win_length=k))
        except NotImplementedError as exc:
            raise OpponentUnavailable(
                f"MinimaxAI is not implemented yet ({exc}). Use '--opponent perfect' for the optimal 3x3 opponent."
            ) from exc
        return player
    raise OpponentUnavailable(f"Unknown opponent '{name}'. Choose from {OPPONENTS}.")


# ---------------------------------------------------------------- matches
def play_match(agent_x, agent_o, board_size: int = 3, k: int = 3) -> Tuple[Optional[int], Dict[int, List[float]]]:
    """Play one game. Returns (get_result(), per-player move times in seconds)."""
    game = TicTacToeGame(board_size=board_size, win_length=k)
    times: Dict[int, List[float]] = {PLAYER_X: [], PLAYER_O: []}
    while not game.is_terminal():
        player = game.current_player
        agent = agent_x if player == PLAYER_X else agent_o
        start = time.perf_counter()
        move = agent.get_move(game)
        times[player].append(time.perf_counter() - start)
        if move is None or not game.make_move(move):
            raise RuntimeError(f"{getattr(agent, 'name', agent)} returned an illegal move {move!r}.")
    return game.get_result(), times


def run_series(opponent: str, games: int = 50, iterations: int = 1000, uct_c: float = 1.4,
               rollout: str = "random", final_policy: str = "max_n", board_size: int = 3, k: int = 3,
               seed: int = 1) -> Dict[str, object]:
    """MCTS vs ``opponent`` over ``games`` games, MCTS moving first in even-numbered games."""
    if games < 2 or games % 2:
        raise ValueError("games must be an even number >= 2 so both sides move first equally often.")
    tally = {"W": 0, "D": 0, "L": 0}
    by_side = {"first": {"W": 0, "D": 0, "L": 0}, "second": {"W": 0, "D": 0, "L": 0}}
    mcts_times: List[float] = []

    for i in range(games):
        random.seed(seed + i)  # rollout policies use the global RNG
        mcts = MCTSAgent(iterations=iterations, uct_c=uct_c, rollout_policy=ROLLOUT_POLICIES[rollout],
                         final_policy=final_policy, seed=seed + i)
        opp = make_opponent(opponent, seed + OPPONENT_SEED_OFFSET + i, board_size, k,
                            iterations, uct_c, rollout, final_policy)
        mcts_first = i % 2 == 0
        agent_x, agent_o = (mcts, opp) if mcts_first else (opp, mcts)
        result, times = play_match(agent_x, agent_o, board_size, k)

        mcts_player = PLAYER_X if mcts_first else PLAYER_O
        outcome = "D" if result in (0, None) else ("W" if result == mcts_player else "L")
        tally[outcome] += 1
        by_side["first" if mcts_first else "second"][outcome] += 1
        mcts_times.extend(times[mcts_player])

    return {
        "board": board_size,
        "k": k,
        "opponent": opponent,
        "iterations": iterations,
        "c": uct_c,
        "rollout": rollout,
        "final_policy": final_policy,
        "games": games,
        **tally,
        "win_rate": tally["W"] / games,
        "no_loss_rate": (tally["W"] + tally["D"]) / games,
        "avg_move_s": sum(mcts_times) / len(mcts_times) if mcts_times else 0.0,
        "seed": seed,
        "by_side": by_side,
    }


def threshold_sweep(budgets: Iterable[int], opponent: str = "perfect", **kwargs) -> List[Dict[str, object]]:
    """One ``run_series`` row per iteration budget (MC-AC6.4)."""
    return [run_series(opponent, iterations=budget, **kwargs) for budget in budgets]


def exhaustive_check(iterations: int = 5000, uct_c: float = 1.4, rollout: str = "random",
                     seed: int = 1) -> Dict[str, Dict[str, object]]:
    """MC-AC6.3: try every opponent reply on 3x3 while MCTS (fixed seed) plays one side.

    The MCTS decision for a position is deterministic (agent and global RNG reseeded with
    ``seed`` before each search) and memoised, so each position is searched once.
    """
    decisions: Dict[Tuple[Tuple[int, ...], int], Move] = {}

    def decide(game: TicTacToeGame) -> Move:
        key = (tuple(cell for row in game.board for cell in row), game.current_player)
        if key not in decisions:
            random.seed(seed)
            agent = MCTSAgent(iterations=iterations, uct_c=uct_c, rollout_policy=ROLLOUT_POLICIES[rollout], seed=seed)
            decisions[key] = agent.get_move(game)
        return decisions[key]

    report: Dict[str, Dict[str, object]] = {}
    for label, mcts_player in (("mcts_first", PLAYER_X), ("mcts_second", PLAYER_O)):
        stats = {"branches": 0, "losses": 0, "draws": 0, "wins": 0, "loss_lines": []}

        def explore(game: TicTacToeGame, line: List[Move]) -> None:
            if game.is_terminal():
                stats["branches"] += 1
                result = game.get_result()
                if result == 0:
                    stats["draws"] += 1
                elif result == mcts_player:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                    stats["loss_lines"].append(list(line))
                return
            moves = [decide(game)] if game.current_player == mcts_player else game.get_valid_moves()
            for move in moves:
                child = game.copy()
                child.make_move(move)
                explore(child, line + [move])

        explore(TicTacToeGame(board_size=3, win_length=3), [])
        report[label] = stats
    report["searched_positions"] = {"count": len(decisions)}
    return report


# ---------------------------------------------------------------- output
def format_rows(rows: Sequence[Dict[str, object]]) -> str:
    header = f"{'board':>5} {'k':>2} {'opponent':>8} {'iter':>5} {'c':>4} {'rollout':>9} {'final':>6} " \
             f"{'games':>5} {'W':>3} {'D':>3} {'L':>3} {'win%':>6} {'noloss%':>7} {'s/move':>7}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['board']:>5} {row['k']:>2} {row['opponent']:>8} {row['iterations']:>5} {row['c']:>4} "
            f"{row['rollout']:>9} {row['final_policy']:>6} {row['games']:>5} {row['W']:>3} {row['D']:>3} "
            f"{row['L']:>3} {100 * row['win_rate']:>6.1f} {100 * row['no_loss_rate']:>7.1f} {row['avg_move_s']:>7.3f}"
        )
    return "\n".join(lines)


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_root_stats(rows: Sequence[Dict[str, object]], top: int = 10) -> str:
    lines = [f"{'Nước':>8} {'W':>9} {'N':>6} {'W/N':>6} {'UCT':>7}", "-" * 40]
    for row in rows[:top]:
        lines.append(f"{str(row['move']):>8} {row['W']:>9.1f} {row['N']:>6} {row['WN']:>6.3f} {row['uct']:>7.3f}")
    if len(rows) > top:
        lines.append(f"... ({len(rows) - top} nước khác)")
    return "\n".join(lines)


def parse_moves(text: str) -> List[Move]:
    """'0,0 1,1' -> [(0, 0), (1, 1)]"""
    return [tuple(int(part) for part in item.split(",")) for item in text.split()] if text else []


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MCTS + UCT (MC-F4, MC-F6).")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.4, dest="uct_c")
    parser.add_argument("--rollout", choices=sorted(ROLLOUT_POLICIES), default="random")
    parser.add_argument("--final-policy", choices=["max_n", "max_wn", "both"], default="max_n")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--opponent", choices=OPPONENTS, default="perfect")
    parser.add_argument("--board", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threshold", type=int, nargs="+", metavar="ITER", help="iteration budgets to sweep")
    parser.add_argument("--exhaustive", action="store_true", help="try every opponent reply (3x3)")
    parser.add_argument("--root-stats", action="store_true", help="print W, N, W/N at the root")
    parser.add_argument("--moves", default="", help="position for --root-stats, e.g. '0,0 1,1'")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--csv", help="write the result table to this CSV file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    args = build_parser().parse_args(argv)

    if args.root_stats:
        game = TicTacToeGame(board_size=args.board, win_length=args.k)
        for move in parse_moves(args.moves):
            if not game.make_move(move):
                print(f"Nước {move} không hợp lệ.")
                return 2
        random.seed(args.seed)
        agent = MCTSAgent(iterations=args.iterations, uct_c=args.uct_c,
                          rollout_policy=ROLLOUT_POLICIES[args.rollout], seed=args.seed)
        move = agent.get_move(game)
        stats = agent.last_stats
        print(f"Bàn {args.board}x{args.board}, k={args.k}, nước đã đi: {parse_moves(args.moves)}")
        print(f"{agent.name}, rollout={args.rollout}, seed={args.seed}: chọn {move} "
              f"(max N = {stats['best_by_n']}, max W/N = {stats['best_by_wn']}, "
              f"trùng nhau: {stats['max_n_equals_max_wn']}, {stats['time_s']:.2f}s, {stats['tree_size']} nút)")
        rows = agent.root_stats()
        print(format_root_stats(rows, args.top))
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["move", "W", "N", "WN", "uct"])
                writer.writeheader()
                writer.writerows(rows)
        return 0

    if args.exhaustive:
        start = time.perf_counter()
        report = exhaustive_check(args.iterations, args.uct_c, args.rollout, args.seed)
        for label in ("mcts_first", "mcts_second"):
            part = report[label]
            print(f"{label}: {part['branches']} nhánh, thắng {part['wins']}, hoà {part['draws']}, thua {part['losses']}")
            for line in part["loss_lines"][:5]:
                print(f"  thua: {line}")
        print(f"Số thế cờ MCTS đã tìm: {report['searched_positions']['count']}, {time.perf_counter() - start:.1f}s")
        return 0 if report["mcts_first"]["losses"] == 0 and report["mcts_second"]["losses"] == 0 else 1

    policies = ["max_n", "max_wn"] if args.final_policy == "both" else [args.final_policy]
    budgets = args.threshold or [args.iterations]
    common = dict(games=args.games, uct_c=args.uct_c, rollout=args.rollout,
                  board_size=args.board, k=args.k, seed=args.seed)
    try:
        rows = [run_series(args.opponent, iterations=budget, final_policy=policy, **common)
                for policy in policies for budget in budgets]
    except (OpponentUnavailable, ValueError) as exc:
        print(f"Lỗi: {exc}")
        return 2

    print(format_rows(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"Đã ghi {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
