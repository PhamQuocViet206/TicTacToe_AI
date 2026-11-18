"""Monte Carlo Tree Search with the UCT selection policy.

    UCT(i) = W_i / N_i + c * sqrt(ln(N_parent) / N_i)

Each iteration runs the four classic phases on a copy of the game:

1. Selection     - walk down the tree, always taking the child with the best UCT.
2. Expansion     - add one untried move (chosen with the agent's own seeded RNG).
3. Simulation    - play the game to the end with the configured rollout policy
                   (``tictactoe.rollout``; the policy only chooses moves).
4. Backpropagation - add the reward to every node on the path.

Reward convention: a node's ``W`` is counted from the point of view of the player
who made the move *leading into* that node (win = 1, draw = 0.5, loss = 0).
Moving one level up flips the point of view, so the reward becomes ``1 - r``.
A parent therefore picks the child that is best for the player to move at the parent.

The final move is the child with the largest visit count N (robust child); W/N is
available through ``final_policy="max_wn"`` for the N-vs-W/N analysis only.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .constants import EMPTY, PLAYER_O, PLAYER_X
    from .game import TicTacToeGame
    from .rollout import RolloutPolicy, random_rollout
except ImportError:  # pragma: no cover
    from constants import EMPTY, PLAYER_O, PLAYER_X
    from game import TicTacToeGame
    from rollout import RolloutPolicy, random_rollout

Move = Tuple[int, int]

FINAL_POLICIES = ("max_n", "max_wn")
SMALL_BOARD_MAX = 5   # boards up to 5x5 consider every empty cell
DEFAULT_RADIUS = 2    # larger boards only expand cells near existing stones


def uct_score(w: float, n: int, parent_n: int, c: float) -> float:
    """UCT value of a child; unvisited children are +inf so they are tried first."""
    if n == 0:
        return math.inf
    exploit = w / n
    if parent_n <= 1:  # ln(1) = 0: no exploration bonus yet
        return exploit
    return exploit + c * math.sqrt(math.log(parent_n) / n)


def candidate_moves(game: TicTacToeGame, radius: Optional[int]) -> List[Move]:
    """Moves the tree may expand from ``game``.

    ``radius=None`` means every legal move. Otherwise only empty cells within
    Chebyshev distance ``radius`` of an existing stone are kept; an empty board
    yields the centre cell, and if no cell qualifies all legal moves are returned.
    """
    if radius is not None and radius < 0:
        raise ValueError("radius must be >= 0 or None.")
    valid = game.get_valid_moves()
    if radius is None or not valid:
        return valid

    size = game.board_size
    board = game.board
    seen = set()
    near: List[Move] = []
    has_stone = False
    for row in range(size):
        for col in range(size):
            if board[row][col] == EMPTY:
                continue
            has_stone = True
            for r in range(max(0, row - radius), min(size, row + radius + 1)):
                for c in range(max(0, col - radius), min(size, col + radius + 1)):
                    if board[r][c] == EMPTY and (r, c) not in seen:
                        seen.add((r, c))
                        near.append((r, c))

    if not has_stone:
        return [(size // 2, size // 2)]
    return near or valid


class Node:
    """Search-tree node. ``W``/``N`` are from the view of ``player_just_moved``."""

    __slots__ = ("move", "parent", "children", "untried", "W", "N", "player_just_moved")

    def __init__(self, move: Optional[Move], parent: Optional["Node"], player_just_moved: int, untried: List[Move]):
        self.move = move
        self.parent = parent
        self.children: List[Node] = []
        self.untried = untried
        self.W = 0.0
        self.N = 0
        self.player_just_moved = player_just_moved

    def add_child(self, move: Move, player_just_moved: int, untried: List[Move]) -> "Node":
        if move in self.untried:
            self.untried.remove(move)
        child = Node(move, self, player_just_moved, untried)
        self.children.append(child)
        return child

    def best_child_uct(self, c: float, rng: random.Random) -> "Node":
        """Child with the highest UCT; exact ties are broken with ``rng`` (MC-EC2.2)."""
        best: List[Node] = []
        best_score = -math.inf
        for child in self.children:
            score = uct_score(child.W, child.N, self.N, c)
            if score > best_score:
                best_score = score
                best = [child]
            elif score == best_score:
                best.append(child)
        return best[0] if len(best) == 1 else rng.choice(best)


class MCTSAgent:
    """MCTS + UCT agent with the same ``get_move(game)`` interface as the other AIs."""

    def __init__(
        self,
        iterations: int = 1000,
        uct_c: float = 1.4,
        rollout_policy: Optional[RolloutPolicy] = None,
        final_policy: str = "max_n",
        seed: Optional[int] = None,
        candidate_radius: Optional[int] = None,
    ):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError(f"iterations must be a positive integer, got {iterations!r}.")
        uct_c = float(uct_c)
        if not uct_c >= 0:  # also rejects NaN
            raise ValueError(f"uct_c must be >= 0, got {uct_c!r}.")
        if final_policy not in FINAL_POLICIES:
            raise ValueError(f"final_policy must be one of {FINAL_POLICIES}, got {final_policy!r}.")
        if candidate_radius is not None and (isinstance(candidate_radius, bool) or candidate_radius < 0):
            raise ValueError(f"candidate_radius must be >= 0 or None, got {candidate_radius!r}.")

        self.iterations = iterations
        self.uct_c = uct_c
        self.rollout_policy: RolloutPolicy = rollout_policy or random_rollout
        self.final_policy = final_policy
        self.seed = seed
        self.candidate_radius = candidate_radius
        self.rng = random.Random(seed)
        self.last_stats: Dict[str, object] = {}
        self._root: Optional[Node] = None
        self._tree_size = 0

    @property
    def name(self) -> str:
        return f"MCTS(c={self.uct_c}, it={self.iterations})"

    # ------------------------------------------------------------------ search
    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        """Search from ``game`` (never modified) and return the chosen move, or None if the game is over."""
        if game.is_terminal() or not game.get_valid_moves():
            self._root = None
            self.last_stats = self._stats(iterations=0, time_s=0.0, best_by_n=None, best_by_wn=None,
                                          warnings=["Game is over: no move to search."])
            return None

        start = time.perf_counter()
        radius = self._effective_radius(game)
        root_moves = candidate_moves(game, radius)
        self._root = Node(None, None, -game.current_player, list(root_moves))
        self._tree_size = 1

        if len(root_moves) == 1:  # MC-EC5.2 / MC-AC8.3: nothing to compare
            move = root_moves[0]
            self.last_stats = self._stats(iterations=0, time_s=time.perf_counter() - start,
                                          best_by_n=move, best_by_wn=move, warnings=[])
            return move

        for _ in range(self.iterations):
            self._iterate(self._root, game, radius)

        by_n, by_wn = self._pick_final(self._root)
        chosen = by_n if self.final_policy == "max_n" else by_wn
        warnings = []
        if self.iterations < len(root_moves):
            warnings.append(
                f"iterations ({self.iterations}) < candidate moves at root ({len(root_moves)}): "
                "some moves were never tried."
            )
        self.last_stats = self._stats(iterations=self.iterations, time_s=time.perf_counter() - start,
                                      best_by_n=by_n.move, best_by_wn=by_wn.move, warnings=warnings)
        return chosen.move

    def _iterate(self, root: Node, game: TicTacToeGame, radius: Optional[int]) -> None:
        node = root
        sim = game.copy()

        # 1. Selection
        while not node.untried and node.children:
            node = node.best_child_uct(self.uct_c, self.rng)
            sim.make_move(node.move)

        # 2. Expansion (a terminal node has no untried moves, so it is never expanded - MC-EC1.1)
        if node.untried:
            move = node.untried[self.rng.randrange(len(node.untried))]
            # Read the mover BEFORE make_move: game.py keeps current_player unchanged when a move ends the game.
            mover = sim.current_player
            sim.make_move(move)
            node = node.add_child(move, mover, candidate_moves(sim, radius))
            self._tree_size += 1

        # 3. Simulation
        policy = self.rollout_policy
        while not sim.is_terminal():
            rollout_move = policy(sim)
            if rollout_move is None or not sim.make_move(rollout_move):
                break  # MC-EC5.3: policy gave up -> score the position as it stands

        # 4. Backpropagation
        reward = self._reward(sim.get_result(), node.player_just_moved)
        while node is not None:
            node.N += 1
            node.W += reward
            reward = 1.0 - reward
            node = node.parent

    @staticmethod
    def _reward(result: Optional[int], player: int) -> float:
        """1 / 0.5 / 0 for ``player``; an unfinished game (None) counts as a draw."""
        if result == player:
            return 1.0
        if result in (PLAYER_X, PLAYER_O):
            return 0.0
        return 0.5

    def _pick_final(self, root: Node) -> Tuple[Node, Node]:
        """(best by N, best by W/N). Ties: N then W/N (or W/N then N), then the seeded RNG."""
        visited = [child for child in root.children if child.N > 0]
        by_n = self._argmax(visited, lambda ch: (ch.N, ch.W / ch.N))
        by_wn = self._argmax(visited, lambda ch: (ch.W / ch.N, ch.N))
        return by_n, by_wn

    def _argmax(self, nodes: Sequence[Node], key) -> Node:
        best_key = max(key(node) for node in nodes)
        best = [node for node in nodes if key(node) == best_key]
        return best[0] if len(best) == 1 else self.rng.choice(best)

    def _effective_radius(self, game: TicTacToeGame) -> Optional[int]:
        if self.candidate_radius is not None:
            return self.candidate_radius
        return None if game.board_size <= SMALL_BOARD_MAX else DEFAULT_RADIUS

    # ---------------------------------------------------------------- reporting
    def root_stats(self) -> List[Dict[str, object]]:
        """W, N, W/N and final UCT of every expanded root child, sorted by N (desc)."""
        if self._root is None:
            return []
        root = self._root
        rows = [
            {
                "move": child.move,
                "W": float(child.W),
                "N": child.N,
                "WN": child.W / child.N if child.N else 0.0,
                "uct": uct_score(child.W, child.N, root.N, self.uct_c),
            }
            for child in root.children
        ]
        rows.sort(key=lambda row: (-row["N"], -row["WN"]))
        return rows

    def _stats(self, iterations, time_s, best_by_n, best_by_wn, warnings) -> Dict[str, object]:
        return {
            "iterations": iterations,
            "tree_size": self._tree_size if self._root is not None else 0,
            "time_s": time_s,
            "c": self.uct_c,
            "rollout": getattr(self.rollout_policy, "__name__", repr(self.rollout_policy)),
            "final_policy": self.final_policy,
            "best_by_n": best_by_n,
            "best_by_wn": best_by_wn,
            "max_n_equals_max_wn": best_by_n == best_by_wn,
            "warnings": warnings,
        }
