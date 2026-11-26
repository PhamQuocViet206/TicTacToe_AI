"""Minimax and alpha-beta search for Tic-Tac-Toe."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .constants import EMPTY, PLAYER_O, PLAYER_X
    from .game import TicTacToeGame
except ImportError:  # pragma: no cover
    from constants import EMPTY, PLAYER_O, PLAYER_X
    from game import TicTacToeGame

Move = Tuple[int, int]
WIN_SCORE = 1_000_000
INFINITY = WIN_SCORE * 10

# Default search depth when ``depth=None`` ("auto"). The search is exhaustive only
# when the remaining game is small enough to finish quickly, so it always terminates:
#   * with alpha-beta  : up to 12 empty cells (measured <= ~3 s on 4x4/5x5 boards)
#   * without pruning  : up to 9 empty cells  (a full 3x3 game; ~550k nodes)
# Larger positions use a fixed depth with the heuristic evaluation.
FULL_SEARCH_MAX_EMPTY_AB = 10
FULL_SEARCH_MAX_EMPTY_PLAIN = 8
FULL_SEARCH_MAX_EMPTY_PLAIN_3X3 = 9
AUTO_DEPTH = 3
AUTO_FULL_SEARCH_NODE_BUDGET = 100_000


class MinimaxAI:
    """Deterministic Minimax agent with an optional alpha-beta pruner."""

    def __init__(
        self,
        depth: Optional[int] = None,
        max_depth: Optional[int] = None,
        alpha_beta: bool = True,
        use_alpha_beta: Optional[bool] = None,
        max_nodes: Optional[int] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ):
        if max_depth is not None:
            depth = max_depth
        if use_alpha_beta is not None:
            alpha_beta = use_alpha_beta

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MinimaxAI argument(s): {unexpected}")
        if depth is not None:
            if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
                raise ValueError("depth must be a positive integer or None.")
        if not isinstance(alpha_beta, bool):
            raise ValueError("alpha_beta must be a boolean.")
        if max_nodes is not None and (
            isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int)
            or max_nodes < 1
        ):
            raise ValueError("max_nodes must be a positive integer or None.")
        if time_limit is not None:
            try:
                time_limit = float(time_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError("time_limit must be a positive number or None.") from exc
            if time_limit < 0:
                raise ValueError("time_limit must be a positive number or None.")

        self.depth = depth
        self.max_depth = depth
        self.pruning_enabled = alpha_beta
        self.use_alpha_beta = alpha_beta
        self.alpha_beta_enabled = alpha_beta
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self.last_stats: Dict[str, object] = {}
        self.nodes_searched = 0
        self.pruned_nodes = 0
        self._ai_player = PLAYER_X
        self._search_started = 0.0
        self._search_aborted = False

    @property
    def name(self) -> str:
        return f"Minimax(depth={self.depth if self.depth is not None else 'full'})"

    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        """Return a legal move without modifying the supplied game."""
        if game.is_terminal() or not game.get_valid_moves():
            self.last_stats = self._make_stats(
                0, 0, 0.0, None, ["Game is over: no move to search."]
            )
            return None

        start = time.perf_counter()
        self.nodes_searched = 0
        self.pruned_nodes = 0
        self._search_started = start
        self._search_aborted = False
        self._ai_player = game.current_player

        moves = self._candidate_moves(game)
        if not moves:
            return None
        if len(moves) == 1:
            move = moves[0]
            self.last_stats = self._make_stats(
                1, 0, time.perf_counter() - start, move, []
            )
            return move

        search_depth = self._choose_depth(game)

        ordered = self._order_moves(game, moves, self._ai_player)

        for move in ordered:
            if self._is_immediate_win(game, move, self._ai_player):
                elapsed = time.perf_counter() - start
                self.last_stats = self._make_stats(
                    1, 0, elapsed, move, []
                )
                return move

        for move in ordered:
            if self._is_immediate_win(game, move, -self._ai_player):
                elapsed = time.perf_counter() - start
                self.last_stats = self._make_stats(
                    1, 0, elapsed, move, []
                )
                return move

        best_move: Optional[Move] = None
        best_score = -INFINITY
        alpha = -INFINITY
        beta = INFINITY
        fallback_warning = False
        exhaustive_auto = (
            self.depth is None
            and self.pruning_enabled
            and search_depth
            == game.board_size * game.board_size - game.moves_count
            and self.max_nodes is None
        )
        if exhaustive_auto:
            self.max_nodes = AUTO_FULL_SEARCH_NODE_BUDGET

        def run_root_search(depth: int) -> None:
            nonlocal best_move, best_score, alpha
            best_move = None
            best_score = -INFINITY
            alpha = -INFINITY
            for move in ordered:
                state = game.copy()
                if not state.make_move(move, self._ai_player):
                    continue
                score = self._search(
                    state,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    pruning=self.pruning_enabled,
                )
                if score > best_score:
                    best_score = score
                    best_move = move
                if self.pruning_enabled:
                    alpha = max(alpha, best_score)

        run_root_search(search_depth)

        if exhaustive_auto and self._search_aborted:
            fallback_warning = True
            self.nodes_searched = 0
            self.pruned_nodes = 0
            self._search_aborted = False
            self.max_nodes = None
            run_root_search(AUTO_DEPTH)
        elif exhaustive_auto:
            self.max_nodes = None

        elapsed = time.perf_counter() - start
        self.last_stats = self._make_stats(
            self.nodes_searched,
            self.pruned_nodes,
            elapsed,
            best_move,
            (
                ["Exhaustive search exceeded the node budget; depth-3 fallback used."]
                if fallback_warning
                else (["Search limit reached."] if self._search_aborted else "")
            ),
        )
        if self.last_stats["warnings"] == [""]:
            self.last_stats["warnings"] = []
        return best_move

    def _choose_depth(self, game: TicTacToeGame) -> int:
        """Depth for ``get_move``: the user's value, or an automatic bounded one."""
        if self.depth is not None:
            return self.depth
        empties = game.board_size * game.board_size - game.moves_count
        if not self.pruning_enabled:
            if game.board_size == 3:
                limit = FULL_SEARCH_MAX_EMPTY_PLAIN_3X3
            else:
                limit = FULL_SEARCH_MAX_EMPTY_PLAIN
        else:
            limit = FULL_SEARCH_MAX_EMPTY_AB
        return empties if empties <= limit else AUTO_DEPTH

    def _search(
        self,
        game: TicTacToeGame,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        pruning: bool = True,
    ) -> float:
        self.nodes_searched += 1

        if self.max_nodes is not None and self.nodes_searched > self.max_nodes:
            self._search_aborted = True
        if self.time_limit is not None:
            if time.perf_counter() - self._search_started >= self.time_limit:
                self._search_aborted = True
        if self._search_aborted:
            return self.evaluate(game)

        result = game.get_result()
        if result is not None:
            if result == self._ai_player:
                return WIN_SCORE + depth
            if result == -self._ai_player:
                return -WIN_SCORE - depth
            return 0
        if depth <= 0:
            return self.evaluate(game)

        moves = self._candidate_moves(game)
        ordered = self._order_moves(game, moves, game.current_player)

        if maximizing:
            value = -INFINITY
            for move in ordered:
                state = game.copy()
                if not state.make_move(move, game.current_player):
                    continue
                value = max(
                    value,
                    self._search(
                        state, depth - 1, alpha, beta, False, pruning
                    ),
                )
                if pruning:
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        self.pruned_nodes += 1
                        break
            return value

        value = INFINITY
        for move in ordered:
            state = game.copy()
            if not state.make_move(move, game.current_player):
                continue
            value = min(
                value,
                self._search(
                    state, depth - 1, alpha, beta, True, pruning
                ),
            )
            if pruning:
                beta = min(beta, value)
                if beta <= alpha:
                    self.pruned_nodes += 1
                    break
        return value

    def evaluate(self, game: TicTacToeGame) -> int:
        """Heuristic evaluation from the AI player's perspective."""
        if game.winner is not None:
            return (
                WIN_SCORE
                if game.winner == self._ai_player
                else -WIN_SCORE
            )

        size = game.board_size
        k = game.win_length
        center = size // 2
        score = 0

        # Position score is modest; it breaks ties and favors central play.
        for row in range(size):
            for col in range(size):
                value = game.board[row][col]
                if value == EMPTY:
                    continue
                distance = max(abs(row - center), abs(col - center))
                sign = 1 if value == self._ai_player else -1
                score += sign * (size - distance)

        for row in range(size):
            for col in range(size):
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    end_r = row + dr * (k - 1)
                    end_c = col + dc * (k - 1)
                    if not game.is_inside_board(end_r, end_c):
                        continue
                    window = [
                        game.board[row + i * dr][col + i * dc]
                        for i in range(k)
                    ]
                    score += self._score_window(window)
        return score

    def _score_window(self, window: Sequence[int]) -> int:
        ai_count = window.count(self._ai_player)
        opponent_count = window.count(-self._ai_player)
        if ai_count and opponent_count:
            return 0

        weights = self._window_weights(len(window))
        if ai_count:
            return weights[ai_count]
        if opponent_count:
            return -weights[opponent_count] * 2
        return 0

    _weights_cache: Dict[int, List[int]] = {}

    @classmethod
    def _window_weights(cls, k: int) -> List[int]:
        """Score by number of stones in a window of length ``k``: 0, 1, 4, 16, ..."""
        weights = cls._weights_cache.get(k)
        if weights is None:
            weights = [0] + [4 ** count for count in range(k)]
            cls._weights_cache[k] = weights
        return weights

    def _candidate_moves(self, game: TicTacToeGame) -> List[Move]:
        valid = game.get_valid_moves()
        if not valid:
            return []
        if game.board_size <= 5:
            return valid

        occupied = [
            (row, col)
            for row in range(game.board_size)
            for col in range(game.board_size)
            if game.board[row][col] != EMPTY
        ]
        if not occupied:
            center = game.board_size // 2
            return [(center, center)]

        radius = 2
        seen: set[Move] = set()
        near: List[Move] = []
        for row, col in occupied:
            for r in range(max(0, row - radius), min(game.board_size, row + radius + 1)):
                for c in range(max(0, col - radius), min(game.board_size, col + radius + 1)):
                    move = (r, c)
                    if game.board[r][c] == EMPTY and move not in seen:
                        seen.add(move)
                        near.append(move)
        return near or valid

    def _is_immediate_win(
        self, game: TicTacToeGame, move: Move, player: int
    ) -> bool:
        row, col = move
        if not game.is_inside_board(row, col) or game.board[row][col] != EMPTY:
            return False
        game.board[row][col] = player
        try:
            return game.check_winner(move, player) == player
        finally:
            game.board[row][col] = EMPTY

    def _order_moves(
        self,
        game: TicTacToeGame,
        moves: Iterable[Move],
        player: int,
    ) -> List[Move]:
        center = game.board_size // 2

        def score(move: Move) -> Tuple[int, int, int]:
            row, col = move
            wins = 2 if self._is_immediate_win(game, move, player) else 0
            blocks = (
                1
                if self._is_immediate_win(game, move, -player)
                else 0
            )
            distance = abs(row - center) + abs(col - center)
            return (-wins - blocks, distance, row + col)

        return sorted(list(moves), key=score)

    def minimax(
        self,
        game: TicTacToeGame,
        depth: int,
        alpha: float = -INFINITY,
        beta: float = INFINITY,
        maximizing: Optional[bool] = None,
    ) -> float:
        """Run pure minimax (without alpha-beta pruning)."""
        return self._run_search(game, depth, alpha, beta, maximizing, pruning=False)

    def alpha_beta(
        self,
        game: TicTacToeGame,
        depth: int,
        alpha: float = -INFINITY,
        beta: float = INFINITY,
        maximizing: Optional[bool] = None,
    ) -> float:
        """Run minimax with alpha-beta pruning."""
        return self._run_search(game, depth, alpha, beta, maximizing, pruning=True)

    def _run_search(
        self,
        game: TicTacToeGame,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: Optional[bool],
        pruning: bool,
    ) -> float:
        """Direct entry point: ``maximizing`` says whether the side to move maximizes.

        ``None`` or ``True`` -> the side to move is the maximizing (AI) player;
        ``False`` -> the side to move is the opponent of the maximizing player.
        """
        if maximizing is None:
            maximizing = True
        self._ai_player = game.current_player if maximizing else -game.current_player
        self._search_started = time.perf_counter()
        self.nodes_searched = 0
        self.pruned_nodes = 0
        self._search_aborted = False
        return self._search(game, depth, alpha, beta, maximizing, pruning=pruning)

    def alphabeta(self, *args, **kwargs) -> float:
        return self.alpha_beta(*args, **kwargs)

    def _alphabeta(self, *args, **kwargs) -> float:
        return self.alpha_beta(*args, **kwargs)

    def _minimax(self, *args, **kwargs) -> float:
        return self.minimax(*args, **kwargs)

    def _make_stats(
        self,
        nodes: int,
        pruned: int,
        time_s: float,
        best_move: Optional[Move],
        warnings: List[str],
    ) -> Dict[str, object]:
        return {
            "nodes": nodes,
            "nodes_searched": nodes,
            "pruned": pruned,
            "pruned_nodes": pruned,
            "time_s": time_s,
            "depth": self.depth,
            "alpha_beta": self.pruning_enabled,
            "best_move": best_move,
            "warnings": warnings,
        }
