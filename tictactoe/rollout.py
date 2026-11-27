"""Move-selection policies for Monte Carlo Tree Search (MCTS) rollouts."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

try:
    from .constants import EMPTY
    from .game import TicTacToeGame
except ImportError:  # pragma: no cover
    from constants import EMPTY
    from game import TicTacToeGame

Move = Tuple[int, int]
RolloutPolicy = Callable[[TicTacToeGame], Optional[Move]]

DIRECTIONS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
SMALL_BOARD_MAX = 3
CANDIDATE_RADIUS = 1

# Bảng trọng số 6 mức theo (chiều dài chuỗi, số đầu mở) cho luật thắng 5 quân (win_length = 5)
ATTACK_WEIGHTS: Dict[Tuple[int, int], int] = {
    (4, 2): 10_000,  # Tứ mở 2 đầu
    (4, 1): 2_000,   # Tứ mở 1 đầu
    (3, 2): 800,     # Tam mở 2 đầu
    (3, 1): 200,     # Tam mở 1 đầu
    (2, 2): 60,      # Nhị mở 2 đầu
    (2, 1): 15,      # Nhị mở 1 đầu
}

DEFENSE_WEIGHTS: Dict[Tuple[int, int], int] = {
    (4, 2): 8_000,   # Chặn đối thủ tạo Tứ mở 2 đầu
    (4, 1): 1_500,   # Chặn đối thủ tạo Tứ mở 1 đầu
    (3, 2): 600,     # Chặn đối thủ tạo Tam mở 2 đầu
    (3, 1): 120,     # Chặn đối thủ tạo Tam mở 1 đầu
    (2, 2): 30,      # Chặn đối thủ tạo Nhị mở 2 đầu
    (2, 1): 10,      # Chặn đối thủ tạo Nhị mở 1 đầu
}


def random_rollout(game: TicTacToeGame) -> Optional[Move]:
    """Choose a uniformly random legal move for a rollout."""
    valid_moves = game.get_valid_moves()
    return random.choice(valid_moves) if valid_moves else None


def _get_candidate_moves(
    game: TicTacToeGame,
    radius: int = CANDIDATE_RADIUS,
) -> List[Move]:
    """Return candidate empty cells near existing stones for rollout evaluation."""
    if game.is_terminal():
        return []

    size = game.board_size
    if size <= SMALL_BOARD_MAX:
        return game.get_valid_moves()

    board = game.board
    seen = set()
    candidates: List[Move] = []
    has_stone = False

    for row, line in enumerate(board):
        for col, cell in enumerate(line):
            if cell == EMPTY:
                continue
            has_stone = True
            r_min = 0 if row - radius < 0 else row - radius
            r_max = size if row + radius + 1 > size else row + radius + 1
            c_min = 0 if col - radius < 0 else col - radius
            c_max = size if col + radius + 1 > size else col + radius + 1
            for r in range(r_min, r_max):
                row_cells = board[r]
                for c in range(c_min, c_max):
                    if row_cells[c] == EMPTY and (r, c) not in seen:
                        seen.add((r, c))
                        candidates.append((r, c))

    if not has_stone:
        center = size // 2
        if board[center][center] == EMPTY:
            return [(center, center)]
        return game.get_valid_moves()

    return candidates if candidates else game.get_valid_moves()


def _analyze_direction(
    game: TicTacToeGame,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> Tuple[int, int]:
    """Count contiguous stones and open ends if `player` places at (row, col)."""
    board = game.board
    size = game.board_size

    # 1. Đếm số quân liên tiếp cùng loại theo chiều thuận (+dr, +dc)
    forward_count = 0
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board[r][c] == player:
        forward_count += 1
        r += dr
        c += dc
    forward_open = 1 if (0 <= r < size and 0 <= c < size and board[r][c] == EMPTY) else 0

    # 2. Đếm số quân liên tiếp cùng loại theo chiều nghịch (-dr, -dc)
    backward_count = 0
    rb, cb = row - dr, col - dc
    while 0 <= rb < size and 0 <= cb < size and board[rb][cb] == player:
        backward_count += 1
        rb -= dr
        cb -= dc
    backward_open = 1 if (0 <= rb < size and 0 <= cb < size and board[rb][cb] == EMPTY) else 0

    count = 1 + forward_count + backward_count
    open_ends = forward_open + backward_open
    return count, open_ends


def _is_winning_move(game: TicTacToeGame, move: Move, player: int) -> bool:
    """Check directly on the board whether placing `player` at `move` forms a win."""
    row, col = move
    board = game.board
    size = game.board_size
    win_length = game.win_length

    for dr, dc in DIRECTIONS:
        r1, c1 = row + dr, col + dc
        r2, c2 = row - dr, col - dc
        has1 = 0 <= r1 < size and 0 <= c1 < size and board[r1][c1] == player
        has2 = 0 <= r2 < size and 0 <= c2 < size and board[r2][c2] == player
        if not has1 and not has2:
            continue

        count = 1
        if has1:
            count += 1
            r, c = r1 + dr, c1 + dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r += dr
                c += dc
        if has2:
            count += 1
            r, c = r2 - dr, c2 - dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

        if count >= win_length:
            return True
    return False


def _score_direction(count: int, open_ends: int, is_attack: bool) -> int:
    """Look up tactical weight for a single direction based on chain length and open ends."""
    if count < 2 or open_ends <= 0:
        return 0
    weights = ATTACK_WEIGHTS if is_attack else DEFENSE_WEIGHTS
    return weights.get((min(count, 4), open_ends), 0)


def _evaluate_move(game: TicTacToeGame, move: Move, player: int) -> int:
    """Compute combined attack and defense score across all four directions for a candidate move."""
    row, col = move
    board = game.board
    size = game.board_size
    opponent = -player
    total_score = 0

    for dr, dc in DIRECTIONS:
        r1, c1 = row + dr, col + dc
        r2, c2 = row - dr, col - dc
        v1 = board[r1][c1] if (0 <= r1 < size and 0 <= c1 < size) else None
        v2 = board[r2][c2] if (0 <= r2 < size and 0 <= c2 < size) else None

        if v1 == player or v2 == player:
            atk_count, atk_open = _analyze_direction(game, row, col, dr, dc, player)
            total_score += _score_direction(atk_count, atk_open, is_attack=True)

        if v1 == opponent or v2 == opponent:
            def_count, def_open = _analyze_direction(game, row, col, dr, dc, opponent)
            total_score += _score_direction(def_count, def_open, is_attack=False)

    return total_score


def heuristic_rollout(game: TicTacToeGame) -> Optional[Move]:
    """Select a rollout move using tactical priority rules and line-potential scoring.

    Steps:
    1. Filter candidate moves near existing stones (Position).
    2. Take immediate win for the current player if available.
    3. Block immediate opponent win if available (break ties using _evaluate_move).
    4. Score all candidates by Attack + Defense across four directions.
    5. Randomly choose among candidates tied for the highest score.
    """
    # Bước 1: Lọc ứng viên theo vị trí (Position)
    candidates = _get_candidate_moves(game)
    if not candidates:
        return None

    player = game.current_player
    opponent = -player

    # Bước 2: Thắng ngay cho người chơi hiện tại
    winning_moves = [move for move in candidates if _is_winning_move(game, move, player)]
    if winning_moves:
        return random.choice(winning_moves)

    # Bước 3: Chặn đối thủ thắng ngay
    blocking_moves = [move for move in candidates if _is_winning_move(game, move, opponent)]
    if len(blocking_moves) == 1:
        return blocking_moves[0]
    if len(blocking_moves) > 1:
        candidates = blocking_moves

    # Bước 4 & Bước 5: Đánh giá chiến thuật (Attack + Defense) và chọn ngẫu nhiên trong nhóm điểm cao nhất
    best_score = -1
    best_moves: List[Move] = []
    for move in candidates:
        score = _evaluate_move(game, move, player)
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return random.choice(best_moves)


ROLLOUT_POLICIES = {
    "random": random_rollout,
    "heuristic": heuristic_rollout,
}
