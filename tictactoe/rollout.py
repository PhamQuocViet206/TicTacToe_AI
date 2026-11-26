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

WIN_ATTACK_SCORE = 100_000
WIN_DEFENSE_SCORE = 90_000

# Bảng trọng số Tấn công (player) và Phòng thủ (opponent) theo (chiều dài chuỗi, số đầu mở)
# Chuẩn hóa cho luật 5 quân liên tiếp (k = 5):
ATTACK_WEIGHTS: Dict[Tuple[int, int], int] = {
    (4, 2): 10_000,  # Tứ mở 2 đầu: thế thắng chắc chắn ở lượt kế
    (4, 1): 1_500,   # Tứ mở 1 đầu: nước chiếu trực tiếp
    (3, 2): 1_000,   # Tam mở 2 đầu: đe dọa tạo Tứ mở 2 đầu
    (3, 1): 150,     # Tam mở 1 đầu: tiềm năng phát triển
    (2, 2): 50,      # Nhị mở 2 đầu: khung phát triển chuỗi sớm
    (2, 1): 10,      # Nhị mở 1 đầu: liên kết cơ bản
    (1, 2): 2,       # Quân đơn lẻ có không gian mở 2 phía
}

DEFENSE_WEIGHTS: Dict[Tuple[int, int], int] = {
    (4, 2): 9_000,   # Chặn Tứ mở 2 đầu của đối thủ
    (4, 1): 1_200,   # Chặn Tứ mở 1 đầu của đối thủ
    (3, 2): 850,     # Chặn Tam mở 2 đầu của đối thủ
    (3, 1): 120,     # Chặn Tam mở 1 đầu của đối thủ
    (2, 2): 40,      # Chặn Nhị mở 2 đầu của đối thủ
    (2, 1): 8,       # Chặn Nhị mở 1 đầu của đối thủ
    (1, 2): 1,       # Cản hướng phát triển đơn lẻ
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
    win_length = game.win_length

    # Đếm số quân liên tiếp theo chiều thuận (+dr, +dc)
    forward_count = 0
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board[r][c] == player:
        forward_count += 1
        r += dr
        c += dc
    forward_open = 1 if (0 <= r < size and 0 <= c < size and board[r][c] == EMPTY) else 0

    # Đếm số quân liên tiếp theo chiều nghịch (-dr, -dc)
    backward_count = 0
    rb, cb = row - dr, col - dc
    while 0 <= rb < size and 0 <= cb < size and board[rb][cb] == player:
        backward_count += 1
        rb -= dr
        cb -= dc
    backward_open = 1 if (0 <= rb < size and 0 <= cb < size and board[rb][cb] == EMPTY) else 0

    count = 1 + forward_count + backward_count
    open_ends = forward_open + backward_open

    # Nếu hành lang bị chặn hẹp không đủ chỗ cho win_length quân thì vô giá trị (0 đầu mở)
    if count < win_length and open_ends > 0:
        space = count
        opponent = -player
        rf, cf = r, c
        while space < win_length and 0 <= rf < size and 0 <= cf < size and board[rf][cf] != opponent:
            space += 1
            rf += dr
            cf += dc
        r_back, c_back = rb, cb
        while (
            space < win_length
            and 0 <= r_back < size
            and 0 <= c_back < size
            and board[r_back][c_back] != opponent
        ):
            space += 1
            r_back -= dr
            c_back -= dc
        if space < win_length:
            return count, 0

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


def _score_direction(count: int, open_ends: int, win_length: int, is_attack: bool) -> int:
    """Look up tactical weight for a single direction based on chain length and open ends."""
    if count >= win_length:
        return WIN_ATTACK_SCORE if is_attack else WIN_DEFENSE_SCORE
    if open_ends <= 0:
        return 0

    # Chuẩn hóa chiều dài chuỗi theo mốc k = 5 để tương thích mọi win_length (ví dụ k = 3 hoặc k = 5)
    if count == 1:
        effective_count = 1
    else:
        effective_count = max(2, min(4, count + (5 - win_length)))

    weights = ATTACK_WEIGHTS if is_attack else DEFENSE_WEIGHTS
    return weights.get((effective_count, open_ends), 0)


def _evaluate_move(game: TicTacToeGame, move: Move, player: int) -> int:
    """Compute combined attack, defense, and center-proximity score for a candidate move."""
    row, col = move
    board = game.board
    size = game.board_size
    opponent = -player
    win_length = game.win_length
    total_score = 0

    for dr, dc in DIRECTIONS:
        r1, c1 = row + dr, col + dc
        r2, c2 = row - dr, col - dc
        v1 = board[r1][c1] if (0 <= r1 < size and 0 <= c1 < size) else None
        v2 = board[r2][c2] if (0 <= r2 < size and 0 <= c2 < size) else None

        if v1 == player or v2 == player:
            atk_count, atk_open = _analyze_direction(game, row, col, dr, dc, player)
            total_score += _score_direction(atk_count, atk_open, win_length, is_attack=True)
        elif v1 == EMPTY and v2 == EMPTY:
            total_score += ATTACK_WEIGHTS[(1, 2)]

        if v1 == opponent or v2 == opponent:
            def_count, def_open = _analyze_direction(game, row, col, dr, dc, opponent)
            total_score += _score_direction(def_count, def_open, win_length, is_attack=False)
        elif v1 == EMPTY and v2 == EMPTY:
            total_score += DEFENSE_WEIGHTS[(1, 2)]

    center = size // 2
    distance = max(abs(row - center), abs(col - center))
    center_bonus = max(0, min(3, center - distance))
    return total_score + center_bonus


def heuristic_rollout(game: TicTacToeGame) -> Optional[Move]:
    """Select a rollout move using tactical priority rules and line-potential scoring.

    Priority order:
    1. Immediate win for the current player.
    2. Block immediate opponent win (if multiple threats exist, pick the highest-rated block).
    3. Maximize combined attack + defense line potential across all four directions.
    """
    candidates = _get_candidate_moves(game)
    if not candidates:
        return None

    player = game.current_player
    opponent = -player

    # 1. Ưu tiên nước thắng ngay lập tức cho người chơi hiện tại
    winning_moves = [move for move in candidates if _is_winning_move(game, move, player)]
    if winning_moves:
        return random.choice(winning_moves)

    # 2. Chặn nước thắng ngay của đối thủ (xử lý cả trường hợp có nhiều mối đe dọa)
    blocking_moves = [move for move in candidates if _is_winning_move(game, move, opponent)]
    if len(blocking_moves) == 1:
        return blocking_moves[0]
    if len(blocking_moves) > 1:
        best_block_score = -1
        best_blocks: List[Move] = []
        for move in blocking_moves:
            score = _evaluate_move(game, move, player)
            if score > best_block_score:
                best_block_score = score
                best_blocks = [move]
            elif score == best_block_score:
                best_blocks.append(move)
        return random.choice(best_blocks)

    # 3. Đánh giá tiềm năng phát triển chuỗi quân và chặn đa đe dọa (fork)
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
