"""Base TicTacToe game state and rules."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

try:
    from .constants import BOARD_SIZE, EMPTY, PLAYER_O, PLAYER_X, WIN_LENGTH
except ImportError:  # pragma: no cover
    from constants import BOARD_SIZE, EMPTY, PLAYER_O, PLAYER_X, WIN_LENGTH

Move = Tuple[int, int]


class TicTacToeGame:
    """Manage the core TicTacToe rules and game state.

    The game board is a 15x15 matrix with 0 for empty squares, 1 for X and -1 for O.
    Winning is defined as ``count >= win_length`` in any of the four directions.
    """

    def __init__(self, board_size: int = BOARD_SIZE, win_length: int = WIN_LENGTH):
        if board_size < win_length:
            raise ValueError("board_size must be greater than or equal to win_length.")
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.reset()

    def reset(self) -> None:
        """Reset the current state to a fresh game."""
        self.board: List[List[int]] = [[EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = PLAYER_X
        self.winner: Optional[int] = None
        self.moves_count = 0
        self.last_move: Optional[Move] = None

    def is_inside_board(self, row: int, col: int) -> bool:
        """Return whether a coordinate is on the board."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def is_valid_move(self, move: Sequence[int]) -> bool:
        """Check whether a move is legal in the current state."""
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return False

        row, col = int(move[0]), int(move[1])
        if not self.is_inside_board(row, col):
            return False
        if self.board[row][col] != EMPTY:
            return False
        if self.is_terminal():
            return False
        return True

    def get_valid_moves(self) -> List[Move]:
        """Return all currently legal moves."""
        if self.is_terminal():
            return []
        valid_moves: List[Move] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == EMPTY:
                    valid_moves.append((row, col))
        return valid_moves

    def make_move(self, move: Sequence[int], player: Optional[int] = None) -> bool:
        """Attempt to place a stone.

        If ``player`` is omitted, the current turn is used. A move is rejected if the
        game is already over, the move is out of range, the target cell is occupied, or
        the provided player is not the one whose turn it is.
        """
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return False

        row, col = int(move[0]), int(move[1])
        if not self.is_valid_move((row, col)):
            return False

        if player is not None:
            if player not in (PLAYER_X, PLAYER_O):
                return False
            if player != self.current_player:
                return False
        else:
            player = self.current_player

        self.board[row][col] = player
        self.last_move = (row, col)
        self.moves_count += 1

        winner = self.check_winner((row, col), player)
        if winner is not None:
            self.winner = winner
            return True

        if self.is_draw():
            return True

        self.current_player = -player
        return True

    def check_winner(self, move: Optional[Sequence[int]] = None, player: Optional[int] = None) -> Optional[int]:
        """Return the winning player if a win is found; otherwise return None."""
        if move is None and player is None:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    value = self.board[row][col]
                    if value != EMPTY:
                        winner = self.check_winner((row, col), value)
                        if winner is not None:
                            return winner
            return None

        if move is None:
            return None

        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return None

        row, col = int(move[0]), int(move[1])
        if not self.is_inside_board(row, col):
            return None

        if player is None:
            player = self.board[row][col]
        if player not in (PLAYER_X, PLAYER_O):
            return None

        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1 + self._count_direction(row, col, dr, dc, player) + self._count_direction(row, col, -dr, -dc, player)
            if count >= self.win_length:
                return player
        return None

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        """Count contiguous stones in one direction from a position."""
        count = 0
        r = row + dr
        c = col + dc
        while self.is_inside_board(r, c) and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def is_draw(self) -> bool:
        """Return whether the board is full without a winner."""
        return self.winner is None and self.moves_count >= self.board_size * self.board_size

    def is_terminal(self) -> bool:
        """Return whether the game has ended in either win or draw."""
        return self.winner is not None or self.is_draw()

    def get_result(self) -> Optional[int]:
        """Return 1 for X win, -1 for O win, 0 for draw, None if the game is unfinished."""
        if self.winner == PLAYER_X:
            return PLAYER_X
        if self.winner == PLAYER_O:
            return PLAYER_O
        if self.is_draw():
            return 0
        return None

    def copy(self) -> "TicTacToeGame":
        """Make a deep, independent copy of the current game state."""
        new_game = TicTacToeGame(board_size=self.board_size, win_length=self.win_length)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.moves_count = self.moves_count
        new_game.last_move = self.last_move
        return new_game

    def display(self) -> None:
        """Print the board in the terminal for debugging and demonstrations."""
        header = "   " + " ".join(f"{col:2}" for col in range(self.board_size))
        print(header)
        for row_index, row in enumerate(self.board):
            values = " ".join(f"{cell:2}" for cell in row)
            print(f"{row_index:2} {values}")

    def __repr__(self) -> str:
        return f"TicTacToeGame(board_size={self.board_size}, win_length={self.win_length}, current_player={self.current_player}, winner={self.winner})"
