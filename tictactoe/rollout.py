"""Move-selection policies for Monte Carlo rollouts."""

from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

try:
    from .game import TicTacToeGame
except ImportError:  # pragma: no cover
    from game import TicTacToeGame

Move = Tuple[int, int]
RolloutPolicy = Callable[[TicTacToeGame], Optional[Move]]


def random_rollout(game: TicTacToeGame) -> Optional[Move]:
    """Choose a uniformly random legal move for a rollout."""
    valid_moves = game.get_valid_moves()
    return random.choice(valid_moves) if valid_moves else None


def heuristic_rollout(game: TicTacToeGame) -> Optional[Move]:
    """Prefer an immediate win, then block an immediate loss, else choose randomly."""
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None

    player = game.current_player
    for move in valid_moves:
        simulated = game.copy()
        if simulated.make_move(move) and simulated.winner == player:
            return move

    opponent = -player
    for move in valid_moves:
        simulated = game.copy()
        simulated.current_player = opponent
        if simulated.make_move(move, player=opponent) and simulated.winner == opponent:
            return move

    return random.choice(valid_moves)


ROLLOUT_POLICIES = {
    "random": random_rollout,
    "heuristic": heuristic_rollout,
}
