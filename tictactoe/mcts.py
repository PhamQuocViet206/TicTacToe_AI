"""Placeholder module for MCTS + UCT integration.

This file can be filled by a teammate working on the MCTS implementation.
Rollout policies are supplied separately from the MCTS algorithm; see
``tictactoe.rollout`` for the available policies.
"""

from __future__ import annotations

from typing import Optional

try:
    from .rollout import RolloutPolicy, random_rollout
except ImportError:  # pragma: no cover
    from rollout import RolloutPolicy, random_rollout


class MCTSAgent:
    """Placeholder MCTS agent with a separately configurable rollout policy."""

    def __init__(
        self,
        iterations: int = 1000,
        uct_c: float = 1.4,
        rollout_policy: Optional[RolloutPolicy] = None,
    ):
        self.iterations = max(1, int(iterations))
        self.uct_c = float(uct_c)
        self.rollout_policy = rollout_policy or random_rollout

    def get_move(self, game):
        raise NotImplementedError("MCTS AI has not been implemented yet.")
