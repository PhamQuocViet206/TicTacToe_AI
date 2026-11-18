"""AI integration layer for the GUI."""

from __future__ import annotations

import random
from typing import Optional, Tuple

try:
    from .game import TicTacToeGame
    from .mcts import MCTSAgent
    from .rollout import ROLLOUT_POLICIES, RolloutPolicy
except ImportError:  # pragma: no cover
    from game import TicTacToeGame
    from mcts import MCTSAgent
    from rollout import ROLLOUT_POLICIES, RolloutPolicy

Move = Tuple[int, int]


class AIController:
    """Bridge between GUI and concrete AI algorithms.

    This controller keeps the GUI independent from any specific algorithm. The
    Random is a temporary standalone demo algorithm. Rollout policies are separate
    from game-playing algorithms and are intended for MCTS simulations.
    """

    def __init__(
        self,
        algorithm: str = "random",
        iterations: int = 1000,
        uct_c: float = 1.4,
        rollout_policy: str = "random",
    ):
        self.algorithm = algorithm.lower()
        self.iterations = max(1, int(iterations))
        self.uct_c = float(uct_c)
        self.rollout_policy_name = rollout_policy.lower()
        self.rollout_policy: Optional[RolloutPolicy] = ROLLOUT_POLICIES.get(self.rollout_policy_name)
        self.last_status = "AI ready"
        self.last_agent: Optional[MCTSAgent] = None

    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        """Return the move selected by the active AI or None when unavailable."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            self.last_status = "No legal moves available."
            return None

        if self.algorithm in {"random", "demo", "experimental"}:
            move = random.choice(valid_moves)
            self.last_status = "Random AI selected a move."
            return move

        if self.algorithm == "minimax":
            self.last_status = "Minimax AI is not integrated yet."
            return None

        if self.algorithm == "mcts":
            if self.rollout_policy is None:
                self.last_status = f"Rollout policy '{self.rollout_policy_name}' is not available."
                return None
            try:
                agent = MCTSAgent(
                    iterations=self.iterations,
                    uct_c=self.uct_c,
                    rollout_policy=self.rollout_policy,
                )
            except ValueError as exc:
                self.last_status = f"Invalid MCTS settings: {exc}"
                return None
            self.last_agent = agent
            move = agent.get_move(game)
            self.last_status = self._mcts_status(agent, move)
            return move

        self.last_status = f"AI algorithm '{self.algorithm}' is not available."
        return None

    def _mcts_status(self, agent: MCTSAgent, move: Optional[Move]) -> str:
        stats = agent.last_stats
        summary = (
            f"MCTS(c={agent.uct_c}, it={self.iterations}, rollout={self.rollout_policy_name}) chose {move}"
        )
        chosen = next((row for row in agent.root_stats() if row["move"] == move), None)
        if chosen is not None:
            summary += f": N={chosen['N']}, W/N={chosen['WN']:.3f}"
        return f"{summary}, time={stats.get('time_s', 0.0):.2f}s."
