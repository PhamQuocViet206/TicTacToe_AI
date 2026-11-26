"""AI integration layer for the GUI."""

from __future__ import annotations

import random
from typing import Optional, Tuple

try:
    from .game import TicTacToeGame
    from .mcts import MCTSAgent
    from .minimax import MinimaxAI
    from .rollout import ROLLOUT_POLICIES, RolloutPolicy
except ImportError:  # pragma: no cover
    from game import TicTacToeGame
    from mcts import MCTSAgent
    from minimax import MinimaxAI
    from rollout import ROLLOUT_POLICIES, RolloutPolicy

Move = Tuple[int, int]


class AIController:
    """Bridge between the GUI and concrete AI algorithms."""

    def __init__(
        self,
        algorithm: str = "random",
        iterations: int = 1000,
        uct_c: float = 1.4,
        rollout_policy: str = "random",
        depth: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        time_limit: Optional[float] = None,
        alpha_beta: bool = True,
        use_alpha_beta: Optional[bool] = None,
    ):
        self.algorithm = algorithm.lower()
        self.iterations = max(1, int(iterations))
        self.uct_c = float(uct_c)
        self.rollout_policy_name = rollout_policy.lower()
        self.rollout_policy: Optional[RolloutPolicy] = ROLLOUT_POLICIES.get(
            self.rollout_policy_name
        )
        self.depth = max_depth if max_depth is not None else depth
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        if use_alpha_beta is not None:
            alpha_beta = use_alpha_beta
        self.alpha_beta = alpha_beta
        self.last_status = "AI ready"
        self.last_agent: Optional[object] = None

    def get_move(self, game: TicTacToeGame) -> Optional[Move]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            self.last_status = "No legal moves available."
            return None

        if self.algorithm in {"random", "demo", "experimental"}:
            move = random.choice(valid_moves)
            self.last_status = "Random AI selected a move."
            return move

        if self.algorithm == "minimax":
            try:
                agent = MinimaxAI(
                    depth=self.depth,
                    max_nodes=self.max_nodes,
                    time_limit=self.time_limit,
                    alpha_beta=self.alpha_beta,
                )
            except (TypeError, ValueError) as exc:
                self.last_status = f"Invalid Minimax settings: {exc}"
                return None
            self.last_agent = agent
            move = agent.get_move(game)
            if move is None:
                self.last_status = "Minimax found no legal move."
            else:
                stats = agent.last_stats
                status = f"Minimax chose {move}: nodes={agent.nodes_searched}"
                if agent._search_aborted:
                    status += ", search limit reached"
                status += f", time={stats['time_s']:.3f}s."
                self.last_status = status
            return move

        if self.algorithm == "mcts":
            if self.rollout_policy is None:
                self.last_status = (
                    f"Rollout policy '{self.rollout_policy_name}' "
                    "is not available."
                )
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
            stats = agent.last_stats
            summary = (
                f"MCTS(c={agent.uct_c}, it={self.iterations}, "
                f"rollout={self.rollout_policy_name}) chose {move}"
            )
            chosen = next(
                (row for row in agent.root_stats() if row["move"] == move),
                None,
            )
            if chosen is not None:
                summary += f": N={chosen['N']}, W/N={chosen['WN']:.3f}"
            self.last_status = (
                f"{summary}, time={stats.get('time_s', 0.0):.2f}s."
            )
            return move

        self.last_status = (
            f"AI algorithm '{self.algorithm}' is not available."
        )
        return None
