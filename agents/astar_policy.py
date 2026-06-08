"""A* oracle policy."""

from __future__ import annotations

import time
from itertools import permutations
from typing import List, Optional, Sequence, Tuple

from envs.patrol_env import ACTION_DELTAS, ACTION_NAMES, PatrolEnv, astar_path, shortest_path_length


Position = Tuple[int, int]


class AStarOraclePolicy:
    """Upper-bound planner using full map knowledge and A* paths."""

    policy_name = "astar_oracle"

    def __init__(self, low_battery_margin: int = 3):
        self.low_battery_margin = low_battery_margin
        self.last_planning_time_ms = 0.0

    def select_action(self, env: PatrolEnv) -> int:
        started = time.perf_counter()
        goal = self._choose_goal(env)
        path = astar_path(env.agent_pos, goal, env.obstacles, env.width, env.height)
        self.last_planning_time_ms = (time.perf_counter() - started) * 1000.0
        if path is None or len(path) < 2:
            return 4
        next_pos = path[1]
        return self._action_to_neighbor(env.agent_pos, next_pos)

    def action_probabilities(self, action: int) -> List[float]:
        probs = [0.0 for _ in ACTION_NAMES]
        probs[int(action)] = 1.0
        return probs

    def _choose_goal(self, env: PatrolEnv) -> Position:
        unvisited = [pos for pos, visited in zip(env.targets, env.visited_targets) if not visited]
        if not unvisited:
            return env.base

        distance_home = shortest_path_length(env.agent_pos, env.base, env.obstacles, env.width, env.height)
        if distance_home is not None and env.battery <= distance_home + self.low_battery_margin:
            return env.base

        best = self._best_remaining_route(env, unvisited)
        if best is None:
            return env.base
        route_cost, first_target = best
        if route_cost + self.low_battery_margin > env.battery:
            return env.base
        return first_target

    def _best_remaining_route(self, env: PatrolEnv, unvisited: List[Position]) -> Optional[Tuple[int, Position]]:
        best: Optional[Tuple[int, Position]] = None
        for order in permutations(unvisited):
            total = 0
            current = env.agent_pos
            feasible = True
            for goal in list(order) + [env.base]:
                length = shortest_path_length(current, goal, env.obstacles, env.width, env.height)
                if length is None:
                    feasible = False
                    break
                total += length
                current = goal
            if feasible and (best is None or total < best[0]):
                best = (total, order[0])
        return best

    @staticmethod
    def _action_to_neighbor(current: Sequence[int], neighbor: Sequence[int]) -> int:
        dx = int(neighbor[0]) - int(current[0])
        dy = int(neighbor[1]) - int(current[1])
        for action, delta in ACTION_DELTAS.items():
            if delta == (dx, dy):
                return action
        return 4
