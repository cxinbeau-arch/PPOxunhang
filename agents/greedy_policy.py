"""Greedy local heuristic baseline."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from envs.patrol_env import ACTION_DELTAS, ACTION_NAMES, PatrolEnv


Position = Tuple[int, int]


class GreedyHeuristicPolicy:
    """Local greedy policy with no global map search."""

    policy_name = "greedy"

    def __init__(self, low_battery_margin: int = 3):
        self.low_battery_margin = low_battery_margin

    def select_action(self, env: PatrolEnv) -> int:
        goal = self._choose_goal(env)
        action_scores = []
        for action, (dx, dy) in ACTION_DELTAS.items():
            candidate = (env.agent_pos[0] + dx, env.agent_pos[1] + dy)
            is_blocked = action != 4 and (not env._is_inside(candidate) or candidate in env.obstacles)
            if is_blocked:
                continue
            distance = self._manhattan(candidate, goal)
            stay_penalty = 0.5 if action == 4 else 0.0
            action_scores.append((distance + stay_penalty, action))
        if not action_scores:
            return 4
        action_scores.sort(key=lambda item: (item[0], item[1]))
        return int(action_scores[0][1])

    def action_probabilities(self, action: int) -> List[float]:
        probs = [0.0 for _ in ACTION_NAMES]
        probs[int(action)] = 1.0
        return probs

    def _choose_goal(self, env: PatrolEnv) -> Position:
        unvisited = [pos for pos, visited in zip(env.targets, env.visited_targets) if not visited]
        distance_home = self._manhattan(env.agent_pos, env.base)
        if not unvisited:
            return env.base
        if env.battery <= distance_home + self.low_battery_margin:
            return env.base
        return min(unvisited, key=lambda pos: self._manhattan(env.agent_pos, pos))

    @staticmethod
    def _manhattan(a: Sequence[int], b: Sequence[int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
