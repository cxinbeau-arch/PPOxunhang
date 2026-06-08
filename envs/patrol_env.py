"""The environment follows the Gymnasium API when Gymnasium is installed. """

from __future__ import annotations

import copy
import ast
import json
import math
import random
import sys
from heapq import heappop, heappush
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - fallback is for pre-install smoke tests.
    yaml = None

try:  # pragma: no cover - exercised only when Gymnasium is installed.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Lightweight fallback for local smoke tests.
    class _FallbackEnv:
        metadata: Dict[str, Any] = {}

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return random.randrange(self.n)

    class _Box:
        def __init__(self, low: float, high: float, shape: Sequence[int], dtype: Any):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    gym = SimpleNamespace(Env=_FallbackEnv)
    spaces = SimpleNamespace(Discrete=_Discrete, Box=_Box)


Position = Tuple[int, int]


ACTION_DELTAS: Dict[int, Position] = {
    0: (0, -1),  # up
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
    4: (0, 0),   # stay
}
ACTION_NAMES = ["up", "down", "left", "right", "stay"]


DEFAULT_REWARD = {
    "task_bonus": 10.0,
    "return_bonus": 20.0,
    "collision_penalty": -1.0,
    "timeout_penalty": -5.0,
    "battery_out_penalty": -10.0,
    "step_penalty": -0.02,
    "target_distance_coef": 0.0,
    "base_distance_coef_after_all_targets": 0.0,
    "base_distance_coef_low_battery": 0.0,
    "revisit_penalty": -0.02,
    "llm_subgoal_coef": 0.0,
}


LEVEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "level": "basic",
        "grid_size": [8, 8],
        "base": [0, 0],
        "start": [0, 0],
        "targets": [[6, 6]],
        "obstacles": [],
        "max_steps": 80,
        "max_battery": 100,
        "reward_mode": "sparse",
        "target_selection": "ordered",
        "battery_cost_per_step": 1.0,
        "low_battery_ratio": 0.30,
    },
    "obstacle": {
        "level": "obstacle",
        "grid_size": [10, 10],
        "base": [0, 0],
        "start": [0, 0],
        "targets": [[8, 8], [1, 8]],
        "obstacles": [[3, 0], [3, 1], [3, 2], [3, 3], [3, 5], [3, 6], [3, 7], [6, 3], [7, 3], [8, 3]],
        "max_steps": 140,
        "max_battery": 160,
        "reward_mode": "dense",
        "target_selection": "nearest",
        "battery_cost_per_step": 1.0,
        "low_battery_ratio": 0.30,
    },
    "multitask": {
        "level": "multitask",
        "grid_size": [12, 12],
        "base": [0, 0],
        "start": [0, 0],
        "targets": [[10, 10], [2, 9], [9, 2]],
        "obstacles": [[4, 0], [4, 1], [4, 2], [4, 4], [4, 5], [4, 6], [4, 7], [7, 4], [8, 4], [9, 4], [10, 4], [1, 6], [2, 6], [3, 6]],
        "max_steps": 190,
        "max_battery": 180,
        "reward_mode": "dense",
        "target_selection": "nearest",
        "battery_cost_per_step": 1.0,
        "low_battery_ratio": 0.35,
    },
    "randomized": {
        "level": "randomized",
        "grid_size": [12, 12],
        "base": [0, 0],
        "start": [0, 0],
        "targets": [[10, 10], [2, 9], [9, 2]],
        "obstacles": [],
        "max_steps": 180,
        "max_battery": 120,
        "reward_mode": "dense",
        "target_selection": "nearest",
        "battery_cost_per_step": 1.0,
        "low_battery_ratio": 0.35,
        "randomized_each_episode": True,
        "random_start": True,
        "random_targets": True,
        "num_targets_range": [2, 4],
        "random_obstacles": True,
        "obstacle_density_range": [0.10, 0.22],
        "ensure_solvable": True,
        "max_generation_retry": 100,
        "battery_mode": "shortest_path_budget",
        "budget_factor_range": [1.15, 1.45],
        "min_battery": 35,
    },
}


def _as_position(value: Sequence[int]) -> Position:
    if len(value) != 2:
        raise ValueError(f"Expected a 2D position, got {value!r}")
    return int(value[0]), int(value[1])


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        text = fh.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _simple_yaml_load(text)


def load_env_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and normalize an environment YAML config."""

    raw = _normalize_nested_config(read_yaml_file(path))

    level = raw.get("level", "basic")
    base = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS["basic"])
    cfg = _deep_merge(base, raw)
    cfg["reward"] = _deep_merge(DEFAULT_REWARD, cfg.get("reward", {}))
    return cfg


def _normalize_nested_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both flat configs and requirement-style nested configs."""

    cfg = copy.deepcopy(raw)
    map_cfg = cfg.pop("map", {}) or {}
    if map_cfg:
        if "size" in map_cfg:
            size = int(map_cfg["size"])
            cfg["grid_size"] = [size, size]
        for key in [
            "random_start",
            "random_targets",
            "num_targets_range",
            "random_obstacles",
            "obstacle_density_range",
            "ensure_solvable",
            "max_generation_retry",
        ]:
            if key in map_cfg:
                cfg[key] = map_cfg[key]

    battery_cfg = cfg.pop("battery", {}) or {}
    if battery_cfg:
        if "mode" in battery_cfg:
            cfg["battery_mode"] = battery_cfg["mode"]
        if "budget_factor_range" in battery_cfg:
            cfg["budget_factor_range"] = battery_cfg["budget_factor_range"]
        if "min_battery" in battery_cfg:
            cfg["min_battery"] = battery_cfg["min_battery"]

    episode_cfg = cfg.pop("episode", {}) or {}
    if "max_steps" in episode_cfg:
        cfg["max_steps"] = episode_cfg["max_steps"]

    reset_cfg = cfg.pop("reset", {}) or {}
    if reset_cfg:
        if "randomized_each_episode" in reset_cfg:
            cfg["randomized_each_episode"] = reset_cfg["randomized_each_episode"]
        if "seed" in reset_cfg:
            cfg["reset_seed"] = reset_cfg["seed"]

    return cfg


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by this project configs."""

    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            current_key = line[:-1]
            result[current_key] = {}
            continue
        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            result[current_key] = _parse_scalar(value.strip())
            continue
        if current_key is None:
            continue
        if indent >= 2 and line.startswith("- "):
            if not isinstance(result.get(current_key), list):
                result[current_key] = []
            result[current_key].append(_parse_scalar(line[2:].strip()))
            continue
        if indent >= 2 and ":" in line:
            if not isinstance(result.get(current_key), dict):
                result[current_key] = {}
            key, value = line.split(":", 1)
            result[current_key][key.strip()] = _parse_scalar(value.strip())
    return result


def _parse_scalar(value: str) -> Any:
    if value == "":
        return {}
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def make_env(config: Union[str, Path, Dict[str, Any]]) -> "PatrolEnv":
    """Create a PatrolEnv from a YAML path or a config dictionary."""

    if isinstance(config, (str, Path)):
        config = load_env_config(config)
    return PatrolEnv(config)


def astar_path(
    start: Position,
    goal: Position,
    obstacles: Iterable[Position],
    width: int,
    height: int,
) -> Optional[List[Position]]:
    """Return an A* path including start and goal, or None if unreachable."""

    obstacle_set = set(obstacles)
    if start == goal:
        return [start]
    if start in obstacle_set or goal in obstacle_set:
        return None

    frontier: List[Tuple[int, int, Position]] = []
    heappush(frontier, (_manhattan(start, goal), 0, start))
    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost_so_far: Dict[Position, int] = {start: 0}

    while frontier:
        _, current_cost, current = heappop(frontier)
        if current == goal:
            break
        if current_cost > cost_so_far[current]:
            continue
        for neighbor in _grid_neighbors(current, width, height, obstacle_set):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _manhattan(neighbor, goal)
                heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return None
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def shortest_path_length(
    start: Position,
    goal: Position,
    obstacles: Iterable[Position],
    width: int,
    height: int,
) -> Optional[int]:
    path = astar_path(start, goal, obstacles, width, height)
    return None if path is None else max(0, len(path) - 1)


def shortest_patrol_route_length(
    start: Position,
    targets: Sequence[Position],
    base: Position,
    obstacles: Iterable[Position],
    width: int,
    height: int,
) -> Optional[int]:
    """Exact shortest target order for small target counts, then return home."""

    if not targets:
        return shortest_path_length(start, base, obstacles, width, height)
    best: Optional[int] = None
    for order in permutations(targets):
        total = 0
        current = start
        feasible = True
        for goal in list(order) + [base]:
            length = shortest_path_length(current, goal, obstacles, width, height)
            if length is None:
                feasible = False
                break
            total += length
            current = goal
        if feasible and (best is None or total < best):
            best = total
    return best


def _grid_neighbors(pos: Position, width: int, height: int, obstacles: set) -> List[Position]:
    neighbors = []
    for action, (dx, dy) in ACTION_DELTAS.items():
        if action == 4:
            continue
        nxt = (pos[0] + dx, pos[1] + dy)
        if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in obstacles:
            neighbors.append(nxt)
    return neighbors


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PatrolEnv(gym.Env):
    """Single-agent grid patrol environment.

    Coordinates are represented as ``[x, y]``. The agent starts at ``base`` and
    must visit all targets, avoid obstacles, manage battery, and return home.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: Optional[Dict[str, Any]] = None, **overrides: Any):
        raw_config = _deep_merge(LEVEL_CONFIGS["basic"], config or {})
        raw_config = _deep_merge(raw_config, overrides)
        raw_config["reward"] = _deep_merge(DEFAULT_REWARD, raw_config.get("reward", {}))
        self.config = raw_config

        size = self.config.get("grid_size", [8, 8])
        self.width, self.height = int(size[0]), int(size[1])
        self.base: Position = _as_position(self.config.get("base", [0, 0]))
        self.start: Position = _as_position(self.config.get("start", self.base))
        self.targets: List[Position] = [_as_position(t) for t in self.config.get("targets", [])]
        self.obstacles = {_as_position(o) for o in self.config.get("obstacles", [])}
        self.max_steps = int(self.config.get("max_steps", 80))
        self.max_battery = float(self.config.get("max_battery", self.max_steps))
        self.battery_cost_per_step = float(self.config.get("battery_cost_per_step", 1.0))
        self.low_battery_ratio = float(self.config.get("low_battery_ratio", 0.30))
        self.reward_cfg = self.config["reward"]
        self.reward_mode = str(self.config.get("reward_mode", "sparse"))
        self.target_selection = str(self.config.get("target_selection", "ordered"))
        self.use_planner_subgoal = bool(self.config.get("use_planner_subgoal", False))
        self.randomized_each_episode = bool(self.config.get("randomized_each_episode", False))
        self.random_start = bool(self.config.get("random_start", False))
        self.random_targets = bool(self.config.get("random_targets", False))
        self.random_obstacles = bool(self.config.get("random_obstacles", False))
        self.num_targets_range = [int(x) for x in self.config.get("num_targets_range", [len(self.targets), len(self.targets)])]
        self.obstacle_density_range = [float(x) for x in self.config.get("obstacle_density_range", [0.0, 0.0])]
        self.ensure_solvable = bool(self.config.get("ensure_solvable", False))
        self.max_generation_retry = int(self.config.get("max_generation_retry", 100))
        self.battery_mode = str(self.config.get("battery_mode", "fixed"))
        self.budget_factor_range = [float(x) for x in self.config.get("budget_factor_range", [1.0, 1.0])]
        self.min_battery = float(self.config.get("min_battery", 0.0))

        if not self.randomized_each_episode:
            self._validate_static_layout()
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

        self.agent_pos: Position = self.start
        self.visited_targets: List[bool] = [False for _ in self.targets]
        self.battery = self.max_battery
        self.step_count = 0
        self.collision_count = 0
        self.revisit_count = 0
        self.position_history: List[Position] = [self.start]
        self.shortest_possible_path_length = shortest_patrol_route_length(
            self.start, self.targets, self.base, self.obstacles, self.width, self.height
        )
        self.done_reason = "running"
        self.llm_subgoal: Optional[Position] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            super().reset()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.randomized_each_episode:
            self._generate_random_layout(seed)

        self.agent_pos = self.start
        self.visited_targets = [False for _ in self.targets]
        self.battery = self.max_battery
        self.step_count = 0
        self.collision_count = 0
        self.revisit_count = 0
        self.position_history = [self.agent_pos]
        self.done_reason = "running"
        self.llm_subgoal = None
        if options and options.get("llm_subgoal") is not None:
            self.set_llm_subgoal(options["llm_subgoal"])

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}; expected 0-4.")

        previous_target = self._current_target()
        previous_target_dist = self._manhattan(self.agent_pos, previous_target)
        previous_base_dist = self._manhattan(self.agent_pos, self.base)
        previous_llm_dist = self._manhattan(self.agent_pos, self._llm_goal())

        reward = float(self.reward_cfg["step_penalty"])
        self.step_count += 1

        dx, dy = ACTION_DELTAS[action]
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        collision = False
        if action != 4 and not self._is_free(next_pos):
            collision = True
            self.collision_count += 1
            reward += float(self.reward_cfg["collision_penalty"])
            next_pos = self.agent_pos
        self.agent_pos = next_pos
        self.position_history.append(self.agent_pos)
        self.battery -= self.battery_cost_per_step

        if self.reward_mode in {"dense", "llm"}:
            new_target_dist = self._manhattan(self.agent_pos, previous_target)
            reward += float(self.reward_cfg["target_distance_coef"]) * (previous_target_dist - new_target_dist)
            if self._all_targets_visited():
                new_base_dist = self._manhattan(self.agent_pos, self.base)
                reward += float(self.reward_cfg["base_distance_coef_after_all_targets"]) * (previous_base_dist - new_base_dist)
            elif self._is_low_battery():
                new_base_dist = self._manhattan(self.agent_pos, self.base)
                reward += float(self.reward_cfg["base_distance_coef_low_battery"]) * (previous_base_dist - new_base_dist)

        if self.reward_mode == "llm" or self.use_planner_subgoal:
            new_llm_dist = self._manhattan(self.agent_pos, self._llm_goal())
            reward += float(self.reward_cfg["llm_subgoal_coef"]) * (previous_llm_dist - new_llm_dist)

        reward += self._apply_target_reward()

        success = self._all_targets_visited() and self.agent_pos == self.base
        terminated = False
        truncated = False
        if success:
            reward += float(self.reward_cfg["return_bonus"])
            terminated = True
            self.done_reason = "success"
        elif self.battery <= 0:
            reward += float(self.reward_cfg["battery_out_penalty"])
            terminated = True
            self.done_reason = "battery_out"
        elif self.step_count >= self.max_steps:
            reward += float(self.reward_cfg["timeout_penalty"])
            truncated = True
            self.done_reason = "timeout"

        info = self._get_info()
        info["collision"] = collision
        info["success"] = success
        info["is_success"] = success
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode: str = "ansi") -> str:
        if mode != "ansi":
            raise NotImplementedError("Only ansi rendering is implemented.")
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.obstacles:
            grid[y][x] = "#"
        for idx, (x, y) in enumerate(self.targets):
            grid[y][x] = "v" if self.visited_targets[idx] else "T"
        bx, by = self.base
        grid[by][bx] = "B"
        ax, ay = self.agent_pos
        grid[ay][ax] = "A"
        rows = [" ".join(row) for row in grid]
        return "\n".join(rows)

    def set_llm_subgoal(self, subgoal: Optional[Sequence[int]]) -> None:
        self.llm_subgoal = None if subgoal is None else _as_position(subgoal)

    def get_state(self) -> Dict[str, Any]:
        unvisited = [list(pos) for pos, visited in zip(self.targets, self.visited_targets) if not visited]
        visited = [list(pos) for pos, visited in zip(self.targets, self.visited_targets) if visited]
        return {
            "agent_position": list(self.agent_pos),
            "base_position": list(self.base),
            "battery": float(self.battery),
            "max_battery": float(self.max_battery),
            "unvisited_targets": unvisited,
            "visited_targets": visited,
            "distance_to_base": self._manhattan(self.agent_pos, self.base),
            "current_target": list(self._current_target()),
            "llm_subgoal": list(self._llm_goal()),
            "step": self.step_count,
            "max_steps": self.max_steps,
            "collision_count": self.collision_count,
            "shortest_possible_path_length": self.shortest_possible_path_length,
            "history_memory": "",
        }

    def render_data(self) -> Dict[str, Any]:
        return {
            "grid_size": [self.width, self.height],
            "agent_position": list(self.agent_pos),
            "base_position": list(self.base),
            "targets": [
                {"position": list(pos), "visited": bool(visited)}
                for pos, visited in zip(self.targets, self.visited_targets)
            ],
            "obstacles": [list(pos) for pos in sorted(self.obstacles)],
            "battery": float(self.battery),
            "max_battery": float(self.max_battery),
            "step": int(self.step_count),
            "max_steps": int(self.max_steps),
            "collision_count": int(self.collision_count),
            "revisit_count": int(self.revisit_count),
            "repeated_cell_count": int(self._repeated_cell_count()),
            "shortest_possible_path_length": self.shortest_possible_path_length,
            "done_reason": self.done_reason,
            "llm_subgoal": list(self._llm_goal()),
            "current_target": list(self._current_target()),
        }

    def _get_obs(self) -> np.ndarray:
        max_distance = self._max_distance()
        current = self._current_target()
        llm_goal = self._llm_goal()
        values = [
            self._norm_x(self.agent_pos[0]),
            self._norm_y(self.agent_pos[1]),
            self._norm_x(self.base[0]),
            self._norm_y(self.base[1]),
            self._norm_x(current[0]),
            self._norm_y(current[1]),
            max(0.0, min(1.0, self.battery / max(1.0, self.max_battery))),
            self._visited_ratio(),
            self._manhattan(self.agent_pos, current) / max_distance,
            self._manhattan(self.agent_pos, self.base) / max_distance,
            self._nearest_distance((0, -1)),
            self._nearest_distance((0, 1)),
            self._nearest_distance((-1, 0)),
            self._nearest_distance((1, 0)),
            self._norm_x(llm_goal[0]),
            self._norm_y(llm_goal[1]),
        ]
        return np.array(values, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        completed = sum(self.visited_targets)
        returned_to_base = bool(self.agent_pos == self.base)
        success = bool(completed == len(self.targets) and returned_to_base)
        return {
            "completed_targets": int(completed),
            "total_targets": int(len(self.targets)),
            "visited_targets": [bool(v) for v in self.visited_targets],
            "visited_ratio": self._visited_ratio(),
            "success": success,
            "is_success": success,
            "returned_to_base": returned_to_base,
            "agent_at_base": returned_to_base,
            "agent_final_position": list(self.agent_pos),
            "base_position": list(self.base),
            "remaining_battery": float(max(0.0, self.battery)),
            "battery": float(self.battery),
            "collision_count": int(self.collision_count),
            "revisit_count": int(self.revisit_count),
            "repeated_cell_count": int(self._repeated_cell_count()),
            "path_length": int(self.step_count),
            "shortest_possible_path_length": self.shortest_possible_path_length,
            "done_reason": self.done_reason,
        }

    def _apply_target_reward(self) -> float:
        reward = 0.0
        for idx, target in enumerate(self.targets):
            if self.agent_pos != target:
                continue
            if not self.visited_targets[idx]:
                self.visited_targets[idx] = True
                reward += float(self.reward_cfg["task_bonus"])
            else:
                self.revisit_count += 1
                reward += float(self.reward_cfg["revisit_penalty"])
        return reward

    def _current_target(self) -> Position:
        unvisited = [pos for pos, visited in zip(self.targets, self.visited_targets) if not visited]
        if not unvisited:
            return self.base
        if self.use_planner_subgoal and self.llm_subgoal is not None:
            return self.llm_subgoal
        if self.target_selection == "nearest":
            return min(unvisited, key=lambda pos: self._manhattan(self.agent_pos, pos))
        return unvisited[0]

    def _llm_goal(self) -> Position:
        if self.llm_subgoal is not None:
            return self.llm_subgoal
        return self._current_target()

    def _all_targets_visited(self) -> bool:
        return all(self.visited_targets) if self.targets else True

    def _is_low_battery(self) -> bool:
        return self.battery <= self.max_battery * self.low_battery_ratio

    def _is_inside(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_free(self, pos: Position) -> bool:
        return self._is_inside(pos) and pos not in self.obstacles

    def _nearest_distance(self, delta: Position) -> float:
        x, y = self.agent_pos
        steps = 0
        while True:
            x += delta[0]
            y += delta[1]
            if not self._is_inside((x, y)):
                return steps / max(1, max(self.width, self.height) - 1)
            steps += 1
            if (x, y) in self.obstacles:
                return steps / max(1, max(self.width, self.height) - 1)

    def _norm_x(self, x: int) -> float:
        return float(x) / max(1, self.width - 1)

    def _norm_y(self, y: int) -> float:
        return float(y) / max(1, self.height - 1)

    def _visited_ratio(self) -> float:
        if not self.targets:
            return 1.0
        return sum(self.visited_targets) / len(self.targets)

    def _max_distance(self) -> float:
        return float(max(1, self.width + self.height - 2))

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _repeated_cell_count(self) -> int:
        return max(0, len(self.position_history) - len(set(self.position_history)))

    def _generate_random_layout(self, seed: Optional[int]) -> None:
        rng = random.Random(seed)
        size = self.config.get("grid_size", [12, 12])
        self.width, self.height = int(size[0]), int(size[1])
        self.base = _as_position(self.config.get("base", [0, 0]))

        all_cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        for attempt in range(max(1, self.max_generation_retry)):
            reserved = {self.base}
            start = rng.choice([cell for cell in all_cells if cell not in reserved]) if self.random_start else _as_position(self.config.get("start", self.base))
            reserved.add(start)

            low, high = self.num_targets_range
            target_count = rng.randint(min(low, high), max(low, high)) if self.random_targets else len(self.config.get("targets", []))
            free_for_targets = [cell for cell in all_cells if cell not in reserved]
            if len(free_for_targets) < target_count:
                continue
            targets = rng.sample(free_for_targets, target_count) if self.random_targets else [_as_position(t) for t in self.config.get("targets", [])]
            reserved.update(targets)

            density_low, density_high = self.obstacle_density_range
            density = rng.uniform(min(density_low, density_high), max(density_low, density_high)) if self.random_obstacles else 0.0
            obstacle_budget = int(round(density * self.width * self.height))
            free_for_obstacles = [cell for cell in all_cells if cell not in reserved]
            obstacle_count = min(obstacle_budget, len(free_for_obstacles))
            obstacles = set(rng.sample(free_for_obstacles, obstacle_count)) if self.random_obstacles and obstacle_count > 0 else set()

            route_length = shortest_patrol_route_length(start, targets, self.base, obstacles, self.width, self.height)
            if self.ensure_solvable and route_length is None:
                continue
            if route_length is None:
                route_length = self.max_steps

            self.start = start
            self.targets = list(targets)
            self.obstacles = obstacles
            self.shortest_possible_path_length = int(route_length)
            if self.battery_mode == "shortest_path_budget":
                factor_low, factor_high = self.budget_factor_range
                factor = rng.uniform(min(factor_low, factor_high), max(factor_low, factor_high))
                self.max_battery = float(max(self.min_battery, math.ceil(route_length * factor)))
            else:
                self.max_battery = float(self.config.get("max_battery", self.max_steps))
            self.battery = self.max_battery
            self._validate_static_layout()
            return

        raise RuntimeError(
            "Failed to generate a solvable randomized PatrolEnv layout "
            f"after {self.max_generation_retry} attempts."
        )

    def _validate_static_layout(self) -> None:
        points = [self.base, self.start] + self.targets
        for pos in points:
            if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
                raise ValueError(f"Position {pos} is outside grid {self.width}x{self.height}.")
        if self.base in self.obstacles:
            raise ValueError("Base cannot be inside an obstacle.")
        if self.start in self.obstacles:
            raise ValueError("Start cannot be inside an obstacle.")
        blocked_targets = [target for target in self.targets if target in self.obstacles]
        if blocked_targets:
            raise ValueError(f"Targets cannot be obstacles: {blocked_targets}")


def _smoke_test(config_path: Optional[str] = None) -> None:
    config = load_env_config(config_path) if config_path else LEVEL_CONFIGS["basic"]
    env = PatrolEnv(config)
    obs, info = env.reset(seed=7)
    total_reward = 0.0
    done = False
    print("Initial observation shape:", obs.shape)
    print(env.render())
    while not done and env.step_count < 12:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(json.dumps({"steps": env.step_count, "reward": total_reward, "info": info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _smoke_test(sys.argv[1] if len(sys.argv) > 1 else None)
