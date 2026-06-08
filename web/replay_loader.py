"""Load and normalize NavAgent replay JSON files for visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ACTION_NAMES = ["up", "down", "left", "right", "stay"]


def load_replay(path: str | Path) -> Dict[str, Any]:
    replay_path = Path(path)
    with replay_path.open("r", encoding="utf-8") as fh:
        raw_replay = json.load(fh)
    normalized = normalize_replay(raw_replay)
    normalized["source_path"] = str(replay_path)
    normalized["episode_type"] = _infer_episode_type(replay_path, normalized)
    return normalized


def normalize_replay(raw_replay: Dict[str, Any]) -> Dict[str, Any]:
    frames = raw_replay.get("frames")
    if frames is None:
        frames = raw_replay.get("steps") or []
    if not isinstance(frames, list):
        frames = []

    summary = _normalize_summary(raw_replay.get("summary") or raw_replay)
    steps: List[Dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        steps.append(_normalize_frame(frame, index, index == len(frames) - 1))

    return {
        "run_name": raw_replay.get("run_name") or summary.get("run_name") or "unknown",
        "episode_type": _episode_type_from_summary(summary),
        "policy": raw_replay.get("policy", "unknown"),
        "use_planner": bool(raw_replay.get("use_planner", False)),
        "steps": steps,
        "summary": summary,
        "config": raw_replay.get("config") or {},
        "memory": raw_replay.get("memory") or [],
        "has_action_probs": any(step.get("action_probs") for step in steps),
    }


def _normalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    total_targets = _as_int(summary.get("total_targets"), default=0)
    completed_targets = _as_int(summary.get("completed_targets"), default=0)
    success = bool(summary.get("success", completed_targets > 0 and completed_targets == total_targets and total_targets > 0))
    return {
        "success": success,
        "completed_targets": completed_targets,
        "total_targets": total_targets,
        "returned_to_base": bool(summary.get("returned_to_base", False)),
        "episode_reward": _as_float(summary.get("episode_reward", summary.get("reward")), default=0.0),
        "episode_length": _as_int(summary.get("episode_length", summary.get("path_length")), default=0),
        "path_length": _as_int(summary.get("path_length", summary.get("episode_length")), default=0),
        "reason_done": summary.get("reason_done") or summary.get("done_reason") or summary.get("termination_reason") or "unknown",
        "done_reason": summary.get("done_reason") or summary.get("reason_done") or summary.get("termination_reason") or "unknown",
        "remaining_battery": _as_float(summary.get("remaining_battery"), default=0.0),
        "collision_count": _as_int(summary.get("collision_count"), default=0),
        "repeated_cell_count": _as_int(summary.get("repeated_cell_count"), default=0),
        "path_efficiency": _as_float(summary.get("path_efficiency"), default=0.0),
    }


def _normalize_frame(frame: Dict[str, Any], fallback_step: int, is_last: bool) -> Dict[str, Any]:
    targets, visited_targets = _normalize_targets(frame.get("targets", []))
    action = frame.get("action_name")
    if action is None:
        action = _action_name(frame.get("action"))
    done_reason = frame.get("done_reason") or frame.get("reason_done") or "running"
    done = bool(frame.get("done", False) or (is_last and done_reason != "running"))
    info = {
        key: value
        for key, value in frame.items()
        if key
        not in {
            "step",
            "agent_pos",
            "agent_position",
            "base_pos",
            "base_position",
            "targets",
            "obstacles",
            "battery",
            "action",
            "action_name",
            "reward",
            "cum_reward",
            "cumulative_reward",
            "done",
            "action_probs",
            "action_probabilities",
        }
    }
    return {
        "step": _as_int(frame.get("step"), default=fallback_step),
        "agent_pos": _as_pair(frame.get("agent_pos") or frame.get("agent_position") or frame.get("agent")),
        "base_pos": _as_pair(frame.get("base_pos") or frame.get("base_position") or frame.get("base")),
        "targets": targets,
        "visited_targets": visited_targets,
        "obstacles": _as_pair_list(frame.get("obstacles", [])),
        "grid_size": _as_pair(frame.get("grid_size"), default=[12, 12]),
        "battery": _as_float(frame.get("battery"), default=0.0),
        "max_battery": _as_float(frame.get("max_battery"), default=0.0),
        "action": action,
        "reward": _as_float(frame.get("reward"), default=0.0),
        "cum_reward": _as_float(frame.get("cum_reward", frame.get("cumulative_reward")), default=0.0),
        "done": done,
        "info": info,
        "action_probs": _normalize_action_probs(frame.get("action_probs", frame.get("action_probabilities"))),
    }


def _normalize_targets(raw_targets: Any) -> Tuple[List[List[int]], List[List[int]]]:
    targets: List[List[int]] = []
    visited: List[List[int]] = []
    if not isinstance(raw_targets, list):
        return targets, visited
    for item in raw_targets:
        if isinstance(item, dict):
            position = _as_pair(item.get("position") or item.get("pos") or item.get("target"))
            if position is None:
                continue
            targets.append(position)
            if bool(item.get("visited", False)):
                visited.append(position)
        else:
            position = _as_pair(item)
            if position is not None:
                targets.append(position)
    return targets, visited


def _normalize_action_probs(raw_probs: Any) -> Optional[Dict[str, float]]:
    if raw_probs is None:
        return None
    if isinstance(raw_probs, dict):
        result = {}
        for name in ACTION_NAMES:
            if name in raw_probs:
                result[name] = _as_float(raw_probs[name], default=0.0)
        for key, value in raw_probs.items():
            if key not in result:
                result[str(key)] = _as_float(value, default=0.0)
        return result if result else None
    if isinstance(raw_probs, Sequence) and not isinstance(raw_probs, (str, bytes)):
        values = list(raw_probs)
        if not values:
            return None
        return {name: _as_float(values[index], default=0.0) for index, name in enumerate(ACTION_NAMES) if index < len(values)}
    return None


def _infer_episode_type(path: Path, normalized: Dict[str, Any]) -> str:
    name = path.stem.lower()
    if "success" in name:
        return "success"
    if "failure" in name or "fail" in name:
        return "failure"
    if "worst" in name:
        return "worst"
    if "partial" in name:
        return "partial"
    return _episode_type_from_summary(normalized.get("summary", {}))


def _episode_type_from_summary(summary: Dict[str, Any]) -> str:
    if summary.get("success"):
        return "success"
    completed = _as_int(summary.get("completed_targets"), default=0)
    if completed > 0:
        return "partial"
    reason = str(summary.get("reason_done") or summary.get("done_reason") or "").lower()
    if "worst" in reason:
        return "worst"
    return "failure"


def _action_name(action: Any) -> Optional[str]:
    if action is None:
        return None
    if isinstance(action, str):
        return action
    try:
        index = int(action)
    except (TypeError, ValueError):
        return str(action)
    if 0 <= index < len(ACTION_NAMES):
        return ACTION_NAMES[index]
    return str(action)


def _as_pair(value: Any, default: Optional[List[int]] = None) -> Optional[List[int]]:
    if value is None:
        return default
    if isinstance(value, dict):
        value = value.get("position") or value.get("pos") or value.get("xy")
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        items = list(value)
        if len(items) >= 2:
            return [_as_int(items[0]), _as_int(items[1])]
    return default


def _as_pair_list(values: Any) -> List[List[int]]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        pair = _as_pair(value)
        if pair is not None:
            result.append(pair)
    return result


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
