"""Evaluate Random, Greedy, A* oracle, PPO, and PPO+Planner policies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.astar_policy import AStarOraclePolicy
from agents.greedy_policy import GreedyHeuristicPolicy
from agents.llm_planner import LLMPlanner
from agents.ppo_executor import PPOExecutor
from agents.random_policy import RandomPolicy
from envs.patrol_env import ACTION_NAMES, PatrolEnv, load_env_config


def evaluate_policy_run(
    config: Dict[str, Any],
    policy: str,
    episodes: int,
    run_name: str,
    model_path: Optional[str] = None,
    out_path: Optional[Path] = None,
    replay_dir: Optional[Path] = None,
    seed: int = 0,
    use_planner: bool = False,
    plan_interval: int = 5,
) -> Dict[str, Any]:
    """Run real episodes and return aggregate metrics."""

    ppo_executor = None
    baseline_policy = make_baseline_policy(policy)
    metric_policy_name = normalize_policy_name(policy)
    planner = LLMPlanner() if use_planner else None
    memory: List[Dict[str, Any]] = []

    if policy == "ppo":
        if not model_path:
            raise ValueError("--model_path is required when --policy ppo")
        ppo_executor = PPOExecutor(model_path)

    episode_rows = []
    first_success_replay = None
    first_failure_replay = None
    best_partial_replay = None
    best_partial_score = None

    for episode_idx in range(episodes):
        env = PatrolEnv(config)
        observation, info = env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0
        planner_output = None
        frames = []

        if planner is not None:
            planner_output = planner.plan(env.get_state(), memory)
            env.set_llm_subgoal(planner_output["next_subgoal"])
            observation = env._get_obs()

        frames.append(_make_frame(env, None, "start", 0.0, 0.0, None, planner_output))

        while not done:
            if planner is not None and env.step_count % max(1, plan_interval) == 0:
                planner_output = planner.plan(env.get_state(), memory)
                env.set_llm_subgoal(planner_output["next_subgoal"])
                observation = env._get_obs()

            planning_started = time.perf_counter()
            if baseline_policy is not None:
                action = baseline_policy.select_action(env)
                probabilities = baseline_policy.action_probabilities(action)
                planning_time_ms = getattr(baseline_policy, "last_planning_time_ms", (time.perf_counter() - planning_started) * 1000.0)
            elif policy == "ppo":
                assert ppo_executor is not None
                action = ppo_executor.predict(observation, deterministic=True)
                probabilities = ppo_executor.action_probabilities(observation)
                planning_time_ms = 0.0
            else:
                raise ValueError(f"Unknown policy {policy!r}; expected random, greedy, astar, or ppo.")

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = bool(terminated or truncated)
            frames.append(
                _make_frame(
                    env,
                    action,
                    ACTION_NAMES[action],
                    reward,
                    episode_reward,
                    probabilities,
                    planner_output,
                    planning_time_ms,
                )
            )

        row = {
            "episode_index": episode_idx,
            "success": bool(info.get("success", False)),
            "completed_targets": int(info.get("completed_targets", 0)),
            "total_targets": int(info.get("total_targets", 0)),
            "returned_to_base": bool(info.get("returned_to_base", False)),
            "episode_reward": float(episode_reward),
            "episode_length": int(env.step_count),
            "path_length": int(info.get("path_length", env.step_count)),
            "collision_count": int(info.get("collision_count", 0)),
            "repeated_cell_count": int(info.get("repeated_cell_count", 0)),
            "remaining_battery": float(info.get("remaining_battery", 0.0)),
            "done_reason": str(info.get("done_reason", "unknown")),
            "agent_final_position": info.get("agent_final_position"),
            "base_position": info.get("base_position"),
            "shortest_possible_path_length": info.get("shortest_possible_path_length"),
            "path_efficiency": _episode_path_efficiency(info, bool(info.get("success", False)), env.step_count),
            "normalized_reward": _normalized_reward(episode_reward, info, config),
            "mean_planning_time_ms": _mean_frame_planning_time(frames),
        }
        episode_rows.append(row)

        replay = {
            "run_name": run_name,
            "policy": metric_policy_name,
            "use_planner": use_planner,
            "episode_index": episode_idx,
            "summary": row,
            "config": _public_config(config),
            "memory": memory,
            "frames": frames,
        }
        if row["success"] and first_success_replay is None:
            first_success_replay = replay
        if not row["success"] and first_failure_replay is None:
            first_failure_replay = replay
        partial_score = (
            row["completed_targets"],
            row["returned_to_base"],
            row["remaining_battery"],
            row["episode_reward"],
            -row["episode_length"],
        )
        if best_partial_score is None or partial_score > best_partial_score:
            best_partial_score = partial_score
            best_partial_replay = replay

        if planner is not None:
            memory.append(
                {
                    "episode_index": episode_idx,
                    "success": row["success"],
                    "done_reason": row["done_reason"],
                    "completed_targets": row["completed_targets"],
                    "remaining_battery": row["remaining_battery"],
                }
            )

    metrics = _aggregate_metrics(run_name, metric_policy_name, episodes, episode_rows, use_planner)
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in ["success_episode.json", "failure_episode.json"]:
            stale_path = replay_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        replay_paths = {}
        if first_success_replay is not None:
            path = replay_dir / "success_episode.json"
            _write_json(path, first_success_replay)
            replay_paths["success_episode"] = str(path)
            replay_paths["success_episode_html"] = _try_export_replay_html(path)
        if first_failure_replay is not None:
            path = replay_dir / "failure_episode.json"
            _write_json(path, first_failure_replay)
            replay_paths["failure_episode"] = str(path)
            replay_paths["failure_episode_html"] = _try_export_replay_html(path)
        if first_success_replay is None and best_partial_replay is not None:
            path = replay_dir / "best_partial_episode.json"
            _write_json(path, best_partial_replay)
            replay_paths["best_partial_episode"] = str(path)
            replay_paths["best_partial_episode_html"] = _try_export_replay_html(path)
        metrics["replay_paths"] = replay_paths

    if out_path is not None:
        _write_json(out_path, metrics)
    return metrics


def _make_frame(
    env: PatrolEnv,
    action: Optional[int],
    action_name: str,
    reward: float,
    cumulative_reward: float,
    probabilities: Optional[List[float]],
    planner_output: Optional[Dict[str, Any]],
    planning_time_ms: float = 0.0,
) -> Dict[str, Any]:
    frame = env.render_data()
    frame.update(
        {
            "action": action,
            "action_name": action_name,
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "action_probabilities": probabilities,
            "planner_output": planner_output,
            "planning_time_ms": float(planning_time_ms),
        }
    )
    return frame


def _aggregate_metrics(
    run_name: str,
    policy: str,
    episodes: int,
    rows: List[Dict[str, Any]],
    use_planner: bool,
) -> Dict[str, Any]:
    success_count = sum(1 for row in rows if row["success"])
    returned_count = sum(1 for row in rows if row["returned_to_base"])
    battery_out_count = sum(1 for row in rows if row["done_reason"] == "battery_out")
    timeout_count = sum(1 for row in rows if row["done_reason"] == "timeout")
    collision_episode_count = sum(1 for row in rows if row["collision_count"] > 0)
    completed_counts = [row["completed_targets"] for row in rows]
    partial_count = sum(1 for row in rows if 0 < row["completed_targets"] < row["total_targets"])
    completed_all_count = sum(1 for row in rows if row["completed_targets"] >= row["total_targets"])
    per_episode_completion = [
        row["completed_targets"] / max(1, row["total_targets"])
        for row in rows
    ]

    return {
        "run_name": run_name,
        "policy": policy,
        "use_planner": bool(use_planner),
        "episodes": int(episodes),
        "success_rate": _rate(success_count, episodes),
        "task_completion_rate": float(mean(per_episode_completion)) if rows else 0.0,
        "return_to_base_rate": _rate(returned_count, episodes),
        "mean_episode_reward": float(mean([row["episode_reward"] for row in rows])) if rows else 0.0,
        "mean_episode_length": float(mean([row["episode_length"] for row in rows])) if rows else 0.0,
        "mean_path_length": float(mean([row["path_length"] for row in rows])) if rows else 0.0,
        "collision_rate": _rate(collision_episode_count, episodes),
        "collision_episode_rate": _rate(collision_episode_count, episodes),
        "mean_collisions": float(mean([row["collision_count"] for row in rows])) if rows else 0.0,
        "mean_collisions_per_episode": float(mean([row["collision_count"] for row in rows])) if rows else 0.0,
        "battery_out_rate": _rate(battery_out_count, episodes),
        "timeout_rate": _rate(timeout_count, episodes),
        "mean_remaining_battery": float(mean([row["remaining_battery"] for row in rows])) if rows else 0.0,
        "mean_repeated_cell_count": float(mean([row["repeated_cell_count"] for row in rows])) if rows else 0.0,
        "path_efficiency": float(mean([row["path_efficiency"] for row in rows])) if rows else 0.0,
        "normalized_reward": float(mean([row["normalized_reward"] for row in rows])) if rows else 0.0,
        "mean_completed_targets": float(mean(completed_counts)) if rows else 0.0,
        "completed_1_plus_rate": _rate(sum(1 for count in completed_counts if count >= 1), episodes),
        "completed_2_plus_rate": _rate(sum(1 for count in completed_counts if count >= 2), episodes),
        "completed_all_rate": _rate(completed_all_count, episodes),
        "completed_all_targets_rate": _rate(completed_all_count, episodes),
        "partial_completion_rate": _rate(partial_count, episodes),
        "mean_planning_time_ms": float(mean([row["mean_planning_time_ms"] for row in rows])) if rows else 0.0,
        "episodes_detail": rows,
    }


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / float(total)


def normalize_policy_name(policy: str) -> str:
    if policy in {"astar", "astar_oracle"}:
        return "astar_oracle"
    if policy in {"greedy", "greedy_heuristic", "heuristic"}:
        return "greedy"
    return policy


def make_baseline_policy(policy: str):
    normalized = normalize_policy_name(policy)
    if normalized == "random":
        return RandomPolicy()
    if normalized == "greedy":
        return GreedyHeuristicPolicy()
    if normalized == "astar_oracle":
        return AStarOraclePolicy()
    return None


def _episode_path_efficiency(info: Dict[str, Any], success: bool, actual_path_length: int) -> float:
    shortest = info.get("shortest_possible_path_length")
    if not success or shortest is None or actual_path_length <= 0:
        return 0.0
    return float(shortest) / float(actual_path_length)


def _normalized_reward(episode_reward: float, info: Dict[str, Any], config: Dict[str, Any]) -> float:
    reward_cfg = config.get("reward", {})
    task_bonus = float(reward_cfg.get("task_bonus", 10.0))
    return_bonus = float(reward_cfg.get("return_bonus", 20.0))
    total_targets = max(1, int(info.get("total_targets", 1)))
    positive_scale = max(1.0, task_bonus * total_targets + return_bonus)
    return float(episode_reward) / positive_scale


def _mean_frame_planning_time(frames: List[Dict[str, Any]]) -> float:
    values = [float(frame.get("planning_time_ms", 0.0)) for frame in frames if frame.get("action") is not None]
    return float(mean(values)) if values else 0.0


def _public_config(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "level",
        "grid_size",
        "base",
        "start",
        "targets",
        "obstacles",
        "max_steps",
        "max_battery",
        "reward_mode",
        "randomized_each_episode",
    ]
    return {key: config.get(key) for key in keys}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _try_export_replay_html(json_path: Path) -> str:
    try:
        from web.export_replay_html import export_html

        html_path = json_path.with_suffix(".html")
        export_html(json_path, html_path)
        return str(html_path)
    except Exception as exc:
        return f"html_export_failed: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to env YAML config.")
    parser.add_argument("--policy", choices=["random", "greedy", "greedy_heuristic", "heuristic", "astar", "astar_oracle", "ppo"], default="ppo")
    parser.add_argument("--model_path", default=None, help="PPO model checkpoint for --policy ppo.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--out", default=None, help="Output evaluation JSON path.")
    parser.add_argument("--replay_dir", default=None, help="Directory for success/failure replay JSON.")
    parser.add_argument("--save_replay", default=None, help="Alias for --replay_dir.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_planner", action="store_true")
    parser.add_argument("--plan_interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_env_config(args.config)
    run_name = args.run_name or (Path(args.out).parent.name if args.out else args.policy)
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "logs" / "eval" / run_name / "evaluation.json"
    replay_target = args.save_replay or args.replay_dir
    replay_dir = Path(replay_target) if replay_target else PROJECT_ROOT / "replays" / run_name
    metrics = evaluate_policy_run(
        config=config,
        policy=args.policy,
        episodes=args.episodes,
        run_name=run_name,
        model_path=args.model_path,
        out_path=out_path,
        replay_dir=replay_dir,
        seed=args.seed,
        use_planner=args.use_planner,
        plan_interval=args.plan_interval,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes_detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
