"""Print per-episode metric audit rows for PatrolEnv policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_planner import LLMPlanner
from agents.ppo_executor import PPOExecutor
from envs.patrol_env import PatrolEnv, load_env_config
from train.evaluate import make_baseline_policy, normalize_policy_name


def audit_episode_rows(
    config: Dict[str, Any],
    policy: str,
    episodes: int,
    seed: int = 0,
    model_path: Optional[str] = None,
    use_planner: bool = False,
) -> List[Dict[str, Any]]:
    rows = []
    baseline_policy = make_baseline_policy(policy)
    ppo_executor = PPOExecutor(model_path) if policy == "ppo" and model_path else None
    planner = LLMPlanner() if use_planner else None
    memory: List[Dict[str, Any]] = []

    for episode_id in range(episodes):
        env = PatrolEnv(config)
        observation, _ = env.reset(seed=seed + episode_id)
        done = False
        while not done:
            if planner is not None:
                planner_output = planner.plan(env.get_state(), memory)
                env.set_llm_subgoal(planner_output["next_subgoal"])
                observation = env._get_obs()
            if baseline_policy is not None:
                action = baseline_policy.select_action(env)
            elif ppo_executor is not None:
                action = ppo_executor.predict(observation, deterministic=True)
            else:
                raise ValueError("--model_path is required when --policy ppo")
            observation, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

        completed = int(info["completed_targets"])
        total = int(info["total_targets"])
        returned_to_base = bool(info["returned_to_base"])
        strict_success = bool(completed == total and returned_to_base)
        rows.append(
            {
                "episode_id": episode_id,
                "policy": normalize_policy_name(policy),
                "success": bool(info["success"]),
                "strict_success_check": strict_success,
                "completed_targets": completed,
                "total_targets": total,
                "agent_final_position": info["agent_final_position"],
                "base_position": info["base_position"],
                "returned_to_base": returned_to_base,
                "battery_left": info["remaining_battery"],
                "episode_length": env.step_count,
                "collision_count": info["collision_count"],
                "reason_done": info["done_reason"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=["random", "greedy", "heuristic", "astar", "astar_oracle", "ppo"], default="greedy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--use_planner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit_episode_rows(
        load_env_config(args.config),
        policy=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        model_path=args.model_path,
        use_planner=args.use_planner,
    )
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
