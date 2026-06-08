"""Train a PPO executor with Stable-Baselines3."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.patrol_env import PatrolEnv, load_env_config, read_yaml_file
from train.evaluate import evaluate_policy_run
from train.make_plots import plot_monitor


def load_ppo_config(path: Path) -> Dict[str, Any]:
    raw = read_yaml_file(path)
    return raw.get("ppo", raw)


def train(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "PPO training requires gymnasium, torch, and stable-baselines3. "
            "Install them with `pip install -r requirements.txt` and use Python 3.10+."
        ) from exc

    env_config = load_env_config(args.config)
    ppo_config = load_ppo_config(Path(args.ppo_config))
    policy_name = ppo_config.pop("policy", "MlpPolicy")

    model_dir = PROJECT_ROOT / "models" / args.run_name
    tensorboard_dir = PROJECT_ROOT / "logs" / "tensorboard" / args.run_name
    monitor_dir = PROJECT_ROOT / "logs" / "monitor" / args.run_name
    eval_dir = PROJECT_ROOT / "logs" / "eval" / args.run_name
    replay_dir = PROJECT_ROOT / "replays" / args.run_name
    for path in [model_dir, tensorboard_dir, monitor_dir, eval_dir, replay_dir]:
        path.mkdir(parents=True, exist_ok=True)

    def make_train_env():
        env = PatrolEnv(env_config)
        return Monitor(
            env,
            filename=str(monitor_dir / "monitor.csv"),
            info_keywords=("success", "completed_targets", "collision_count", "remaining_battery", "done_reason"),
        )

    train_env = DummyVecEnv([make_train_env])
    eval_env = Monitor(PatrolEnv(env_config), info_keywords=("success", "completed_targets", "collision_count"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(eval_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    if args.resume_model:
        model = PPO.load(
            args.resume_model,
            env=train_env,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            verbose=args.verbose,
            **ppo_config,
        )
        reset_num_timesteps = False
    else:
        model = PPO(
            policy_name,
            train_env,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            verbose=args.verbose,
            **ppo_config,
        )
        reset_num_timesteps = True
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=args.progress_bar,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(str(model_dir / "final_model.zip"))

    best_model = model_dir / "best_model.zip"
    eval_model_path = best_model if best_model.exists() else model_dir / "final_model.zip"
    metrics = evaluate_policy_run(
        config=env_config,
        policy="ppo",
        episodes=args.final_eval_episodes,
        run_name=args.run_name,
        model_path=str(eval_model_path),
        out_path=eval_dir / "evaluation.json",
        replay_dir=replay_dir,
        seed=args.seed + 1000,
        use_planner=args.use_planner,
    )
    try:
        curve_path = plot_monitor(args.run_name)
        target_curve_path = eval_dir / "training_curve.png"
        if curve_path.resolve() != target_curve_path.resolve():
            shutil.copyfile(curve_path, target_curve_path)
        metrics["training_curve"] = str(eval_dir / "training_curve.png")
    except Exception as exc:
        metrics["training_curve_error"] = str(exc)
    _write_training_manifest(args, env_config, ppo_config, metrics, model_dir / "training_manifest.json")
    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes_detail"}, ensure_ascii=False, indent=2))


def _write_training_manifest(
    args: argparse.Namespace,
    env_config: Dict[str, Any],
    ppo_config: Dict[str, Any],
    metrics: Dict[str, Any],
    path: Path,
) -> None:
    payload = {
        "run_name": args.run_name,
        "config": str(args.config),
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "env_level": env_config.get("level"),
        "reward_mode": env_config.get("reward_mode"),
        "ppo": ppo_config,
        "final_metrics": {k: v for k, v in metrics.items() if k != "episodes_detail"},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ppo_config", default=str(PROJECT_ROOT / "configs" / "ppo_default.yaml"))
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--resume_model", default=None, help="Existing PPO checkpoint to continue training from.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--final_eval_episodes", type=int, default=100)
    parser.add_argument("--use_planner", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress_bar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
