"""Create plots and CSV tables from real monitor/evaluation logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EVAL_COLUMNS = [
    "run_name",
    "policy",
    "success_rate",
    "task_completion_rate",
    "return_to_base_rate",
    "mean_episode_reward",
    "mean_episode_length",
    "mean_path_length",
    "collision_episode_rate",
    "mean_collisions_per_episode",
    "battery_out_rate",
    "timeout_rate",
    "mean_completed_targets",
    "mean_remaining_battery",
    "mean_repeated_cell_count",
    "path_efficiency",
    "normalized_reward",
    "completed_all_targets_rate",
    "partial_completion_rate",
    "mean_planning_time_ms",
]

DISPLAY_POLICIES = ["random", "greedy", "astar_oracle", "ppo"]


def _monitor_path(run_name: str) -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "logs" / "monitor" / run_name / "monitor.csv",
        PROJECT_ROOT / "logs" / "monitor" / run_name / "monitor.csv.monitor.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def _read_monitor(run_name: str, window: int = 20) -> pd.DataFrame:
    monitor_path = _monitor_path(run_name)
    if monitor_path is None:
        raise FileNotFoundError(f"Monitor CSV not found for run {run_name}")
    df = pd.read_csv(monitor_path, comment="#")
    if "r" not in df.columns or "l" not in df.columns:
        raise ValueError(f"Monitor CSV lacks required r/l columns: {monitor_path}")
    curve = pd.DataFrame(
        {
            "episode": np.arange(len(df), dtype=int),
            "episode_reward": df["r"].astype(float),
            "episode_length": df["l"].astype(float),
            "elapsed_time": df["t"].astype(float) if "t" in df.columns else np.nan,
        }
    )
    curve["episode_reward_rolling"] = curve["episode_reward"].rolling(window, min_periods=1).mean()
    curve["episode_length_rolling"] = curve["episode_length"].rolling(window, min_periods=1).mean()
    curve["source_monitor"] = str(monitor_path.relative_to(PROJECT_ROOT))
    return curve


def _read_eval_npz(run_name: str) -> pd.DataFrame:
    eval_path = PROJECT_ROOT / "logs" / "eval" / run_name / "evaluations.npz"
    if not eval_path.exists():
        return pd.DataFrame()
    try:
        data = np.load(eval_path, allow_pickle=True)
    except Exception:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    timesteps = data.get("timesteps", [])
    results = data.get("results", [])
    lengths = data.get("ep_lengths", [])
    for index, timestep in enumerate(timesteps):
        rewards = np.asarray(results[index], dtype=float) if index < len(results) else np.asarray([], dtype=float)
        ep_lengths = np.asarray(lengths[index], dtype=float) if index < len(lengths) else np.asarray([], dtype=float)
        rows.append(
            {
                "eval_index": index,
                "eval_timestep": int(timestep),
                "eval_reward": float(np.mean(rewards)) if rewards.size else np.nan,
                "eval_reward_std": float(np.std(rewards)) if rewards.size else np.nan,
                "eval_episode_length": float(np.mean(ep_lengths)) if ep_lengths.size else np.nan,
                "source_eval_npz": str(eval_path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def export_training_curve(run_name: str, window: int = 20) -> pd.DataFrame:
    curve = _read_monitor(run_name, window=window)
    eval_curve = _read_eval_npz(run_name)
    if not eval_curve.empty:
        for col in eval_curve.columns:
            if pd.api.types.is_numeric_dtype(eval_curve[col]):
                curve[col] = np.nan
            else:
                curve[col] = pd.Series([None] * len(curve), dtype=object)
        for row_index, (_, row) in enumerate(eval_curve.iterrows()):
            if row_index >= len(curve):
                break
            for col in eval_curve.columns:
                curve.loc[row_index, col] = row[col]

    out_dir = PROJECT_ROOT / "logs" / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / "training_curve.csv", index=False)
    return curve


def plot_monitor(run_name: str, window: int = 20) -> Path:
    import matplotlib.pyplot as plt

    curve = export_training_curve(run_name, window=window)
    out_dir = PROJECT_ROOT / "logs" / "eval" / run_name
    out_path = out_dir / "training_curve.png"

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    axes[0].plot(curve["episode"], curve["episode_reward"], alpha=0.28, label="episode reward")
    axes[0].plot(curve["episode"], curve["episode_reward_rolling"], label=f"rolling reward ({window})")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(curve["episode"], curve["episode_length"], alpha=0.35, color="tab:orange", label="episode length")
    axes[1].plot(curve["episode"], curve["episode_length_rolling"], color="tab:red", label=f"rolling length ({window})")
    axes[1].set_ylabel("Length")
    axes[1].legend()

    if "eval_timestep" in curve.columns and "eval_reward" in curve.columns:
        eval_rows = curve.dropna(subset=["eval_timestep", "eval_reward"])
        if not eval_rows.empty:
            axes[2].plot(eval_rows["eval_timestep"], eval_rows["eval_reward"], marker="o", label="eval reward")
            axes[2].set_xlabel("Timestep")
            axes[2].set_ylabel("Eval reward")
            axes[2].legend()
        else:
            axes[2].axis("off")
            axes[2].text(0.05, 0.5, "No eval reward curve in evaluations.npz", transform=axes[2].transAxes)
    else:
        axes[2].axis("off")
        axes[2].text(0.05, 0.5, "No eval reward curve file found", transform=axes[2].transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def collect_eval_table() -> pd.DataFrame:
    rows = []
    for path in (PROJECT_ROOT / "logs" / "eval").glob("*/evaluation.json"):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("policy") == "heuristic":
            continue
        rows.append({key: data.get(key) for key in EVAL_COLUMNS})
    return pd.DataFrame(rows)


def write_eval_table() -> Path:
    table = collect_eval_table()
    out = PROJECT_ROOT / "logs" / "eval" / "comparison_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def write_clean_eval_table() -> Path:
    source = PROJECT_ROOT / "logs" / "eval" / "comparison_table.csv"
    if source.exists():
        table = pd.read_csv(source)
    else:
        table = collect_eval_table()
    if "policy" in table.columns:
        table = table[table["policy"].astype(str).isin(DISPLAY_POLICIES)].copy()
    out = PROJECT_ROOT / "logs" / "eval" / "comparison_table_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def _monitor_run_names() -> List[str]:
    base = PROJECT_ROOT / "logs" / "monitor"
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--all", action="store_true", help="Export training curves for every run with monitor.csv.")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--write_eval_table", action="store_true")
    parser.add_argument("--write_clean_eval_table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_names: Iterable[str] = []
    if args.all:
        run_names = _monitor_run_names()
    elif args.run_name:
        run_names = [args.run_name]

    for run_name in run_names:
        try:
            print(plot_monitor(run_name, args.window))
        except Exception as exc:
            print(f"[warning] {run_name}: {exc}")

    if args.write_eval_table:
        print(write_eval_table())
    if args.write_clean_eval_table:
        print(write_clean_eval_table())


if __name__ == "__main__":
    main()
