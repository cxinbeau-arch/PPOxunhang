"""Build an index of real evaluation, replay, model, and training artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "web" / "artifacts_index.json"


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def infer_stage(run_name: str) -> str:
    name = run_name.lower()
    if "stage1" in name or "curriculum_stage1" in name:
        return "stage1"
    if "stage2" in name or "curriculum_stage2" in name:
        return "stage2"
    if "stage3" in name or "curriculum_stage3" in name:
        return "stage3"
    if "basic" in name:
        return "basic"
    if "randomized" in name:
        return "randomized"
    if "exp00" in name:
        return "baseline"
    return "unknown"


def infer_config(run_name: str) -> Optional[str]:
    stage = infer_stage(run_name)
    mapping = {
        "stage1": "configs/env_curriculum_stage1.yaml",
        "stage2": "configs/env_curriculum_stage2.yaml",
        "stage3": "configs/env_curriculum_stage3.yaml",
        "basic": "configs/env_basic.yaml",
        "randomized": "configs/env_randomized.yaml",
    }
    if stage in mapping:
        return mapping[stage]
    name = run_name.lower()
    if "obstacle" in name:
        return "configs/env_obstacle.yaml"
    if "multitask" in name:
        return "configs/env_multitask.yaml"
    if "astar" in name or "greedy" in name or "random" in name:
        return "configs/env_randomized.yaml" if "randomized" in name else None
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _run_name_from_eval(path: Path, data: Dict[str, Any]) -> str:
    return str(data.get("run_name") or path.parent.name)


def _run_name_from_replay(path: Path) -> str:
    try:
        relative = path.relative_to(PROJECT_ROOT / "replays")
    except ValueError:
        return path.parent.name
    if relative.parts:
        if relative.parts[0] == "live":
            return path.stem.replace("_live_episode", "")
        return relative.parts[0]
    return path.parent.name


def _read_comparison_policies(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "run_name" not in df.columns:
        return {}
    policies: Dict[str, str] = {}
    for _, row in df.iterrows():
        run_name = str(row.get("run_name") or "")
        if not run_name:
            continue
        policy = row.get("policy")
        if pd.notna(policy):
            policies[run_name] = str(policy)
    return policies


def _sorted_rel(paths: Iterable[Path]) -> List[str]:
    return sorted(rel(path) for path in paths)


def build_index(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    global PROJECT_ROOT
    PROJECT_ROOT = project_root.resolve()

    comparison_table = PROJECT_ROOT / "logs" / "eval" / "comparison_table.csv"
    policies = _read_comparison_policies(comparison_table)

    eval_files_by_run: Dict[str, List[Path]] = defaultdict(list)
    eval_policy_by_run: Dict[str, str] = {}
    for path in sorted((PROJECT_ROOT / "logs" / "eval").glob("*/evaluation*.json")):
        data = _load_json(path)
        run_name = _run_name_from_eval(path, data)
        eval_files_by_run[run_name].append(path)
        if data.get("policy"):
            eval_policy_by_run[run_name] = str(data["policy"])

    replay_files_by_run: Dict[str, List[Path]] = defaultdict(list)
    html_replays_by_run: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted((PROJECT_ROOT / "replays").rglob("*.json")):
        replay_files_by_run[_run_name_from_replay(path)].append(path)
    for path in sorted((PROJECT_ROOT / "replays").rglob("*.html")):
        html_replays_by_run[_run_name_from_replay(path)].append(path)

    best_models = {path.parent.name: path for path in sorted((PROJECT_ROOT / "models").glob("*/best_model.zip"))}
    final_models = {path.parent.name: path for path in sorted((PROJECT_ROOT / "models").glob("*/final_model.zip"))}

    train_logs_by_run: Dict[str, List[Path]] = defaultdict(list)
    for base in [PROJECT_ROOT / "logs" / "monitor", PROJECT_ROOT / "logs" / "tensorboard"]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            try:
                parts = path.relative_to(base).parts
            except ValueError:
                continue
            if parts:
                train_logs_by_run[parts[0]].append(path)

    all_runs = sorted(
        set(policies)
        | set(eval_files_by_run)
        | set(replay_files_by_run)
        | set(html_replays_by_run)
        | set(best_models)
        | set(final_models)
        | set(train_logs_by_run)
    )
    all_runs = [run_name for run_name in all_runs if run_name and not run_name.startswith(".")]

    runs: List[Dict[str, Any]] = []
    for run_name in all_runs:
        policy = policies.get(run_name) or eval_policy_by_run.get(run_name)
        if policy is None and run_name in best_models:
            policy = "ppo"
        runs.append(
            {
                "run_name": run_name,
                "policy": policy or "unknown",
                "stage": infer_stage(run_name),
                "config": infer_config(run_name),
                "model_path": rel(best_models[run_name]) if run_name in best_models else None,
                "final_model_path": rel(final_models[run_name]) if run_name in final_models else None,
                "eval_files": _sorted_rel(eval_files_by_run.get(run_name, [])),
                "replay_files": _sorted_rel(replay_files_by_run.get(run_name, [])),
                "html_replays": _sorted_rel(html_replays_by_run.get(run_name, [])),
                "training_logs": _sorted_rel(train_logs_by_run.get(run_name, [])),
            }
        )

    index = {
        "project_root": str(PROJECT_ROOT),
        "comparison_table": rel(comparison_table) if comparison_table.exists() else None,
        "runs": runs,
        "missing": {
            "comparison_table": not comparison_table.exists(),
            "logs_eval": not (PROJECT_ROOT / "logs" / "eval").exists(),
            "replays": not (PROJECT_ROOT / "replays").exists(),
            "models": not (PROJECT_ROOT / "models").exists(),
            "monitor_logs": not (PROJECT_ROOT / "logs" / "monitor").exists(),
            "tensorboard_logs": not (PROJECT_ROOT / "logs" / "tensorboard").exists(),
        },
    }
    return index


def write_index(index: Dict[str, Any], out_path: Path = INDEX_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(INDEX_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_index(Path(args.project_root))
    out = write_index(index, Path(args.out))
    print(out)
    print(json.dumps({"runs": [run["run_name"] for run in index["runs"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
