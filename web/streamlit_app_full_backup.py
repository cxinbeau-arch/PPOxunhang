"""Streamlit dashboard for real NavAgent-PPO training artifacts."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.build_artifacts_index import build_index, infer_stage, write_index
from web.replay_loader import ACTION_NAMES, load_replay


INDEX_PATH = PROJECT_ROOT / "web" / "artifacts_index.json"
COMPARISON_PATH = PROJECT_ROOT / "logs" / "eval" / "comparison_table.csv"
DEFAULT_POLICIES = ["random", "greedy", "astar_oracle", "ppo"]


def _path(relative_or_abs: str | Path | None) -> Optional[Path]:
    if not relative_or_abs:
        return None
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def load_index() -> Dict[str, Any]:
    if INDEX_PATH.exists():
        try:
            return load_json(INDEX_PATH)
        except Exception:
            pass
    index = build_index(PROJECT_ROOT)
    write_index(index, INDEX_PATH)
    return index


def read_comparison_table() -> Optional[pd.DataFrame]:
    if not COMPARISON_PATH.exists():
        return None
    df = pd.read_csv(COMPARISON_PATH)
    if "run_name" in df.columns:
        df["stage"] = df["run_name"].astype(str).map(infer_stage)
    return df


def run_lookup(index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {run["run_name"]: run for run in index.get("runs", [])}


def metric_bar(df: pd.DataFrame, metric: str) -> Optional[go.Figure]:
    if metric not in df.columns:
        return None
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return None
    color = plot_df["stage"] if "stage" in plot_df.columns else None
    fig = go.Figure()
    for stage, group in plot_df.groupby(color if color is not None else pd.Series(["all"] * len(plot_df))):
        fig.add_trace(go.Bar(x=group["run_name"], y=group[metric], name=str(stage)))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=28, b=80),
        xaxis_title="run_name",
        yaxis_title=metric,
        barmode="group",
    )
    return fig


def filter_table(df: pd.DataFrame, st: Any) -> pd.DataFrame:
    filtered = df.copy()
    col1, col2, col3 = st.columns(3)
    with col1:
        if "policy" in filtered.columns:
            options = sorted(str(value) for value in filtered["policy"].dropna().unique())
            default = [policy for policy in DEFAULT_POLICIES if policy in options]
            selected = st.multiselect("policy", options, default=default or options)
            deprecated = [policy for policy in options if policy not in DEFAULT_POLICIES]
            if deprecated:
                st.caption(f"Deprecated/hidden by default: {', '.join(deprecated)}")
            if selected:
                filtered = filtered[filtered["policy"].astype(str).isin(selected)]
    with col2:
        if "stage" in filtered.columns:
            options = sorted(str(value) for value in filtered["stage"].dropna().unique())
            selected = st.multiselect("stage", options, default=options)
            if selected:
                filtered = filtered[filtered["stage"].astype(str).isin(selected)]
    with col3:
        if "run_name" in filtered.columns:
            options = sorted(str(value) for value in filtered["run_name"].dropna().unique())
            selected = st.multiselect("run_name", options, default=options)
            if selected:
                filtered = filtered[filtered["run_name"].astype(str).isin(selected)]
    return filtered


def summarize_seed_files(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eval_files = [_path(path) for path in run.get("eval_files", [])]
    eval_files = [path for path in eval_files if path is not None and path.exists()]
    summary_files = [path for path in eval_files if "summary_multi_seed" in path.name]
    if summary_files:
        data = load_json(summary_files[0])
        return {
            "run_name": run["run_name"],
            "stage": run.get("stage"),
            "source": summary_files[0].relative_to(PROJECT_ROOT).as_posix(),
            "success_rate_mean": data.get("success_rate_mean"),
            "success_rate_std": data.get("success_rate_std"),
            "task_completion_rate_mean": data.get("task_completion_rate_mean"),
            "task_completion_rate_std": data.get("task_completion_rate_std"),
            "battery_out_rate_mean": data.get("battery_out_rate_mean"),
            "battery_out_rate_std": data.get("battery_out_rate_std"),
            "has_multi_seed_summary": True,
        }

    seed_files = [path for path in eval_files if "evaluation_seed" in path.name]
    if not seed_files:
        return None
    rows = [load_json(path) for path in seed_files]
    result = {
        "run_name": run["run_name"],
        "stage": run.get("stage"),
        "source": ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in seed_files),
        "has_multi_seed_summary": False,
    }
    for metric in ["success_rate", "task_completion_rate", "battery_out_rate"]:
        values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
        if values:
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return result


def render_overview(st: Any, comparison: Optional[pd.DataFrame]) -> None:
    st.header("区域 A：总览 Dashboard")
    if comparison is None:
        st.warning("未找到 logs/eval/comparison_table.csv。")
        return

    filtered = filter_table(comparison, st)
    st.dataframe(filtered, use_container_width=True)

    metrics = [
        "success_rate",
        "task_completion_rate",
        "return_to_base_rate",
        "mean_episode_reward",
        "mean_episode_length",
        "battery_out_rate",
        "mean_completed_targets",
        "path_efficiency",
    ]
    cols = st.columns(2)
    for index, metric in enumerate(metrics):
        with cols[index % 2]:
            fig = metric_bar(filtered, metric)
            if fig is None:
                st.warning(f"comparison_table.csv 中未找到可绘制列：{metric}")
            else:
                st.plotly_chart(fig, use_container_width=True)


def render_stage_compare(st: Any, index: Dict[str, Any], comparison: Optional[pd.DataFrame]) -> None:
    st.header("区域 B：Stage 对比")
    if comparison is not None and "policy" in comparison.columns:
        stage_df = comparison[
            (comparison["stage"].isin(["stage1", "stage2", "stage3"]))
            & (comparison["policy"].astype(str).str.lower() == "ppo")
        ].copy()
        if stage_df.empty:
            st.warning("comparison_table.csv 中未找到 Stage 1/2/3 的 PPO 行。")
        else:
            visible_cols = [
                col
                for col in [
                    "run_name",
                    "stage",
                    "success_rate",
                    "task_completion_rate",
                    "return_to_base_rate",
                    "mean_episode_reward",
                    "battery_out_rate",
                    "mean_completed_targets",
                    "path_efficiency",
                ]
                if col in stage_df.columns
            ]
            st.dataframe(stage_df[visible_cols], use_container_width=True)
            for metric in ["success_rate", "task_completion_rate", "battery_out_rate"]:
                fig = metric_bar(stage_df, metric)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("未能读取 comparison_table.csv，Stage 总表对比不可用。")

    rows = []
    for run in index.get("runs", []):
        if run.get("stage") in {"stage1", "stage2", "stage3"} and run.get("policy") == "ppo":
            summary = summarize_seed_files(run)
            if summary:
                rows.append(summary)
            else:
                st.warning(f"{run['run_name']} 未找到 evaluation_seed*.json 或 evaluation_summary_multi_seed.json。")
    if rows:
        seed_df = pd.DataFrame(rows)
        st.subheader("多 seed 结果")
        st.dataframe(seed_df, use_container_width=True)
        for row in rows:
            if not row.get("has_multi_seed_summary"):
                st.warning(f"{row['run_name']} 没有多 seed summary 文件，页面仅基于已存在的 seed evaluation 计算。")
    else:
        st.warning("未发现 Stage 1/2/3 多 seed evaluation 文件。")


def replay_options(index: Dict[str, Any], run_name: str, episode_type: str) -> List[str]:
    run = run_lookup(index).get(run_name, {})
    files = list(run.get("replay_files", []))
    if episode_type == "all":
        return files
    return [path for path in files if episode_type in Path(path).stem.lower()]


def draw_replay_figure(replay: Dict[str, Any], step_index: int) -> go.Figure:
    steps = replay.get("steps", [])
    step = steps[step_index] if steps else {}
    grid = step.get("grid_size") or replay.get("config", {}).get("grid_size") or [12, 12]
    width, height = int(grid[0]), int(grid[1])

    fig = go.Figure()
    cell_x = []
    cell_y = []
    cell_data = []
    for y in range(height):
        for x in range(width):
            cell_x.append(x)
            cell_y.append(y)
            cell_data.append([x, y])
    fig.add_trace(
        go.Scatter(
            x=cell_x,
            y=cell_y,
            mode="markers",
            marker=dict(size=24, color="rgba(0,0,0,0)"),
            customdata=cell_data,
            hovertemplate="cell=(%{customdata[0]}, %{customdata[1]})<extra></extra>",
            name="clickable cells",
            showlegend=False,
        )
    )

    trajectory = [item.get("agent_pos") for item in steps[: step_index + 1] if item.get("agent_pos") is not None]
    if trajectory:
        fig.add_trace(
            go.Scatter(
                x=[pos[0] for pos in trajectory],
                y=[pos[1] for pos in trajectory],
                mode="lines",
                line=dict(color="#64748b", width=3),
                name="trajectory",
            )
        )

    obstacles = step.get("obstacles") or []
    if obstacles:
        fig.add_trace(
            go.Scatter(
                x=[pos[0] for pos in obstacles],
                y=[pos[1] for pos in obstacles],
                mode="markers",
                marker=dict(size=28, color="#334155", symbol="square"),
                name="obstacle",
            )
        )

    targets = step.get("targets") or []
    visited = {tuple(pos) for pos in step.get("visited_targets") or []}
    unvisited_targets = [pos for pos in targets if tuple(pos) not in visited]
    if unvisited_targets:
        fig.add_trace(
            go.Scatter(
                x=[pos[0] for pos in unvisited_targets],
                y=[pos[1] for pos in unvisited_targets],
                mode="markers",
                marker=dict(size=22, color="#f97316", symbol="diamond"),
                name="unvisited target",
            )
        )
    if visited:
        fig.add_trace(
            go.Scatter(
                x=[pos[0] for pos in visited],
                y=[pos[1] for pos in visited],
                mode="markers",
                marker=dict(size=22, color="#16a34a", symbol="diamond"),
                name="visited target",
            )
        )

    base = step.get("base_pos")
    if base:
        fig.add_trace(
            go.Scatter(
                x=[base[0]],
                y=[base[1]],
                mode="markers",
                marker=dict(size=28, color="#2563eb", symbol="square"),
                name="base",
            )
        )

    agent = step.get("agent_pos")
    if agent:
        fig.add_trace(
            go.Scatter(
                x=[agent[0]],
                y=[agent[1]],
                mode="markers",
                marker=dict(size=30, color="#dc2626", symbol="circle"),
                name="agent",
            )
        )

    fig.update_xaxes(range=[-0.5, width - 0.5], dtick=1, showgrid=True, zeroline=False)
    fig.update_yaxes(range=[height - 0.5, -0.5], dtick=1, showgrid=True, zeroline=False, scaleanchor="x")
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))
    return fig


def extract_selected_cell(selection: Any) -> Optional[List[int]]:
    if not selection:
        return None
    points: Any = None
    if isinstance(selection, dict):
        points = selection.get("selection", {}).get("points")
    else:
        selection_obj = getattr(selection, "selection", None)
        points = getattr(selection_obj, "points", None) if selection_obj is not None else None
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if custom and len(custom) >= 2:
        return [int(custom[0]), int(custom[1])]
    return None


def cell_info(step: Dict[str, Any], cell: List[int]) -> Dict[str, Any]:
    position = tuple(cell)
    targets = {tuple(pos) for pos in step.get("targets") or []}
    visited = {tuple(pos) for pos in step.get("visited_targets") or []}
    obstacles = {tuple(pos) for pos in step.get("obstacles") or []}
    return {
        "coord": cell,
        "is_obstacle": position in obstacles,
        "is_target": position in targets,
        "is_visited_target": position in visited,
        "is_base": position == tuple(step.get("base_pos") or []),
        "is_agent_current": position == tuple(step.get("agent_pos") or []),
    }


def action_prob_figure(action_probs: Dict[str, float]) -> go.Figure:
    names = [name for name in ACTION_NAMES if name in action_probs]
    names.extend(name for name in action_probs if name not in names)
    fig = go.Figure(go.Bar(x=names, y=[action_probs[name] for name in names], marker_color="#2563eb"))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=30), yaxis=dict(range=[0, 1]))
    return fig


def render_replay(st: Any, index: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    st.header("区域 C：交互式 Episode Replay")
    lookup = run_lookup(index)
    run_names = sorted(run for run, item in lookup.items() if item.get("replay_files"))
    if not run_names:
        st.warning("未发现 replay JSON 文件。")
        return None

    left, middle, right = st.columns([0.9, 1.55, 0.9])
    with left:
        run_name = st.selectbox("run_name", run_names, key="replay_run")
        episode_type = st.selectbox("episode 类型", ["all", "success", "failure", "worst", "partial"], key="replay_type")
        files = replay_options(index, run_name, episode_type)
        if not files:
            st.warning(f"{run_name} 没有 {episode_type} replay。")
            return None
        replay_file = st.selectbox("replay 文件", files, format_func=lambda path: Path(path).name, key="replay_file")

    path = _path(replay_file)
    if path is None or not path.exists():
        st.warning(f"replay 文件缺失：{replay_file}")
        return None

    replay = load_replay(path)
    steps = replay.get("steps", [])
    if not steps:
        st.warning("该 replay 没有可视化 steps。")
        return replay

    state_key = f"step::{replay_file}"
    play_key = f"play::{replay_file}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    if play_key not in st.session_state:
        st.session_state[play_key] = False

    with left:
        max_step = len(steps) - 1
        st.session_state[state_key] = st.slider("step slider", 0, max_step, int(st.session_state[state_key]))
        b1, b2, b3 = st.columns(3)
        if b1.button("上一步"):
            st.session_state[state_key] = max(0, int(st.session_state[state_key]) - 1)
        if b2.button("下一步"):
            st.session_state[state_key] = min(max_step, int(st.session_state[state_key]) + 1)
        if b3.button("重置"):
            st.session_state[state_key] = 0
        speed = st.slider("播放速度 step/s", 1, 8, 3)
        if st.button("自动播放 / 暂停"):
            st.session_state[play_key] = not bool(st.session_state[play_key])

    step_index = int(st.session_state[state_key])
    step = steps[step_index]

    with middle:
        selection = None
        fig = draw_replay_figure(replay, step_index)
        try:
            selection = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="replay_map",
            )
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("当前 Streamlit 版本不支持 Plotly 点击事件，可用下方坐标输入查看 cell 信息。")

        selected_cell = extract_selected_cell(selection)
        coord_cols = st.columns(2)
        with coord_cols[0]:
            x = st.number_input("cell x", min_value=0, value=int((selected_cell or step.get("agent_pos") or [0, 0])[0]), step=1)
        with coord_cols[1]:
            y = st.number_input("cell y", min_value=0, value=int((selected_cell or step.get("agent_pos") or [0, 0])[1]), step=1)
        st.json(cell_info(step, [int(x), int(y)]))

    with right:
        st.metric("step index", step.get("step"))
        st.metric("battery", f"{step.get('battery', 0.0):.1f}")
        st.metric("cum_reward", f"{step.get('cum_reward', 0.0):.3f}")
        st.json(
            {
                "agent_pos": step.get("agent_pos"),
                "action": step.get("action"),
                "reward": step.get("reward"),
                "completed_targets": len(step.get("visited_targets") or []),
                "done": step.get("done"),
                "info": step.get("info"),
            }
        )
        if step.get("action_probs"):
            st.plotly_chart(action_prob_figure(step["action_probs"]), use_container_width=True)
        elif step.get("step") == 0 and step.get("action") == "start":
            st.info("初始状态，无动作概率")
        else:
            st.warning("该 replay 未保存 action_probs")

    if st.session_state.get(play_key):
        time.sleep(1.0 / max(1, int(speed)))
        st.session_state[state_key] = (int(st.session_state[state_key]) + 1) % len(steps)
        rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if rerun:
            rerun()

    return {"replay": replay, "step": step}


def load_training_curve(run_name: str) -> tuple[Optional[pd.DataFrame], str]:
    csv_path = PROJECT_ROOT / "logs" / "eval" / run_name / "training_curve.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path.relative_to(PROJECT_ROOT).as_posix()

    monitor_path = PROJECT_ROOT / "logs" / "monitor" / run_name / "monitor.csv"
    if not monitor_path.exists():
        alt = PROJECT_ROOT / "logs" / "monitor" / run_name / "monitor.csv.monitor.csv"
        monitor_path = alt if alt.exists() else monitor_path
    if monitor_path.exists():
        df = pd.read_csv(monitor_path, comment="#")
        result = pd.DataFrame(
            {
                "episode": np.arange(len(df)),
                "episode_reward": df["r"] if "r" in df.columns else np.nan,
                "episode_length": df["l"] if "l" in df.columns else np.nan,
                "elapsed_time": df["t"] if "t" in df.columns else np.nan,
            }
        )
        return result, monitor_path.relative_to(PROJECT_ROOT).as_posix()

    event_files = list((PROJECT_ROOT / "logs" / "tensorboard" / run_name).rglob("events.out.tfevents.*"))
    if event_files:
        return None, "发现 TensorBoard event 文件，但当前页面未解析 event，请运行 train/make_plots.py 导出 training_curve.csv。"
    return None, "当前 run 未找到训练曲线文件，请运行 train/make_plots.py 导出。"


def render_training_curves(st: Any, index: Dict[str, Any]) -> None:
    st.header("区域 D：训练曲线")
    run_names = sorted(run["run_name"] for run in index.get("runs", []))
    if not run_names:
        st.warning("artifacts_index.json 中没有 run。")
        return
    run_name = st.selectbox("选择 run", run_names, key="curve_run")
    df, source = load_training_curve(run_name)
    if df is None:
        st.warning(source)
        return
    st.caption(f"source: {source}")
    st.dataframe(df.head(200), use_container_width=True)

    plot_specs = [
        ("episode_reward", "episode reward 曲线"),
        ("success_rate", "success_rate 曲线"),
        ("episode_length", "episode length 曲线"),
        ("eval_reward", "eval reward 曲线"),
        ("mean_episode_reward", "eval reward 曲线"),
    ]
    x_col = "episode" if "episode" in df.columns else None
    for col, title in plot_specs:
        if col not in df.columns:
            st.warning(f"训练曲线中没有列：{col}")
            continue
        fig = go.Figure(go.Scatter(x=df[x_col] if x_col else df.index, y=df[col], mode="lines", name=col))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=32, b=40), title=title)
        st.plotly_chart(fig, use_container_width=True)

    png_path = PROJECT_ROOT / "logs" / "eval" / run_name / "training_curve.png"
    if png_path.exists():
        st.image(str(png_path), caption=png_path.relative_to(PROJECT_ROOT).as_posix())


def render_live_rollout(st: Any, index: Dict[str, Any]) -> None:
    st.header("区域 E：模型 Live Rollout")
    runs = [run for run in index.get("runs", []) if run.get("model_path")]
    if not runs:
        st.warning("未找到 models/{run_name}/best_model.zip。")
        return
    run_names = [run["run_name"] for run in runs]
    selected = st.selectbox("选择 PPO checkpoint", run_names, key="live_run")
    run = run_lookup(index)[selected]
    model_path = _path(run.get("model_path"))
    config_path = _path(run.get("config"))
    st.write({"model": run.get("model_path"), "config": run.get("config")})
    if config_path is None or not config_path.exists():
        st.warning(f"未找到可加载 config：{run.get('config')}")
        return
    if model_path is None or not model_path.exists():
        st.warning(f"未找到模型：{run.get('model_path')}")
        return

    if st.button("Run Live Rollout"):
        try:
            from envs.patrol_env import load_env_config
            from train.evaluate import evaluate_policy_run
        except Exception as exc:
            st.error(f"Live Rollout 依赖导入失败：{exc}")
            return

        live_dir = PROJECT_ROOT / "replays" / "live" / f"{selected}_tmp"
        out_path = PROJECT_ROOT / "logs" / "eval" / "live" / f"{selected}_live_evaluation.json"
        try:
            metrics = evaluate_policy_run(
                config=load_env_config(config_path),
                policy="ppo",
                episodes=1,
                run_name=f"{selected}_live",
                model_path=str(model_path),
                out_path=out_path,
                replay_dir=live_dir,
                seed=0,
            )
            candidates = [
                live_dir / "success_episode.json",
                live_dir / "failure_episode.json",
                live_dir / "best_partial_episode.json",
            ]
            source = next((path for path in candidates if path.exists()), None)
            if source is None:
                st.error("Live Rollout 运行完成，但未生成 replay JSON。")
                return
            target = PROJECT_ROOT / "replays" / "live" / f"{selected}_live_episode.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            st.success(f"Live replay 已保存：{target.relative_to(PROJECT_ROOT).as_posix()}")
            st.json({k: v for k, v in metrics.items() if k != "episodes_detail"})
        except Exception as exc:
            st.error(f"Live Rollout 加载或运行失败：{exc}")


def render_llm_planner(st: Any, replay_state: Optional[Dict[str, Any]]) -> None:
    st.header("区域 F：LLM Planner 显示区")
    if not replay_state:
        st.info("请选择 replay 后查看 planner 信息。")
        return
    step = replay_state.get("step", {})
    info = step.get("info") or {}
    planner_output = info.get("planner_output") or info.get("llm_plan") or info.get("planner_plan")
    if planner_output:
        st.json(planner_output)
    else:
        st.info("当前 replay 未记录 LLM Planner 输出；此阶段主要展示 PPO Executor 训练结果。")


def render_missing_summary(st: Any, index: Dict[str, Any]) -> None:
    missing = index.get("missing", {})
    missing_items = [name for name, value in missing.items() if value]
    if missing_items:
        st.warning(f"缺失项：{', '.join(missing_items)}")


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is not installed. Run `pip install -r requirements.txt`.") from exc

    st.set_page_config(page_title="NavAgent-PPO 自主巡检智能体训练可视化系统", layout="wide")
    st.title("NavAgent-PPO 自主巡检智能体训练可视化系统")

    if st.sidebar.button("重建 artifacts index"):
        write_index(build_index(PROJECT_ROOT), INDEX_PATH)
        st.sidebar.success("artifacts_index.json 已重建")

    index = load_index()
    render_missing_summary(st, index)
    comparison = read_comparison_table()

    tab_overview, tab_stage, tab_replay, tab_curve, tab_live, tab_llm = st.tabs(
        ["总览", "Stage 对比", "Episode Replay", "训练曲线", "Live Rollout", "LLM Planner"]
    )
    with tab_overview:
        render_overview(st, comparison)
    with tab_stage:
        render_stage_compare(st, index, comparison)
    with tab_replay:
        replay_state = render_replay(st, index)
    with tab_curve:
        render_training_curves(st, index)
    with tab_live:
        render_live_rollout(st, index)
    with tab_llm:
        render_llm_planner(st, locals().get("replay_state"))


if __name__ == "__main__":
    main()
