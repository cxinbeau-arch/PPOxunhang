from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))


try:
    from replay_loader import load_replay as project_load_replay
except Exception:
    project_load_replay = None


st.set_page_config(
    page_title="NavAgent-PPO Episode Replay",
    layout="wide",
)


# =========================
# Replay loading
# =========================

def safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_xy(value: Any, default: Tuple[int, int] | None = None) -> Tuple[int, int] | None:
    if value is None:
        return default
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return int(value["x"]), int(value["y"])
        if "col" in value and "row" in value:
            return int(value["col"]), int(value["row"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return default


def as_xy_list(value: Any) -> List[Tuple[int, int]]:
    if value is None:
        return []
    out: List[Tuple[int, int]] = []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            p = as_xy(item)
            if p is not None:
                out.append(p)
    return out


def get_step_list(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(raw.get("steps"), list):
        return raw["steps"]
    if isinstance(raw.get("trajectory"), list):
        return raw["trajectory"]
    if isinstance(raw.get("episode"), dict) and isinstance(raw["episode"].get("steps"), list):
        return raw["episode"]["steps"]
    return []


def normalize_replay_fallback(raw: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    steps = get_step_list(raw)

    summary = raw.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    # Try to infer global fields from first step.
    first = steps[0] if steps else {}
    first_info = first.get("info", {}) if isinstance(first.get("info"), dict) else {}

    grid_size = (
        summary.get("grid_size")
        or raw.get("grid_size")
        or first_info.get("grid_size")
        or [10, 10]
    )
    if isinstance(grid_size, (list, tuple)) and len(grid_size) >= 2:
        width, height = int(grid_size[0]), int(grid_size[1])
    else:
        width, height = 10, 10

    normalized_steps = []
    cum_reward = 0.0

    for i, step in enumerate(steps):
        info = step.get("info", {}) if isinstance(step.get("info"), dict) else {}

        agent_pos = (
            as_xy(step.get("agent_pos"))
            or as_xy(step.get("agent"))
            or as_xy(step.get("pos"))
            or as_xy(step.get("position"))
            or as_xy(info.get("agent_pos"))
            or (0, 0)
        )

        base_pos = (
            as_xy(step.get("base_pos"))
            or as_xy(step.get("base"))
            or as_xy(raw.get("base_pos"))
            or as_xy(summary.get("base_pos"))
            or (0, 0)
        )

        targets = (
            as_xy_list(step.get("targets"))
            or as_xy_list(step.get("all_targets"))
            or as_xy_list(raw.get("targets"))
            or as_xy_list(summary.get("targets"))
        )

        visited_targets = (
            as_xy_list(step.get("visited_targets"))
            or as_xy_list(step.get("visited"))
            or as_xy_list(info.get("visited_targets"))
        )

        obstacles = (
            as_xy_list(step.get("obstacles"))
            or as_xy_list(raw.get("obstacles"))
            or as_xy_list(summary.get("obstacles"))
            or as_xy_list(info.get("obstacles"))
        )

        reward = float(step.get("reward", 0.0) or 0.0)
        if "cum_reward" in step:
            cum_reward = float(step.get("cum_reward") or 0.0)
        else:
            cum_reward += reward

        action_probs = (
            step.get("action_probs")
            or step.get("policy_distribution")
            or step.get("action_probabilities")
        )
        if not isinstance(action_probs, dict):
            action_probs = None

        normalized_steps.append(
            {
                "step": int(step.get("step", i)),
                "agent_pos": agent_pos,
                "base_pos": base_pos,
                "targets": targets,
                "visited_targets": visited_targets,
                "obstacles": obstacles,
                "battery": step.get("battery", info.get("battery")),
                "action": step.get("action", "start" if i == 0 else None),
                "reward": reward,
                "cum_reward": cum_reward,
                "done": bool(step.get("done", False)),
                "info": info,
                "action_probs": action_probs,
            }
        )

    return {
        "run_name": raw.get("run_name", source_path.parent.name),
        "episode_type": raw.get("episode_type", source_path.stem),
        "grid_size": [width, height],
        "steps": normalized_steps,
        "summary": summary,
        "source_path": str(source_path),
    }


def load_replay(path: Path) -> Dict[str, Any]:
    if project_load_replay is not None:
        try:
            loaded = project_load_replay(path)
            if isinstance(loaded, dict) and "steps" in loaded:
                # Ensure source_path exists for display.
                loaded.setdefault("source_path", str(path))
                if "grid_size" not in loaded:
                    first = loaded["steps"][0] if loaded["steps"] else {}
                    info = first.get("info", {}) if isinstance(first.get("info"), dict) else {}
                    loaded["grid_size"] = info.get("grid_size", [10, 10])
                return loaded
        except Exception as e:
            st.warning(f"项目 replay_loader 读取失败，已使用兼容解析器。错误：{e}")

    raw = safe_read_json(path)
    return normalize_replay_fallback(raw, path)


# =========================
# File discovery
# =========================

def find_replay_files() -> List[Path]:
    replay_root = ROOT / "replays"
    if not replay_root.exists():
        return []
    files = sorted(replay_root.rglob("*.json"))

    # Keep likely episode replay files, exclude summaries / index-like files.
    keep = []
    for p in files:
        name = p.name.lower()
        if any(x in name for x in ["success", "failure", "worst", "partial", "episode", "live"]):
            if "summary" not in name and "evaluation" not in name:
                keep.append(p)
    return keep


def infer_run_name(path: Path) -> str:
    try:
        rel = path.relative_to(ROOT / "replays")
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return path.parent.name


def infer_episode_type(path: Path) -> str:
    name = path.name.lower()
    if "success" in name:
        return "success"
    if "failure" in name:
        return "failure"
    if "worst" in name:
        return "worst"
    if "partial" in name:
        return "partial"
    if "live" in name:
        return "live"
    return "unknown"


def build_options(replay_files: List[Path]) -> pd.DataFrame:
    rows = []
    for p in replay_files:
        rows.append(
            {
                "run_name": infer_run_name(p),
                "episode_type": infer_episode_type(p),
                "file_name": p.name,
                "path": str(p),
            }
        )
    return pd.DataFrame(rows)


# =========================
# Plotting
# =========================

def get_grid_size(replay: Dict[str, Any], step: Dict[str, Any]) -> Tuple[int, int]:
    size = replay.get("grid_size")
    if not size:
        info = step.get("info", {}) if isinstance(step.get("info"), dict) else {}
        size = info.get("grid_size", [10, 10])
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return 10, 10


def draw_replay(replay: Dict[str, Any], step_idx: int) -> go.Figure:
    steps = replay["steps"]
    step = steps[step_idx]
    width, height = get_grid_size(replay, step)

    agent = step["agent_pos"]
    base = step["base_pos"]
    targets = step.get("targets", [])
    visited = set(step.get("visited_targets", []))
    obstacles = step.get("obstacles", [])

    path_points = [s["agent_pos"] for s in steps[: step_idx + 1] if s.get("agent_pos") is not None]

    fig = go.Figure()

    if path_points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in path_points],
                y=[p[1] for p in path_points],
                mode="lines+markers",
                name="trajectory",
                line=dict(width=3),
                marker=dict(size=5),
            )
        )

    if obstacles:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in obstacles],
                y=[p[1] for p in obstacles],
                mode="markers",
                name="obstacle",
                marker=dict(symbol="square", size=24, color="#334155"),
            )
        )

    unvisited_targets = [p for p in targets if tuple(p) not in visited]
    visited_targets = [p for p in targets if tuple(p) in visited]

    if unvisited_targets:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in unvisited_targets],
                y=[p[1] for p in unvisited_targets],
                mode="markers",
                name="unvisited target",
                marker=dict(symbol="diamond", size=20, color="#f97316"),
            )
        )

    if visited_targets:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in visited_targets],
                y=[p[1] for p in visited_targets],
                mode="markers",
                name="visited target",
                marker=dict(symbol="diamond", size=20, color="#22c55e"),
            )
        )

    if base is not None:
        fig.add_trace(
            go.Scatter(
                x=[base[0]],
                y=[base[1]],
                mode="markers",
                name="base",
                marker=dict(symbol="square", size=24, color="#2563eb"),
            )
        )

    if agent is not None:
        fig.add_trace(
            go.Scatter(
                x=[agent[0]],
                y=[agent[1]],
                mode="markers",
                name="agent",
                marker=dict(symbol="circle", size=24, color="#dc2626"),
            )
        )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.12),
        xaxis=dict(
            range=[-0.5, width - 0.5],
            dtick=1,
            showgrid=True,
            zeroline=False,
            title="x",
        ),
        yaxis=dict(
            range=[height - 0.5, -0.5],
            dtick=1,
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            title="y",
        ),
    )
    return fig


def show_action_probs(step: Dict[str, Any]) -> None:
    probs = step.get("action_probs")
    action = step.get("action")

    if action == "start":
        st.info("初始状态，无动作概率。")
        return

    if not isinstance(probs, dict) or not probs:
        st.warning("该 step 未保存 action_probs。")
        return

    df = pd.DataFrame(
        {
            "action": list(probs.keys()),
            "probability": [float(v) for v in probs.values()],
        }
    )
    st.bar_chart(df.set_index("action"))


def cell_info(step: Dict[str, Any], x: int, y: int) -> Dict[str, Any]:
    p = (x, y)
    targets = set(step.get("targets", []))
    visited = set(step.get("visited_targets", []))
    obstacles = set(step.get("obstacles", []))
    agent = step.get("agent_pos")
    base = step.get("base_pos")

    return {
        "coord": [x, y],
        "is_obstacle": p in obstacles,
        "is_target": p in targets,
        "is_visited_target": p in visited,
        "is_base": base is not None and p == tuple(base),
        "is_agent_current": agent is not None and p == tuple(agent),
    }


# =========================
# UI
# =========================

st.title("NavAgent-PPO Episode Replay")
st.caption("只展示真实 replay JSON。无训练曲线、无总览表、无 LLM 假输出。")

replay_files = find_replay_files()

if not replay_files:
    st.error("没有在 replays/ 目录下找到 replay JSON 文件。")
    st.stop()

options = build_options(replay_files)

with st.sidebar:
    st.header("Replay 选择")

    run_names = sorted(options["run_name"].unique().tolist())
    run_name = st.selectbox("run_name", run_names)

    filtered = options[options["run_name"] == run_name]

    episode_types = sorted(filtered["episode_type"].unique().tolist())
    episode_type = st.selectbox("episode 类型", episode_types)

    filtered = filtered[filtered["episode_type"] == episode_type]

    selected_path_str = st.selectbox(
        "replay 文件",
        filtered["path"].tolist(),
        format_func=lambda s: Path(s).name,
    )

selected_path = Path(selected_path_str)
replay = load_replay(selected_path)
steps = replay.get("steps", [])

if not steps:
    st.error(f"replay 文件中没有 steps：{selected_path}")
    st.stop()

if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

if st.session_state.step_idx >= len(steps):
    st.session_state.step_idx = len(steps) - 1

left, mid, right = st.columns([1.15, 2.0, 1.25])

with left:
    st.subheader("控制")

    st.write(f"**run_name:** `{run_name}`")
    st.write(f"**episode_type:** `{episode_type}`")
    st.write(f"**file:** `{selected_path.name}`")

    step_idx = st.slider(
        "step slider",
        min_value=0,
        max_value=len(steps) - 1,
        value=st.session_state.step_idx,
        key="slider_step",
    )
    st.session_state.step_idx = step_idx

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("上一步", use_container_width=True):
            st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
            st.rerun()
    with c2:
        if st.button("下一步", use_container_width=True):
            st.session_state.step_idx = min(len(steps) - 1, st.session_state.step_idx + 1)
            st.rerun()
    with c3:
        if st.button("重置", use_container_width=True):
            st.session_state.step_idx = 0
            st.rerun()

    st.divider()
    st.write("选择格子查看信息")
    current_step = steps[st.session_state.step_idx]
    width, height = get_grid_size(replay, current_step)

    cell_x = st.number_input("cell x", min_value=0, max_value=max(0, width - 1), value=int(current_step["agent_pos"][0]))
    cell_y = st.number_input("cell y", min_value=0, max_value=max(0, height - 1), value=int(current_step["agent_pos"][1]))

    st.json(cell_info(current_step, int(cell_x), int(cell_y)))

with mid:
    st.subheader("地图回放")
    fig = draw_replay(replay, st.session_state.step_idx)
    st.plotly_chart(fig, use_container_width=True)

with right:
    step = steps[st.session_state.step_idx]

    st.subheader("当前 Step")
    st.metric("step index", st.session_state.step_idx)
    st.metric("battery", step.get("battery", "NA"))
    st.metric("cum_reward", round(float(step.get("cum_reward", 0.0)), 3))

    st.write("**step data**")
    st.json(
        {
            "agent_pos": step.get("agent_pos"),
            "action": step.get("action"),
            "reward": step.get("reward"),
            "completed_targets": len(step.get("visited_targets", [])),
            "done": step.get("done"),
            "info": step.get("info", {}),
        }
    )

    st.subheader("动作概率")
    show_action_probs(step)
