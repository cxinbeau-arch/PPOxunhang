"""Export one replay JSON to a standalone interactive HTML viewer."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.replay_loader import load_replay


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    body { font-family: Arial, "Microsoft YaHei", sans-serif; margin: 20px; color: #172033; background: #f8fafc; }
    h1 { font-size: 22px; margin: 0 0 16px; }
    .layout { display: grid; grid-template-columns: minmax(320px, 620px) minmax(280px, 1fr); gap: 20px; align-items: start; }
    .panel { background: #fff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 14px; }
    canvas { width: 100%; max-width: 620px; aspect-ratio: 1; border: 1px solid #cbd5e1; background: #fff; display: block; }
    button { margin: 8px 4px 8px 0; padding: 7px 10px; border: 1px solid #94a3b8; background: #fff; border-radius: 6px; cursor: pointer; }
    input[type=range] { width: 100%; }
    pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; overflow: auto; max-height: 260px; }
    .metric { line-height: 1.8; }
    .bars { display: grid; gap: 8px; margin-top: 8px; }
    .bar-row { display: grid; grid-template-columns: 52px 1fr 56px; align-items: center; gap: 8px; }
    .bar-bg { height: 12px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 12px; background: #2563eb; }
    .warn { color: #b45309; background: #fffbeb; border: 1px solid #fde68a; padding: 8px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>__TITLE__</h1>
  <div class="layout">
    <div class="panel">
      <canvas id="grid" width="640" height="640"></canvas>
      <input id="slider" type="range" min="0" max="0" value="0">
      <div>
        <button id="prev">上一步</button>
        <button id="next">下一步</button>
        <button id="play">播放/暂停</button>
        <button id="reset">重置</button>
      </div>
      <label>播放间隔 ms <input id="speed" type="range" min="80" max="1200" value="300"></label>
    </div>
    <div class="panel">
      <div class="metric" id="status"></div>
      <h3>Action Probabilities</h3>
      <div id="probs"></div>
      <h3>Step Info</h3>
      <pre id="stepInfo"></pre>
      <h3>Summary</h3>
      <pre id="summary"></pre>
    </div>
  </div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const replay = JSON.parse(document.getElementById("payload").textContent);
    const steps = replay.steps || [];
    const canvas = document.getElementById("grid");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("slider");
    const statusEl = document.getElementById("status");
    const probsEl = document.getElementById("probs");
    const stepInfoEl = document.getElementById("stepInfo");
    const summaryEl = document.getElementById("summary");
    const speedEl = document.getElementById("speed");
    let timer = null;
    slider.max = Math.max(0, steps.length - 1);
    summaryEl.textContent = JSON.stringify(replay.summary || {}, null, 2);

    function drawCircle(x, y, cell, color, radiusFactor) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x * cell + cell / 2, y * cell + cell / 2, cell * radiusFactor, 0, Math.PI * 2);
      ctx.fill();
    }

    function draw(index) {
      const step = steps[index];
      if (!step) return;
      const gridSize = step.grid_size || [12, 12];
      const w = gridSize[0];
      const h = gridSize[1];
      const cell = Math.min(canvas.width / w, canvas.height / h);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          ctx.strokeRect(x * cell, y * cell, cell, cell);
        }
      }
      for (const pos of step.obstacles || []) {
        ctx.fillStyle = "#334155";
        ctx.fillRect(pos[0] * cell, pos[1] * cell, cell, cell);
      }
      const trajectory = steps.slice(0, index + 1).map(item => item.agent_pos).filter(Boolean);
      if (trajectory.length > 1) {
        ctx.strokeStyle = "#64748b";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(trajectory[0][0] * cell + cell / 2, trajectory[0][1] * cell + cell / 2);
        for (const pos of trajectory.slice(1)) ctx.lineTo(pos[0] * cell + cell / 2, pos[1] * cell + cell / 2);
        ctx.stroke();
      }
      const visited = new Set((step.visited_targets || []).map(pos => `${pos[0]},${pos[1]}`));
      for (const pos of step.targets || []) {
        drawCircle(pos[0], pos[1], cell, visited.has(`${pos[0]},${pos[1]}`) ? "#16a34a" : "#f97316", 0.28);
      }
      if (step.base_pos) {
        ctx.fillStyle = "#2563eb";
        ctx.fillRect(step.base_pos[0] * cell + 5, step.base_pos[1] * cell + 5, cell - 10, cell - 10);
      }
      if (step.agent_pos) drawCircle(step.agent_pos[0], step.agent_pos[1], cell, "#dc2626", 0.33);
      statusEl.innerHTML = [
        `Step: ${step.step} / ${steps.length - 1}`,
        `Agent: ${JSON.stringify(step.agent_pos)}`,
        `Action: ${step.action}`,
        `Reward: ${Number(step.reward || 0).toFixed(3)}`,
        `Cum reward: ${Number(step.cum_reward || 0).toFixed(3)}`,
        `Battery: ${Number(step.battery || 0).toFixed(1)}`,
        `Done: ${Boolean(step.done)}`
      ].join("<br>");
      renderProbs(step.action_probs);
      stepInfoEl.textContent = JSON.stringify(step, null, 2);
      slider.value = index;
    }

    function renderProbs(probs) {
      if (!probs) {
        probsEl.innerHTML = '<div class="warn">该 replay 未保存 action_probs</div>';
        return;
      }
      const rows = Object.entries(probs).map(([name, value]) => {
        const pct = Math.max(0, Math.min(1, Number(value || 0))) * 100;
        return `<div class="bar-row"><span>${name}</span><div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div><span>${pct.toFixed(1)}%</span></div>`;
      }).join("");
      probsEl.innerHTML = `<div class="bars">${rows}</div>`;
    }

    function setIndex(value) {
      const next = Math.max(0, Math.min(steps.length - 1, value));
      draw(next);
    }

    slider.addEventListener("input", () => setIndex(Number(slider.value)));
    document.getElementById("prev").onclick = () => setIndex(Number(slider.value) - 1);
    document.getElementById("next").onclick = () => setIndex(Number(slider.value) + 1);
    document.getElementById("reset").onclick = () => setIndex(0);
    document.getElementById("play").onclick = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        return;
      }
      timer = setInterval(() => {
        let next = Number(slider.value) + 1;
        if (next >= steps.length) next = 0;
        draw(next);
      }, Number(speedEl.value));
    };
    draw(0);
  </script>
</body>
</html>
"""


def export_html(replay_path: Path, out_path: Path) -> Path:
    replay = load_replay(replay_path)
    title = f"NavAgent Replay - {replay.get('run_name', replay_path.stem)} - {replay.get('episode_type', 'episode')}"
    payload = json.dumps(replay, ensure_ascii=False).replace("</", "<\\/")
    html_text = HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, help="Input replay JSON path.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_html(Path(args.replay), Path(args.out))
    print(out)


if __name__ == "__main__":
    main()
