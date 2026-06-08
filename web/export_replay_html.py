"""Export a replay JSON file to a standalone HTML animation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    .layout {{ display: grid; grid-template-columns: minmax(320px, 560px) 1fr; gap: 24px; align-items: start; }}
    canvas {{ border: 1px solid #cbd5e1; background: #ffffff; width: 100%; max-width: 560px; aspect-ratio: 1; }}
    pre {{ background: #f8fafc; padding: 12px; overflow: auto; border: 1px solid #e2e8f0; }}
    input[type=range] {{ width: 100%; }}
    .metric {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="layout">
    <div>
      <canvas id="grid" width="560" height="560"></canvas>
      <input id="slider" type="range" min="0" max="0" value="0">
      <button id="play">Play</button>
      <button id="pause">Pause</button>
    </div>
    <div>
      <div class="metric" id="status"></div>
      <pre id="planner"></pre>
      <pre id="summary"></pre>
    </div>
  </div>
  <script>
    const replay = {payload};
    const frames = replay.frames || [];
    const canvas = document.getElementById("grid");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("slider");
    const statusEl = document.getElementById("status");
    const plannerEl = document.getElementById("planner");
    const summaryEl = document.getElementById("summary");
    slider.max = Math.max(0, frames.length - 1);
    let timer = null;

    function draw(index) {{
      const frame = frames[index];
      if (!frame) return;
      const [w, h] = frame.grid_size;
      const cell = Math.min(canvas.width / w, canvas.height / h);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#e2e8f0";
      for (let y = 0; y < h; y++) {{
        for (let x = 0; x < w; x++) {{
          ctx.strokeRect(x * cell, y * cell, cell, cell);
        }}
      }}
      for (const [x, y] of frame.obstacles || []) {{
        ctx.fillStyle = "#334155";
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }}
      const [bx, by] = frame.base_position;
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(bx * cell + 4, by * cell + 4, cell - 8, cell - 8);
      for (const target of frame.targets || []) {{
        const [x, y] = target.position;
        ctx.fillStyle = target.visited ? "#16a34a" : "#f97316";
        ctx.beginPath();
        ctx.arc(x * cell + cell / 2, y * cell + cell / 2, cell * 0.28, 0, Math.PI * 2);
        ctx.fill();
      }}
      const [sx, sy] = frame.llm_subgoal || frame.current_target || frame.base_position;
      ctx.strokeStyle = "#a855f7";
      ctx.lineWidth = 3;
      ctx.strokeRect(sx * cell + 8, sy * cell + 8, cell - 16, cell - 16);
      const [ax, ay] = frame.agent_position;
      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      ctx.arc(ax * cell + cell / 2, ay * cell + cell / 2, cell * 0.32, 0, Math.PI * 2);
      ctx.fill();
      statusEl.innerHTML = `Step ${{frame.step}}/${{frame.max_steps}}<br>Action: ${{frame.action_name}}<br>Reward: ${{frame.reward.toFixed(3)}} | Return: ${{frame.cumulative_reward.toFixed(3)}}<br>Battery: ${{frame.battery.toFixed(1)}}/${{frame.max_battery}}`;
      plannerEl.textContent = JSON.stringify(frame.planner_output || {{}}, null, 2);
      summaryEl.textContent = JSON.stringify(replay.summary || {{}}, null, 2);
      slider.value = index;
    }}

    slider.addEventListener("input", () => draw(Number(slider.value)));
    document.getElementById("play").onclick = () => {{
      if (timer) return;
      timer = setInterval(() => {{
        let next = Number(slider.value) + 1;
        if (next >= frames.length) next = 0;
        draw(next);
      }}, 300);
    }};
    document.getElementById("pause").onclick = () => {{
      clearInterval(timer);
      timer = null;
    }};
    draw(0);
  </script>
</body>
</html>
"""


def export_html(replay_path: Path, out_path: Path) -> Path:
    with replay_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    title = f"NavAgent Replay - {payload.get('run_name', replay_path.stem)}"
    html = HTML_TEMPLATE.format(title=title, payload=json.dumps(payload, ensure_ascii=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(export_html(Path(args.replay), Path(args.out)))


if __name__ == "__main__":
    main()
