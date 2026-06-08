from pathlib import Path
import subprocess
import sys
import re
import html

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPLAY_OUT = DOCS / "replays"

DOCS.mkdir(exist_ok=True)
REPLAY_OUT.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

replay_root = ROOT / "replays"
json_files = []

for p in sorted(replay_root.rglob("*.json")):
    name = p.name.lower()
    if "evaluation" in name or "summary" in name:
        continue
    if any(k in name for k in ["success", "failure", "worst", "partial", "episode", "live"]):
        json_files.append(p)

items = []
failures = []

for p in json_files:
    run_name = p.parent.name
    out_name = safe_name(f"{run_name}_{p.stem}.html")
    out_path = REPLAY_OUT / out_name

    cmd = [
        sys.executable,
        str(ROOT / "web" / "export_interactive_html.py"),
        "--replay",
        str(p),
        "--out",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        items.append((run_name, p.name, out_path.relative_to(DOCS)))
        print(f"[OK] {p} -> {out_path}")
    except Exception as e:
        failures.append((str(p), str(e)))
        print(f"[FAIL] {p}: {e}")

rows = []
for run_name, replay_name, rel_path in items:
    rows.append(
        f"""
        <tr>
          <td>{html.escape(run_name)}</td>
          <td>{html.escape(replay_name)}</td>
          <td><a href="{html.escape(str(rel_path))}" target="_blank">打开回放</a></td>
        </tr>
        """
    )

failure_block = ""
if failures:
    failure_rows = "".join(
        f"<li><code>{html.escape(p)}</code>: {html.escape(err)}</li>"
        for p, err in failures
    )
    failure_block = f"""
    <h2>导出失败文件</h2>
    <ul>{failure_rows}</ul>
    """

index_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>NavAgent-PPO 自主巡检智能体可视化</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      margin: 32px;
      background: #f6f7fb;
      color: #111827;
    }}
    h1 {{ margin-bottom: 8px; }}
    .note {{
      color: #475569;
      margin-bottom: 24px;
      line-height: 1.7;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #111827;
      color: white;
    }}
    a {{
      color: #2563eb;
      font-weight: 700;
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    code {{
      background: #eef2ff;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>NavAgent-PPO 自主巡检智能体可视化</h1>
  <div class="note">
    本页面展示 PPO 自主巡检智能体的真实 episode replay。点击“打开回放”即可查看对应阶段的轨迹、动作、奖励和电量变化。
  </div>

  <table>
    <thead>
      <tr>
        <th>run_name</th>
        <th>replay 文件</th>
        <th>操作</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  {failure_block}
</body>
</html>
"""

(DOCS / "index.html").write_text(index_html, encoding="utf-8")
(DOCS / ".nojekyll").write_text("", encoding="utf-8")

print("\nGitHub Pages 静态站点生成完成")
print(f"入口文件: {DOCS / 'index.html'}")
print(f"成功导出: {len(items)} 个 replay")
print(f"失败: {len(failures)} 个")
