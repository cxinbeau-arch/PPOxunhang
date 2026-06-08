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

def classify_run(run_name: str) -> tuple[int, str]:
    name = run_name.lower()

    if name.startswith("exp00"):
        return 0, "Baseline / 启发式对比"
    if "basic_ppo" in name:
        return 1, "PPO Basic"
    if "curriculum_stage1" in name or "stage1" in name:
        return 2, "PPO Stage 1"
    if "curriculum_stage2" in name or "stage2" in name:
        return 3, "PPO Stage 2"
    if "curriculum_stage3" in name or "stage3" in name:
        return 4, "PPO Stage 3"
    if name.startswith("seed"):
        return 5, "Multi-seed Replay"
    return 9, "Other"

def replay_priority(file_name: str) -> int:
    n = file_name.lower()
    if "success" in n:
        return 0
    if "best_partial" in n or "partial" in n:
        return 1
    if "failure_or_worst" in n or "worst" in n:
        return 2
    if "failure" in n:
        return 3
    return 9

replay_root = ROOT / "replays"
json_files = []

for p in sorted(replay_root.rglob("*.json")):
    name = p.name.lower()
    if "evaluation" in name or "summary" in name:
        continue

    # 只展示成功 episode，隐藏 failure / worst / partial / live 等其他 replay。
    if "success" in name:
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
        section_order, section_name = classify_run(run_name)
        items.append({
            "run_name": run_name,
            "replay_name": p.name,
            "rel_path": out_path.relative_to(DOCS),
            "section_order": section_order,
            "section_name": section_name,
            "replay_order": replay_priority(p.name),
        })
        print(f"[OK] {p} -> {out_path}")
    except Exception as e:
        failures.append((str(p), str(e)))
        print(f"[FAIL] {p}: {e}")

items.sort(key=lambda x: (
    x["section_order"],
    x["run_name"],
    x["replay_order"],
    x["replay_name"]
))

grouped = {}
for item in items:
    grouped.setdefault((item["section_order"], item["section_name"]), []).append(item)

sections_html = []
for (_, section_name), rows in grouped.items():
    row_html = []
    for r in rows:
        row_html.append(f"""
        <tr>
          <td>{html.escape(r["run_name"])}</td>
          <td>{html.escape(r["replay_name"])}</td>
          <td><a href="{html.escape(str(r["rel_path"]))}" target="_blank">打开回放</a></td>
        </tr>
        """)

    sections_html.append(f"""
    <section class="section-block">
      <h2>{html.escape(section_name)}</h2>
      <table>
        <thead>
          <tr>
            <th>run_name</th>
            <th>replay 文件</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </section>
    """)

failure_block = ""
if failures:
    failure_rows = "".join(
        f"<li><code>{html.escape(p)}</code>: {html.escape(err)}</li>"
        for p, err in failures
    )
    failure_block = f"""
    <section class="section-block">
      <h2>导出失败文件</h2>
      <ul>{failure_rows}</ul>
    </section>
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
    h1 {{
      margin-bottom: 24px;
      font-size: 38px;
    }}
    h2 {{
      margin: 0 0 14px 0;
      font-size: 24px;
      color: #0f172a;
    }}
    .section-block {{
      margin-bottom: 36px;
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
      background: #0f172a;
      color: white;
    }}
    a {{
      color: #2563eb;
      font-weight: 700;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    code {{
      background: #eef2ff;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>NavAgent-PPO 自主巡检智能体可视化</h1>

  {''.join(sections_html)}

  {failure_block}
</body>
</html>
"""

(DOCS / "index.html").write_text(index_html, encoding="utf-8")
(DOCS / ".nojekyll").write_text("", encoding="utf-8")

print("\\nGitHub Pages 静态站点生成完成")
print(f"入口文件: {DOCS / 'index.html'}")
print(f"成功导出: {len(items)} 个 replay")
print(f"失败: {len(failures)} 个")
