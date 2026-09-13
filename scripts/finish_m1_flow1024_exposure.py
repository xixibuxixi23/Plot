"""Wait for the authorized 20k run, score saved trees and write its final report."""

import json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "outputs/m1_flow1024_exposure1250_v1"
launch = json.loads((out / "launch.json").read_text())
while not (out / "completion.json").exists():
    p = Path("/proc") / str(launch["pid"]) / "stat"
    if not p.exists() or p.read_text().split()[2] == "Z":
        (out / "monitor_error.txt").write_text(
            "Training exited before completion; inspect run.log. No final result claimed."
        )
        raise SystemExit(1)
    time.sleep(30)
subprocess.run(
    [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/inspect_projected_flow.py"),
        "--output",
        str(out),
    ],
    cwd=ROOT,
    check=True,
)
rows = [json.loads(p.read_text()) for p in sorted(out.glob("audit_*.json"))]
tree = json.loads((out / "visible_tree_metrics.json").read_text())
text = [
    "# 1024窗口充分训练结果",
    "",
    "原flow、batch64、20,000更新，每窗口额外曝光1250次。与此前1024实验同一第20轮初始化，重置AdamW；保留完整1024窗口评估。256实验的窗口集合和随机种子不同，不能视为严格因果配对。",
    "",
    "| 更新 | 每窗口曝光 | 可见召回 | 精确率 | 半格命中 |",
    "|---|---|---|---|---|",
]
for r in rows:
    m = r["mean"]
    text.append(
        f"| {r['step']} | {r['step'] / 16:g} | {m['surface_exact_recall']:.2%} | {m['visible_exact_precision']:.2%} | {m['gt_camera_half_block_hit']:.2%} |"
    )
text += [
    "",
    "树木分项（可见表面）：",
    json.dumps(tree["mean"], indent=2),
    "",
    "达到高召回支持延长训练预算；未达到则只能说明该配置与预算仍不足，不能单独证明架构容量上限。完整原始预测保留在rank目录。",
]
(out / "experiment_report.md").write_text("\n".join(text))
print("Completed full training, 1024-window evaluations, tree scoring and report.", flush=True)
