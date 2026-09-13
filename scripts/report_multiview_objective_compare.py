"""Report matched direct/flow warm-start memorization from actual completed artifacts."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/m1_objective_compare256_v1"


def main():
    runs = {}
    for mode in ["direct", "flow"]:
        p = ROOT / f"outputs/m1_objective_{mode}256_v1"
        complete = json.loads((p / "completion.json").read_text())
        audits = [json.loads(f.read_text()) for f in sorted(p.glob("audit_*.json"))]
        assert audits[-1]["step"] == complete["steps"] and audits[-1]["count"] == 256
        runs[mode] = dict(
            manifest=json.loads((p / "manifest.json").read_text()),
            audits=audits,
            train=[json.loads(s) for s in (p / "train.jsonl").read_text().splitlines()],
        )
    for key in ["indices", "initialize", "steps", "lr", "batch", "exposures"]:
        assert runs["direct"]["manifest"][key] == runs["flow"]["manifest"][key], key
    keys = [
        "latent_mse",
        "surface_exact_recall",
        "visible_exact_precision",
        "gt_camera_half_block_hit",
    ]
    lines = [
        "# 256 窗口：直接预测与 flow 对照",
        "",
        "两组使用相同的 256 个等间隔全量窗口、GT 相机、两名玩家照片、冻结 PERSIST 编解码器、12 层 1024 宽 Transformer。",
        "",
        "共同从全量第 20 轮 flow checkpoint 初始化，重置 AdamW，lr=1e-4、weight decay=0、全局 batch 64、各 5,000 步，每个窗口 1,250 次曝光。每四步恰好覆盖 256 窗口一次，两组顺序相同。",
        "",
        "**这是已有 flow 权重的改造对照；源权重已经见过这些场景，不是从零训练或泛化实验。源初始化可能更有利于 flow，不能把结果概括为两类方法的普遍优劣。**",
        "",
        "- direct：零 latent 输入叠加原有三维位置编码作为固定 query，时间嵌入固定 t=0，直接回归干净 latent；一次前向推理。保留原有时间嵌入模块便于共享初始化，但不采样时间或噪声。",
        "- flow：原有 sigmoid-normal 时间采样与 velocity MSE；20 步 Euler 推理，从 seed=1234+全局索引的噪声出发。",
        "- 不加 mask 监督、不输入场景 ID，原全量权重不覆盖。",
        "",
        "## 可比的生成结果",
        "",
        "两组训练 loss 的目标不同，绝对值不应直接比较。下面的 latent MSE 都比较实际推理生成的干净 latent 与同一个 GT，几何评分覆盖全部 256 个已拟合窗口。",
        "",
        "| 模式 | 步数 | 干净 latent MSE | 可见位置＋材质召回 | 精确率 | 半格深度命中 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, r in runs.items():
        for a in r["audits"]:
            m = a["mean"]
            lines.append(
                f"| {mode} | {a['step']} | {m[keys[0]]:.6f} | "
                + " | ".join(f"{100 * m[k]:.2f}%" for k in keys[1:])
                + " |"
            )
    for mode, r in runs.items():
        final = r["audits"][-1]
        success = sum(all(s[k] >= 0.9 for k in keys[1:]) for s in final["samples"])
        lines += ["", f"{mode} 最终三项几何指标同时 ≥90% 的窗口：{success}/256。", ""]
    lines += [
        "",
        "## 本次结论",
        "",
        "两种目标最终都能在 256 窗口达到约 99.5% 以上召回，不能认为 flow 本身无法记忆。直接预测在已记录的审计点中，第 2,000 步三项均值已超过 90%，flow 在第 4,000 步超过；直接预测也只需一次前向。",
        "",
        "这支持优先扩大直接预测方案的拟合规模，但尚未验证全量。256 窗口每个学习 1,250 次，原全量仅 20 次；规模与曝光差异仍需验证。",
        "",
        "## 树木与可视化",
        "",
    ]
    for mode in runs:
        path = ROOT / f"outputs/m1_objective_{mode}256_v1/tree_metrics.json"
        if path.exists():
            tree = json.loads(path.read_text())
            runs[mode]["tree"] = tree
            lines.append(
                f"{mode}：{tree['count']} 个含树窗口，完整体素内树干/树叶精确召回宏平均 {100 * tree['macro_recall']:.2f}%。包括不可见体素，不等于可见表面召回。"
            )
        lines.append(
            f"图片与 GT/预测深度对照：[等间隔窗口示例](../m1_objective_{mode}256_v1/gallery/fitted_scene_03.png)。"
        )
    lines += [
        "",
        "## 验证与限制",
        "",
        "直接预测入口通过固定零 query/固定时间、重复输出一致和条件梯度检查。Ruff 通过。评估检查全部选定索引无遗漏无重复，所选窗口不含 raw ID 127 未知单元。",
        "",
        "每 1,000 步评估全部 256 个拟合窗口，最终权重、优化器、逐窗口指标和原始预测保存于各组目录。此实验不代表全量 46,200 窗口的重建成绩。",
        "",
        "没有多种子试验，也没有单独对每种目标搜索最佳超参数。小规模结果可用于决定下一步，不构成全量 90% 保证。",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "experiment_report.md").write_text("\n".join(lines) + "\n")
    (OUT / "comparison.json").write_text(json.dumps(runs, indent=2))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for mode, r in runs.items():
        for ax, key in zip(axes, keys):
            ax.plot(
                [a["step"] for a in r["audits"]],
                [a["mean"][key] * (1 if key == "latent_mse" else 100) for a in r["audits"]],
                marker="o",
                label=mode,
            )
    for ax, key in zip(axes, keys):
        ax.set(title=key, xlabel="Updates")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
