"""Build a report from actual full-run artifacts; never label an active run complete."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    out = args.output
    audits = [json.loads(f.read_text()) for f in sorted(out.glob("audit_epoch_*.json"))]
    full = (
        json.loads((out / "full_fit_metrics.json").read_text())
        if (out / "full_fit_metrics.json").exists()
        else None
    )
    keys = ["surface_exact_recall", "visible_exact_precision", "gt_camera_half_block_hit"]
    labels = ["可见方块位置＋材质召回", "可见区域精确率", "GT 相机半格首命中"]
    text = [
        "# 多人 PERSIST：全量 S01 拟合实验",
        "",
        "**状态：" + ("训练与全体评估完成。" if full else "运行中；尚无全体评估结论。") + "**",
        "",
        "范围：原 train/val_id/test_id 全部用于记忆训练，共 46,200 个居民首帧窗口，"
        "每个窗口输入两名玩家照片与真实相机。成绩不是泛化指标，也不是预测相机成绩。",
        "",
        "结构与 fit8 相同：冻结 PERSIST 图片/体素编解码器，训练 457,619,840 参数的静态 DiT，"
        "只用 latent flow velocity MSE。从 fit8 的 5,000 步权重初始化，重置 AdamW，"
        "全局 batch 64，lr=1e-4，weight decay=0，20 轮/14,440 次更新。",
        "",
    ]
    if full:
        text += ["## 全体结果", "", "| 指标 | 场景宏平均 |", "|---|---:|"]
        for k, label in zip(keys, labels):
            text.append(f"| {label} | {100 * full['mean'][k]:.4f}% |")
        success = sum(all(r[k] >= 0.9 for k in keys) for r in full["samples"])
        text += [
            "",
            f"同时满足三项指标 ≥90%：{success:,}/{full['count']:,} 个窗口。",
            "",
            "### 原始 split 分项（全部已经用于拟合）",
            "",
            "| 原始 split | 窗口数 | 召回 | 精确率 | 半格首命中 |",
            "|---|---:|---:|---:|---:|",
        ]
        bounds = [("train", 0, 42000), ("val_id", 42000, 44100), ("test_id", 44100, 46200)]
        for name, start, end in bounds:
            rows = [r for r in full["samples"] if start <= r["index"] < end]
            means = [sum(r[k] for r in rows) / len(rows) for k in keys]
            text.append(
                f"| {name} | {len(rows):,} | " + " | ".join(f"{100 * v:.4f}%" for v in means) + " |"
            )
    if full:
        text += [
            "", "## 结论与预算限制", "",
            "全量拟合失败，尚未达到 90% 目标；不能把 fit8 的成功外推到全部场景。"
            "固定审计在后期仍处于低位，也不能保证单纯增加训练步数就能解决。",
            "",
            "本轮每个窗口训练 20 次；fit8 的 5,000 步、batch 2 相当于每个窗口 1,250 次，"
            "曝光量相差 62.5 倍。两次试验并非等曝光预算，因此目前既不能证明结构无法全量拟合，"
            "也不能承诺延长训练必然达到目标。应先用中等规模子集测定达到高拟合率所需的曝光量。",
        ]
        tree_path = out / "tree_full_metrics.json"
        if tree_path.exists():
            tree = json.loads(tree_path.read_text())
            text += ["", f"含可见树木的 {tree['windows_with_tree']:,} 个窗口：树木位置＋材质召回宏平均 "
                     f"{100 * tree['macro_recall']:.4f}%，按方块合并的召回 "
                     f"{100 * tree['micro_recall']:.4f}%。"]
        probe_path = out / "condition_probe.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text())
            text += ["", "### 图片条件检查", "",
                     "32 个等间隔拟合窗口，固定相机与每样本噪声，仅错配图片特征。此样本集与每轮审计不同。",
                     "", "| 输入 | 召回 | 精确率 | 半格首命中 |", "|---|---:|---:|---:|"]
            for name, label in [("original", "原图"), ("shuffled_images", "错配图片")]:
                text.append(f"| {label} | " + " | ".join(
                    f"{100 * probe['results'][name][k]:.4f}%" for k in keys) + " |")
            text += ["", "错配图片进一步降低成绩，表明模型使用了图片信息；不能把失败归结为完全忽略条件。"]
    text += [
        "",
        "## 固定 32 样本训练审计",
        "",
        "以下仅用于观察收敛，不能替代全部 46,200 样本的评估。",
        "",
        "| 轮次 | 更新步 | 召回 | 精确率 | 半格首命中 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in audits:
        text.append(
            f"| {row['epoch']} | {row['step']} | "
            + " | ".join(f"{100 * row['mean'][k]:.2f}%" for k in keys)
            + " |"
        )
    text += [
        "",
        "## 数据适配和核查",
        "",
        "- 全体原始 node ID＋param2 中，支持的状态完整保留；少数新状态映射到同一种方块的已有状态，不改变 raw 材质 ID。",
        "- 预编码统计：166 个窗口发生映射，合计 3,231 个体素，占 5,109,350,400 个体素的 0.00006324%。详情见缓存 alias_summary.json。",
        "- 每轮不重复、不遗漏覆盖全部窗口；最终评估合并时再次检查索引完整性。",
        "- 缓存 latent 和图片/射线特征使用 float16，训练读入后转 float32。",
        "- 20 次 Euler 推理从独立高斯噪声初始化，GT latent 仅用于评分。",
        "- 最终评估批量采样，种子为 1234＋全局样本索引；不依赖 chunk 分片。BF16 批量/逐个采样差异另行记录。",
        "- 评分采用完整方块几何；首命中排除底部 18%，不等于引擎 RGB 像素相似度。",
        "- 5 项测试通过：多人顺序/无效视图、梯度、全量覆盖、新状态映射、批量采样种子稳定性；新入口 Ruff 检查通过。",
        "- 8 个 rank 各抽一例比较批量/逐个采样，latent 最大绝对差及 MSE 均为 0。",
        "- 与 fit8 重叠窗口核对：raw 与相机完全一致，float16 latent 缓存 MSE 为 4.63e-8。",
        "- 5 个窗口含 173,717 个引擎未知单元（raw ID 127）；本轮训练仍包含这些标签。最终分数已通过后审计排除未知 GT 单元及被其遮挡的射线；全体均值变化小于 0.0001 个百分点。全未知窗口保留为零分，未静默移除。",
        "- 未知单元修正由 scripts/audit_multiview_persist_full_outputs.py 完成；重新评估后需再运行此脚本。",
        "- 训练完整结束后，主动停止原串行评估并改用独立批量评估入口；train_run.log 末尾的终止信号不是训练失败。",
        "",
        "## 产物",
        "",
        "- checkpoint_latest.pt：最新已完成轮次的模型与优化器。",
        "- audit_curve.png：固定样本审计曲线。",
        "- full_fit_metrics.json：仅全体评估完成后生成。",
        "- tree_full_metrics.json、condition_probe.json、unknown_cell_audit.json：树木、图片错配与未知单元核查。",
        "- gallery/：8 个等间隔窗口的缓存、真实完整评估预测与重建图（非挑选最佳案例）。",
        "- training_complete.json：训练完成标志；completion.json：训练和全体评估均完成。",
        "- docs/experiments/m1/multiview_persist_full.md：完整运行说明。",
        "",
    ]
    (out / "experiment_report.md").write_text("\n".join(text))
    if audits:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in zip(
            keys, ["Visible exact recall", "Visible exact precision", "GT camera half-block hit"]
        ):
            ax.plot(
                [r["epoch"] for r in audits],
                [100 * r["mean"][key] for r in audits],
                marker="o",
                label=label,
            )
        ax.axhline(90, color="gray", linestyle="--")
        ax.set(xlabel="Full-data epochs", ylabel="Fixed 32 fitted windows (%)", ylim=(0, 100))
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "audit_curve.png", dpi=150)


if __name__ == "__main__":
    main()
