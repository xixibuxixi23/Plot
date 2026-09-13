"""Full-data direct latent regression report, generated only after completion."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/m1_direct_full_s01_v1"


def main():
    completion = json.loads((OUT / "completion.json").read_text())
    full = json.loads((OUT / "full_fit_metrics.json").read_text())
    assert completion["count"] == full["count"] == 46200
    assert [r["index"] for r in full["samples"]] == list(range(46200))
    train = [json.loads(s) for s in (OUT / "epoch_loss.jsonl").read_text().splitlines()]
    fixed = [json.loads(p.read_text()) for p in sorted(OUT.glob("fixed_loss_epoch_*.json"))]
    audit = [json.loads(p.read_text()) for p in sorted(OUT.glob("audit_epoch_*.json"))]
    keys = ["surface_exact_recall", "visible_exact_precision", "gt_camera_half_block_hit"]
    labels = ["可见位置＋材质召回", "可见区域精确率", "GT 相机半格深度命中"]
    text = [
        "# 直接预测体素 latent：全量 S01 记忆训练",
        "",
        "**训练和全部窗口评估已完成。**",
        "",
        "输入两名玩家照片与真实相机；原 train/val_id/test_id 全部参与拟合，共 46,200 个目标居民首帧窗口。不是泛化评估。",
        "",
        "沿用固定三维 query、共享 MultiViewVoxelDiT 和冻结 PERSIST 编解码器，一次前向直接预测干净 latent；唯一训练目标为 latent MSE，无 mask 监督。",
        "",
        "从 256 窗口直接预测模型的 5,000 步权重初始化，重置 AdamW。lr=1e-4、weight decay=0，全局 batch 64，20 轮/14,440 次更新/924,000 次样本曝光。源模型还继承了此前全量 flow 训练，不能视为从零训练的对照。",
        "",
        "| 全量指标 | 宏平均 |",
        "|---|---:|",
    ]
    for key, label in zip(keys, labels):
        text.append(f"| {label} | {100 * full['mean'][key]:.2f}% |")
    text += [
        "",
        f"生成干净 latent MSE：{full['mean']['latent_mse']:.6f}。",
        f"三项几何指标同时 ≥90%：{sum(all(r[k] >= 0.9 for k in keys) for r in full['samples'])}/46,200 个窗口。",
        "",
        "## 训练轨迹",
        "",
        "| 轮次 | 全 batch 训练 MSE | 固定 128 窗口 MSE | 固定 32 窗口召回 |",
        "|---|---:|---:|---:|",
    ]
    fm = {r["epoch"]: r["mean"] for r in fixed}
    am = {r["epoch"]: r["mean"] for r in audit}
    for r in train:
        e = r["epoch"]
        text.append(f"| {e} | {r['loss']:.6f} | {fm[e]:.6f} | {100 * am[e][keys[0]]:.2f}% |")
    tree_path = OUT / "tree_full_metrics.json"
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
        text += [
            "",
            f"完整体素树干/树叶召回宏平均：{100 * tree['macro_recall']:.2f}%（{tree['windows_with_tree']} 个含树窗口，包括不可见部分，不是可见表面召回）。",
        ]
    text += [
        "",
        "## 结论",
        "",
        "本轮全量拟合未达到 90% 目标。虽然总体几何指标高于历史 flow 记录，但树木恢复仍很差；同场景可视化中大量树干与树冠缺失，不能据此认为照片对应问题已解决。",
        "",
        "训练 MSE 从第 1 轮 0.839 降到第 20 轮 0.728，仍在缓慢下降；当前不能断言模型永远无法拟合，也不能保证单纯延长训练会成功。256 场景结果证明小范围记忆可行，尚未证明全量有效。",
        "",
        "[与旧全量试验同场景的重建图](gallery/fitted_scene_02.png)",
        "",
        "## 核查与比较限制",
        "",
        "- 每轮全量覆盖；最终索引恰好为 0..46199，无重复无遗漏。",
        "- 直接推理输入固定零 latent 和固定 t=0，GT latent 仅作为监督或评分目标；无需随机采样。",
        "- 5 项回归测试通过，包括与 256 窗口直接预测入口一致、固定 query 和条件梯度、续训学习率、数据映射与覆盖等检查；Ruff 通过。",
        "- 复用已有缓存中同材质状态别名映射；少量 raw ID 127 未知单元仍在训练目标中。最终未知单元后审计见 unknown_cell_audit.json。",
        "- 256 窗口每个学习 1,250 次，本轮全量每个学习 20 次。不能用这两次结果单独区分规模与训练曝光量的影响。",
        "- 之前全量 flow 的召回/精确率/半格命中为 23.25%/16.30%/24.28%；初始化与累计训练经历不同，只能作为历史参考。",
        "",
        "产物：checkpoint_latest.pt、full_fit_metrics.json、tree_full_metrics.json、loss_curve.png、gallery/。",
    ]
    (OUT / "experiment_report.md").write_text("\n".join(text) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [r["epoch"] for r in train], [r["loss"] for r in train], label="All-batch training"
    )
    axes[0].plot([r["epoch"] for r in fixed], [r["mean"] for r in fixed], label="Fixed 128 windows")
    axes[0].set(ylabel="Clean latent MSE", xlabel="Epoch")
    for k, label in zip(keys, ["Recall", "Precision", "Half-block hit"]):
        axes[1].plot([r["epoch"] for r in audit], [100 * r["mean"][k] for r in audit], label=label)
    axes[1].set(ylabel="Fixed 32 fitted windows (%)", xlabel="Epoch", ylim=(0, 100))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", dpi=150)
    print("\n".join(text[:22]))


if __name__ == "__main__":
    main()
