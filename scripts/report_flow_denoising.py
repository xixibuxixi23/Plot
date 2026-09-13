"""Report matched noisy-GT and image-only flow diagnostics."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/m1_flow_denoising_probe_v1"


def main():
    assert (OUT / "completion.json").exists()
    data = json.loads((OUT / "results.json").read_text())
    m = data["manifest"]
    results = data["results"]
    text = [
        "# Flow：带噪真值还原与照片条件诊断",
        "",
        f"固定快照：step {m['step']}，epoch {m['epoch']}；64个等间隔已拟合窗口。训练任务继续独立运行。",
        "",
        "相机、GT、噪声种子不变。错配图片仅对16通道RGB latent跨样本循环移位，保留每个目标窗口自己的相机射线和有效视图。",
        "",
        "t越大初始噪声越多：x_t=(1-t)z_GT+[1e-5+(1-1e-5)t]ε。t=1完全不含GT；t<1使用真实场景信息，只是特权诊断，不能当成图片重建成绩。",
        "",
        "各起点都沿按t缩放的原20步Euler时间表积分到0。相同起点的原图/错图共用噪声。不同t的任务难度和步长不同，不能单独由此证明训练失败原因。",
        "",
        "| 起点t | 图片 | 可见方块召回 | 精确率 | 半格命中 | 最终latent MSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    keys = ["surface_exact_recall", "visible_exact_precision", "gt_camera_half_block_hit"]
    for t in m["time_grid"]:
        for mode, label in [("original", "原图"), ("shuffled_images", "错配图")]:
            r = results[f"{mode}_t{t:.1f}"]["mean"]
            text.append(
                f"| {t} | {label} | "
                + " | ".join(f"{100 * r[k]:.2f}%" for k in keys)
                + f" | {r['latent_mse']:.6f} |"
            )
    text += [
        "",
        "## 不经模型直接解码（控制实验）",
        "",
        "| t | 可见方块召回 | 精确率 | 半格命中 |",
        "|---|---:|---:|---:|",
    ]
    for t in [0.0, 0.1, 0.5, 0.9]:
        r = results[f"no_model_t{t:.1f}"]["mean"]
        text.append(f"| {t} | " + " | ".join(f"{100 * r[k]:.2f}%" for k in keys) + " |")
    text += [
        "",
        "t=0为真实latent的codec上限；其余为同样加噪后直接解码，没有神经网络去噪步骤。",
        "",
        "## 固定时间步的单次速度误差",
        "",
        "| t | 原图velocity MSE | 错图velocity MSE | 原图单步干净latent估计MSE |",
        "|---|---:|---:|---:|",
    ]
    for t in m["time_grid"]:
        a = data["one_step"][f"original_t{t:.1f}"]
        b = data["one_step"][f"shuffled_images_t{t:.1f}"]
        text.append(
            f"| {t} | {a['velocity_mse']:.6f} | {b['velocity_mse']:.6f} | {a['clean_estimate_mse']:.6f} |"
        )
    text += [
        "",
        "单步干净估计使用训练路径的解析关系 z_est=(1-1e-5)x_t-[1e-5+(1-1e-5)t]v_pred，不是额外训练的头。低t时这个估计误差自然较小，不能和高t直接比较来宣称图片理解更好。",
        "",
        "GT从不用于t=1推理初值；带噪GT实验有意提供目标信息。错配图仍保留相机，因此成绩不能描述为完全无条件生成。单快照、单噪声种子族，不能替代全量评估或因果证明。",
        "",
        "完整逐样本指标、原始预测和snapshot已保存；源训练checkpoint未修改。",
    ]
    text += [
        "",
        "## 本次诊断结论",
        "",
        "模型具有实际去噪能力：t=0.5直接解码召回20.52%，经模型还原提高到86.24%；错配图片仍84.49%，说明这一起点主要依靠输入体素中的场景结构。t=0.1直接解码本就99.31%，不能用该低噪声成绩单独证明模型去噪强。",
        "",
        "图片条件并未被忽略：纯噪声正确图片召回39.19%，错配图片5.07%。但仅凭图片的生成显著弱于带结构初值的还原，薄弱环节集中在从高噪声建立对应场景并完成生成轨迹。",
        "",
        "这不证明模型存在故意走捷径，也未区分条件学习不足、容量/优化限制和多步误差累积。数据没有回答增加帧数或更换网络能否解决。下一步应围绕高噪声条件生成诊断，不能仅凭总loss下降判断重建改善。",
    ]
    (OUT / "experiment_report.md").write_text("\n".join(text) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, label in [("original", "Correct images"), ("shuffled_images", "Shuffled images")]:
        ax.plot(
            m["time_grid"],
            [100 * results[f"{mode}_t{t:.1f}"]["mean"][keys[0]] for t in m["time_grid"]],
            marker="o",
            label=label,
        )
    ax.plot(
        [0, 0.1, 0.5, 0.9],
        [100 * results[f"no_model_t{t:.1f}"]["mean"][keys[0]] for t in [0, 0.1, 0.5, 0.9]],
        marker="x",
        linestyle="--",
        label="Decode without denoising",
    )
    ax.set(
        xlabel="Starting noise time (1 = no GT signal)",
        ylabel="Visible exact recall (%)",
        ylim=(0, 100),
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "diagnostic_curve.png", dpi=150)
    print("\n".join(text))


if __name__ == "__main__":
    main()
