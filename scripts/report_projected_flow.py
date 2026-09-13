"""Summarize the completed matched flow projection trial."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/m1_projection_comparison1024_v1"


def main():
    runs = {}
    for mode, path in [
        ("projected", "m1_projected_flow1024_v1"),
        ("baseline", "m1_baseline_flow1024_v1"),
    ]:
        p = ROOT / "outputs" / path
        assert json.loads((p / "completion.json").read_text())["steps"] == 5120
        runs[mode] = dict(
            manifest=json.loads((p / "manifest.json").read_text()),
            audits=[json.loads(f.read_text()) for f in sorted(p.glob("audit_*.json"))],
            tree=json.loads((p / "visible_tree_metrics.json").read_text()),
        )
    for k in ["indices", "initialize", "steps", "lr", "batch", "seed"]:
        assert runs["projected"]["manifest"][k] == runs["baseline"]["manifest"][k], k
    # An exact common starting point is expected from the zero residual adapter.
    assert runs["projected"]["audits"][0]["mean"] == runs["baseline"]["audits"][0]["mean"]
    keys = [
        "surface_exact_recall",
        "visible_exact_precision",
        "gt_camera_half_block_hit",
        "latent_mse",
    ]
    text = [
        "# 显式投影连接：1024窗口 flow 对照",
        "",
        "相同全量 flow 第20轮权重初始化，重置 AdamW，lr=1e-4，global batch64，5120步，每窗口320次曝光。1024个等间隔已拟合窗口；无未知GT单元。两组训练顺序和噪声随机种子相同。",
        "",
        "保留原12层flow DiT、全局图片交叉注意力和velocity MSE。新增连接：216个三维patch中心投影到两张图片，双线性读取冻结的16通道图片latent；对有效视图求均值，以16→1024无偏置线性层加入初始三维token。新增16,384参数，零初始化。投影在训练前预计算，不使用GT深度或体素内容。",
        "",
        "timestep和噪声仍按原flow采样；推理仍20步Euler。不是mask监督。投影只判断画内/相机前方，不处理遮挡；每个8³方块区域只取一个中心，这是本试验的限制。",
        "",
        "| 模型 | 步数 | 可见召回 | 精确率 | 半格深度命中 | 生成latent MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, r in runs.items():
        for a in r["audits"]:
            m = a["mean"]
            text.append(
                f"| {mode} | {a['step']} | "
                + " | ".join(f"{100 * m[k]:.2f}%" for k in keys[:3])
                + f" | {m[keys[3]]:.6f} |"
            )
    text += [
        "",
        "## 树干与树叶（最终全部1024窗口）",
        "",
        "| 模型 | 类别 | 含可见目标窗口 | 可见召回宏平均 | 可见域精确率微平均 |",
        "|---|---|---:|---:|---:|",
    ]
    for mode, r in runs.items():
        for k, m in r["tree"]["mean"].items():
            text.append(
                f"| {mode} | {k} | {m['windows']} | {100 * m['macro_recall']:.2f}% | {100 * m['micro_precision']:.2f}% |"
            )
    text += [
        "",
        "精确率分母为GT可见表面及已知自由空间内预测的该类方块，正确要求raw node ID和位置一致；不声称完整树木拓扑正确。",
        "",
        "验证：新增投影4项测试和原模型2项回归测试共6项通过。覆盖射线与投影互逆（含竖直相机）、背向/画外点、玩家交换/无效视图、零初始化保持输出和新层梯度。两组起始审计相同，评估每次完整覆盖1024窗口，无重复遗漏。最终新增层16,384个权重均非零，范数1.648，见adapter_check.json。",
        "",
        "仅单种子、已有权重上的条件连接对照；不代表全量成绩。原flow模型及历史权重均保留。",
        "",
        "## 本轮结论",
        "",
        "没有观察到稳定收益，保持原flow为主线，不将投影版本替换为默认模型。最终召回和树干召回略低，精确率及树叶召回略高，幅度均不足以支持这次改动。两组均未充分拟合，结论只针对本预算下的中心单点采样、视角均值、一次残差注入，不等于否定所有显式几何连接。",
        "",
        "中心采样可能错过细树干、缺少遮挡判断可能引入不对应特征；这些是尚未验证的解释，不能当作已定位的根因。",
        "",
        "同场景图：[投影组](../m1_projected_flow1024_v1/gallery/fitted_scene_00.png) · [原flow](../m1_baseline_flow1024_v1/gallery/fitted_scene_00.png)。",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "experiment_report.md").write_text("\n".join(text) + "\n")
    (OUT / "comparison.json").write_text(json.dumps(runs, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for mode, r in runs.items():
        for ax, k in zip(axes, keys):
            ax.plot(
                [a["step"] for a in r["audits"]],
                [a["mean"][k] * 100 for a in r["audits"]],
                marker="o",
                label=mode,
            )
    for ax, k in zip(axes, keys):
        ax.set(title=k, xlabel="Updates")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    print("\n".join(text))


if __name__ == "__main__":
    main()
