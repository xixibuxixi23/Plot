"""Matched high-noise training pilot and Euler step-count report."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/m1_noise_schedule_compare_v1"


def main():
    source = json.loads((OUT / "source.json").read_text())
    sampling = json.loads((OUT / "sampling_steps.json").read_text())
    results = {}
    for mode in ["original", "high_noise"]:
        p = OUT / mode
        complete = json.loads((p / "training_complete.json").read_text())
        results[mode] = dict(
            complete=complete,
            manifest=json.loads((p / "manifest.json").read_text()),
            eval=json.loads((p / "final_eval64.json").read_text()),
            fixed=[json.loads(f.read_text()) for f in sorted(p.glob("fixed_loss_epoch_*.json"))],
        )
    assert results["original"]["complete"] == results["high_noise"]["complete"]
    assert results["original"]["fixed"][0] == results["high_noise"]["fixed"][0]
    assert (
        results["original"]["eval"]["indices"]
        == results["high_noise"]["eval"]["indices"]
        == sampling["indices"]
    )
    keys = ["surface_exact_recall", "visible_exact_precision", "gt_camera_half_block_hit"]
    text = [
        "# 原flow：推理步数与高噪声训练对照",
        "",
        f"固定起点：step {source['step']}、epoch {source['epoch']}，完整模型和AdamW状态共同恢复。原百万步长训练保持运行。",
        "",
        "## 20/50/100步推理",
        "",
        "同一快照、同一64个已拟合窗口、同一噪声种子（1234+全局索引），只改变Euler步数。",
        "",
        "| 推理步数 | 方块召回 | 精确率 | 半格命中 |",
        "|---|---:|---:|---:|",
    ]
    for n, r in sampling["results"].items():
        text.append(f"| {n} | " + " | ".join(f"{100 * r['mean'][k]:.2f}%" for k in keys) + " |")
    text += [
        "",
        "## 训练分布对照",
        "",
        "原组：t=sigmoid(N(0,1))。高噪声组：每个样本以50%概率保留该t，否则替换为Uniform(0.8,1.0)。替换用独立局部随机数生成器，不扰动两组共享的体素噪声、基础时间步及数据顺序。",
        "",
        "两组全量46,200窗口各续训3轮/2,166步，batch64、lr1e-4，保留优化器动量，不改网络或loss。每轮固定128窗口/固定时间网格测loss、固定32窗口测几何；最后以下表的同一64窗口完整生成评分。仅短期单种子对照，不代表长训练或全量评估。",
        "",
        "| 模式 | 推理初值 | 召回 | 精确率 | 半格命中 |",
        "|---|---|---:|---:|---:|",
    ]
    for mode, r in results.items():
        for t, label in [("1.0", "纯噪声"), ("0.5", "带噪GT，仅诊断")]:
            m = r["eval"]["results"][t]["mean"]
            text.append(
                f"| {mode} | {label} | " + " | ".join(f"{100 * m[k]:.2f}%" for k in keys) + " |"
            )
    text += [
        "",
        "## 纯噪声生成的可见树木分项",
        "",
        "| 模式 | 类别 | 召回宏平均 | 精确率微平均 |",
        "|---|---|---:|---:|",
    ]
    for mode, r in results.items():
        for k, m in r["eval"]["results"]["1.0"]["tree"].items():
            text.append(
                f"| {mode} | {k} | {100 * m['macro_recall']:.2f}% | {100 * m['micro_precision']:.2f}% |"
            )
    text += [
        "",
        "## 固定时间网格的速度MSE",
        "",
        "| 时间t | 起点 | 原组最终 | 高噪声组最终 |",
        "|---|---:|---:|---:|",
    ]
    for t, v in results["original"]["fixed"][0]["by_time"].items():
        text.append(
            f"| {t} | {v:.6f} | {results['original']['fixed'][-1]['by_time'][t]:.6f} | {results['high_noise']['fixed'][-1]['by_time'][t]:.6f} |"
        )
    text += [
        "",
        "两组原始训练loss不能直接比较，因为采样的时间分布不同。固定诊断时间网格可以逐t比较；t=0.5重建使用真实结构初值，不能算照片重建率。",
        "",
        "验证：7项全量入口回归测试通过，包含混合分布范围/比例、固定种子确定性、全局RNG不被额外采样扰动。两组起点固定诊断相同，最终步骤和窗口索引一致。",
        "",
        "代码、快照、逐样本指标、原始预测和日志保留于此输出目录。主线训练配置不变。",
    ]
    text += [
        "",
        "## 本轮结论",
        "",
        "20→50→100步没有提高当前固定窗口重建率，因此保留20步。高噪声组相对相同预算的原分布对照，纯噪声召回提高约4.27个百分点，精确率约3.56个百分点，半格命中约3.01个百分点。相对训练起点的召回只提高约1.33个百分点，原分布组本轮有所回落，不能忽略这一点。",
        "",
        "可见树干召回2.25%→2.04%，树叶18.01%→17.27%，没有解决用户最关心的树木对应问题。t=0.5带噪GT还原86.57%→86.00%，仅小幅下降。高噪声固定速度误差略低，中低噪声误差略高。",
        "",
        "只能认为总体生成在本次短期对照中有小幅收益，不能称为重建突破。单种子、3轮、64窗口不足以推断长期效果或统计显著性；原百万步主线继续使用原分布，未替换权重或配置。",
    ]
    (OUT / "experiment_report.md").write_text("\n".join(text) + "\n")
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2))
    print("\n".join(text))


if __name__ == "__main__":
    main()
