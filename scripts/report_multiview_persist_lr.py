"""Compare matched continuation runs from the same full-data checkpoint."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/m1_multiview_persist_lr_comparison_v1'
RUNS = {'3e-5': ROOT / 'outputs/m1_multiview_persist_lr3e5_continue_v1',
        '1e-4': ROOT / 'outputs/m1_multiview_persist_lr1e4_control_v1'}


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    for name, path in RUNS.items():
        if not (path / 'training_complete.json').exists():
            raise RuntimeError(f'{name} has not completed')
        results[name] = dict(
            fixed=[json.loads(f.read_text()) for f in sorted(path.glob('fixed_loss_epoch_*.json'))],
            training=[json.loads(s) for s in (path / 'epoch_loss.jsonl').read_text().splitlines()],
            audits=[json.loads(f.read_text()) for f in sorted(path.glob('audit_epoch_*.json'))],
            manifest=json.loads((path / 'manifest.json').read_text()),
        )
    assert results['3e-5']['fixed'][0] == results['1e-4']['fixed'][0], 'Different starting probes'
    keys = ['surface_exact_recall', 'visible_exact_precision', 'gt_camera_half_block_hit']
    baseline_path = ROOT / 'outputs/m1_multiview_persist_gtcam_s01_all_v1/audit_epoch_020.json'
    baseline = json.loads(baseline_path.read_text())
    text = ['# 全量 PERSIST 降学习率续训对照', '',
            '两组从全量第 20 轮同一个 checkpoint 出发，保留 Adam 动量。各续训 5 轮（3,610 步），全部 46,200 窗口每轮覆盖一次；全局 batch 64。两组数据顺序、训练随机种子一致，仅学习率不同。', '',
            '固定 loss：128 个等间隔已拟合窗口，噪声种子 90210+全局索引，时间步 0.1/0.3/0.5/0.7/0.9。这个均匀时间网格的均值不等于训练的 sigmoid-normal 时间分布均值。', '',
            '重建指标：沿用原全量实验固定 32 个窗口及各 rank 的相同推理噪声，只是训练诊断，不是重新评估全部 46,200 个窗口。', '',
            '| 学习率 | 起始固定 loss | 最终固定 loss | 最后轮全体训练 loss | 召回 | 精确率 | 半格首命中 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        text.append(f"| {name} | {r['fixed'][0]['mean']:.6f} | {r['fixed'][-1]['mean']:.6f} | {r['training'][-1]['loss']:.6f} | " +
                    ' | '.join(f"{100*r['audits'][-1]['mean'][k]:.2f}%" for k in keys) + ' |')
    low = results['3e-5']['fixed']
    control = results['1e-4']['fixed']
    text += ['', f"固定 loss 相对起点下降：3e-5 为 {100*(1-low[-1]['mean']/low[0]['mean']):.2f}%，"
             f"1e-4 为 {100*(1-control[-1]['mean']/control[0]['mean']):.2f}%。",
             '', '本次低学习率的 loss 略低，但重建召回、精确率及深度命中没有一致改善，尚未解决场景对应问题。'
             '这是单一训练种子、5 轮续训的结果，没有多种子显著性结论；不能据此承诺长期训练达到 90%。',
             '', '起始第 20 轮同一 32 窗口重建分数（召回/精确率/半格首命中）：' +
             '/'.join(f"{100*baseline['mean'][k]:.2f}%" for k in keys) + '。', '',
             '## 固定时间步误差', '', '| 时间步 | 起始 | 3e-5 最终 | 1e-4 最终 |', '|---|---:|---:|---:|']
    for t, value in results['3e-5']['fixed'][0]['by_time'].items():
        text.append(f"| {t} | {value:.6f} | {results['3e-5']['fixed'][-1]['by_time'][t]:.6f} | {results['1e-4']['fixed'][-1]['by_time'][t]:.6f} |")
    text += ['', 't 越大越接近纯噪声；t 越小越接近真实体素。', '',
             '验证：学习率覆盖保留 Adam moments 的回归测试、原有数据映射/覆盖/采样测试共 4 项通过；加载后实际学习率记入各组 manifest 和 epoch_loss。', '',
             '两组 checkpoint_latest.pt 保留模型和优化器，原全量 checkpoint 未覆盖。曲线见 loss_comparison.png，逐轮原始数据见 comparison.json。']
    (OUT / 'comparison.json').write_text(json.dumps(results, indent=2))
    (OUT / 'experiment_report.md').write_text('\n'.join(text)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in results.items():
        axes[0].plot([x['epoch'] for x in r['fixed']], [x['mean'] for x in r['fixed']], marker='o', label=name)
        axes[1].plot([x['epoch'] for x in r['training']], [x['loss'] for x in r['training']], marker='o', label=name)
        axes[2].plot([20]+[x['epoch'] for x in r['audits']], [100*baseline['mean'][keys[0]]]+[100*x['mean'][keys[0]] for x in r['audits']], marker='o', label=name)
    for ax, title in zip(axes, ['Fixed 128-window flow MSE', 'All-batch epoch training MSE', 'Fixed 32-window visible recall (%)']):
        ax.set(title=title, xlabel='Epoch')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'loss_comparison.png', dpi=150)
    print('\n'.join(text))


if __name__ == '__main__':
    main()
