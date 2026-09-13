# PERSIST 全量降学习率续训

从全量第 20 轮、14,440 步的同一个 checkpoint 比较 3e-5 与原学习率 1e-4。
各续训 5 轮，保留 Adam moments。模型、数据、损失、全局 batch 64 都不变。
输出分别为 `outputs/m1_multiview_persist_lr3e5_continue_v1` 与
`outputs/m1_multiview_persist_lr1e4_control_v1`，原始权重不覆盖。

`--lr` 用于新建优化器；续训通过 `--resume-lr` 显式覆盖 checkpoint 学习率。
不提供 `--resume-lr` 时保留 checkpoint 中的学习率。实际值写入 manifest。

从 PLOT 仓库根目录运行低学习率组：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_persist_full.py --phase train \
  --resume outputs/m1_multiview_persist_gtcam_s01_all_v1/checkpoint_latest.pt \
  --resume-lr 3e-5 --epochs 25 --fixed-loss-samples 128 \
  --output outputs/m1_multiview_persist_lr3e5_continue_v1
```

对照组改为 `--resume-lr 1e-4` 以及对应的新输出目录，仍从同一个第 20 轮 checkpoint 开始。
两组使用相同种子（20260910+rank）及 DistributedSampler 的相同 epoch 序列。
这是匹配随机序列的对照，不是原始训练 RNG 的逐位续接。

- `epoch_loss.jsonl`：每轮全部 rank、全部 batch 按样本数加权的训练均值。
- `fixed_loss_epoch_*.json`：128 个固定拟合窗口，5 个固定时间步，独立局部噪声生成器。
  均匀时间网格均值不同于训练 sigmoid-normal 时间分布的均值，不能直接比较绝对值。
- `audit_epoch_*.json`：原实验固定 32 个拟合窗口、同样的 20 步采样与噪声。
  此轮不重复全部 46,200 个窗口的昂贵重建评估。
- `training_complete.json`：本组 5 轮训练和逐轮诊断均结束。

两组结束后运行 `.venv/bin/python scripts/report_multiview_persist_lr.py`，
生成 `outputs/m1_multiview_persist_lr_comparison_v1/experiment_report.md` 及曲线。
脚本会验证两组起始固定诊断完全相同；未完成时拒绝生成完成报告。
