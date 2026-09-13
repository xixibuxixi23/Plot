# 直接预测：全量记忆训练

本轮训练和全体评估已完成：召回 42.72%、精确率 63.47%、半格深度命中 43.04%，
完整体素树干/树叶召回宏平均 6.38%。未达到 90% 目标。
见 [实验报告](../../../outputs/m1_direct_full_s01_v1/experiment_report.md)。

全量 46,200 个窗口，两张图片＋GT 相机。固定 query，一次前向直接回归干净体素 latent。
复用现有 256 窗口 direct 权重，重置 AdamW；lr=1e-4，global batch64，20epochs。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_persist_full.py --phase all --objective direct \
  --initialize outputs/m1_objective_direct256_v1/checkpoint_latest.pt \
  --epochs 20 --fixed-loss-samples 128 --output outputs/m1_direct_full_s01_v1
```

训练完成后由同一入口自动评估全部窗口，objective 写入 checkpoint，完整评估从
checkpoint 读取生成方式。直接预测不使用推理噪声，最终 full_fit_metrics 记录
objective=direct 及确定性推理协议。本轮启动时的旧通用 seed_protocol 字段在评估结束后已更正。

后审计和报告：

```bash
.venv/bin/python scripts/audit_multiview_persist_full_outputs.py --output outputs/m1_direct_full_s01_v1
.venv/bin/python scripts/report_direct_full.py
```

`epoch_loss.jsonl` 为全体 batch 按样本数加权均值；固定128窗口诊断和固定32窗口几何审计
每轮保存。最终全量评估后才有 completion.json。权重与所有日志在独立输出目录，原实验不覆盖。

恢复训练必须提供 `--objective direct --resume <checkpoint>`，并将 `--epochs` 设为目标总轮数。
`--resume-lr` 才会覆盖 checkpoint 学习率。此实验不属于从零训练或 held-out 泛化评估。
