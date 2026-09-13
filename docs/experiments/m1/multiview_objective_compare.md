# 256 场景直接预测与 flow 对照

入口 `experiments/m1/train_multiview_objective_compare.py`，仅支持本试验 8 卡、每卡 batch 8。
保持现有 MultiViewVoxelDiT 和冻结 codecs，通过固定零输入与固定 t=0 构造 direct 路径，
现有三维位置编码作为空间 query。时间模块为共享初始化而保留，但 direct 不采样时间或噪声。

两组共同从全量第 20 轮 checkpoint 初始化，AdamW 重新建立，lr=1e-4。每组 5,000 步，
256 个等间隔窗口，每个窗口学习 1,250 次。每四步恰好覆盖全部窗口一次。
这是 flow 预训练权重上的改造对照，不能声称是从零训练的公平方法排名。

从 PLOT 仓库根目录分别运行两组：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_objective_compare.py --mode direct \
  --output outputs/m1_objective_direct256_v1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_objective_compare.py --mode flow \
  --output outputs/m1_objective_flow256_v1
```

每 1,000 步保存最新 checkpoint 并完整评估 256 窗口。直接预测的推理只调用一次模型，
flow 为原有 20 步 Euler。两种训练 loss 不能直接比较；审计中的 latent_mse 均为生成
干净 latent 与 GT 的误差，可以比较。选定窗口没有 raw ID 127 未知体素。

两组完成后运行：

```bash
.venv/bin/python scripts/inspect_multiview_objective_outputs.py --mode direct
.venv/bin/python scripts/inspect_multiview_objective_outputs.py --mode flow
.venv/bin/python scripts/report_multiview_objective_compare.py
```

树木统计覆盖完整目标体素中的树干和树叶，包括不可见部分；它不是可见表面召回。
可视化使用各组 gallery/cache.pt 和实际预测，通过已有 visualize_multiview_voxel_dit.py 生成。
完整报告在 `outputs/m1_objective_compare256_v1/experiment_report.md`。
