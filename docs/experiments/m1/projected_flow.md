# 显式投影 flow：1024窗口对照

已完成两组5,120步对照，未观察到稳定收益。原flow召回30.01%，投影组28.80%；
可见树干召回分别1.72%和1.56%。保留原flow主线。
完整结果见 [实验报告](../../../outputs/m1_projection_comparison1024_v1/experiment_report.md)。

原模型保持为基线。实验模型在 `experiments/m1/projected_flow.py`。
取6³三维patch中心（48³方块坐标3.5,11.5,...,43.5），转为cube-centred/48坐标，
用与现有camera_rays一致的ENU相机基向量投影到图像归一化坐标。
对冻结16通道图片VAE特征做bilinear grid_sample（align_corners=False），
画外、背后或缺失视图不参与平均。16→1024无偏置线性层零初始化，加到原3D token。
原全局交叉注意力、flow训练目标与20步推理均保留。不使用GT深度或mask监督。

新增16,384参数；中心单点采样且不处理遮挡，不能预先保证细树干受益。
两组共同从全量flow第20轮checkpoint初始化，重置AdamW，lr1e-4，batch64，
1024窗口各5120更新，每个窗口320次曝光。原source checkpoint已见过数据，属于记忆实验。

从 PLOT 仓库根目录运行（两组顺序执行）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_projected_flow_compare.py --mode projected --steps 5120 \
  --output outputs/m1_projected_flow1024_v1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_projected_flow_compare.py --mode flow --steps 5120 \
  --output outputs/m1_baseline_flow1024_v1
```

之后分别对输出运行 `scripts/inspect_projected_flow.py --output <目录>`，
用 `scripts/report_projected_flow.py` 合并报告。
报告输出 `outputs/m1_projection_comparison1024_v1/experiment_report.md`。

树干/树叶使用引擎node groups区分，按GT可见表面计算召回，按可见表面及自由空间计算精确率。
每次生成评估覆盖全部1024窗口；同时记录两个玩家视角的聚合半格首命中。
