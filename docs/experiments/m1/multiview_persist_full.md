# 多人 PERSIST：全量 S01 记忆训练

入口：`experiments/m1/train_multiview_persist_full.py`。
默认输出：`outputs/m1_multiview_persist_gtcam_s01_all_v1`。
默认缓存：`outputs/m1_multiview_persist_s01_cache`。

本轮已完成 20 轮训练及全部 46,200 个窗口评估：可见位置＋材质召回 23.25%，
精确率 16.30%，半格首命中 24.28%，含树窗口的树木召回宏平均 4.58%。
未达到 90% 拟合目标。详见 [完整实验报告](../../../outputs/m1_multiview_persist_gtcam_s01_all_v1/experiment_report.md)。
最终指标包含 `scripts/audit_multiview_persist_full_outputs.py` 对 5 个含未知单元窗口的修正；
重新运行评估后应再执行该后审计脚本。

降学习率的匹配续训对照见 [续训说明](multiview_persist_lr.md)：保留优化器状态，
显式设置 `--resume-lr`，并记录固定诊断 loss。

本轮明确采用记忆任务口径：原 train/val_id/test_id 的 42,000/2,100/2,100 个
居民首帧窗口全部拟合，总计 46,200；每个窗口输入两名玩家的图片及真实相机，
不是 held-out 测试。窗口保留原始 ENU 朝向，两名居民都分别作为目标窗口。

## 训练配置

- 静态多人 DiT：457,619,840 个参数，结构与 fit8 pilot 相同。
- 从 fit8 的 5,000 步模型初始化，重新建立 AdamW 优化器。
- 冻结 Pixel VAE、voxel Encoder、voxel Decoder，唯一损失为 flow velocity MSE。
- 8 GPU，每卡 batch 8，全局常规 batch 64；每轮末尾 batch 56。
- 20 轮，每轮 722 次更新，共 14,440 次全量训练更新，924,000 个样本曝光。
- 学习率 1e-4，weight decay 0，BF16，梯度裁剪 1.0。
- 每轮在固定 32 个已拟合样本上审计；最终另对全部 46,200 个窗口评分。
- 每轮原子保存模型和优化器，支持从已完成轮次恢复。恢复不会逐位重放噪声 RNG。

## 缓存与词表

特征以 mmap NPY 保存：每个窗口的体素 latent、两名玩家的图片/射线特征、
原始方块 ID 与 GT 相机。浮点特征缓存为 float16，训练时恢复 float32。
8 个预处理进程按唯一索引分工，写入并 flush 后才标记 ready，可从未完成样本继续。

保留 PERSIST 支持的全部 node ID＋param2 组合。全量数据出现少量不在现成词表中的
状态组合；仅对这些组合，映射到同一 raw node ID 的 param2=0（若不存在，则使用
该节点词表中的第一个状态）。绝不换材质或静默映射为空气；未知 raw node ID 会报错。
所有发生映射的样本、原始 node/param2 与数量均记录在 `aliases_rank_*.jsonl`。
重启可能产生重复日志，统计时按 sample index 去重。

## 运行

从 PLOT 仓库根目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_persist_full.py
```

默认顺序为预编码、20 轮训练、全体评估。可用 `--phase prepare/train/evaluate` 单独执行。
训练恢复时传 `--phase train --resume outputs/m1_multiview_persist_gtcam_s01_all_v1/checkpoint_latest.pt`，
恢复后需单独运行 `--phase evaluate` 完成全体评估。
若使用 `--phase all --resume ...`，则缓存检查、恢复训练和全体评估一起执行。

缓存中 `metadata.json` 校验数据根目录、样本数、split 数量、词表及归一化参数。
`ready.npy` 是逐样本写入状态；缓存的 `completion.json` 只表示预编码完成。
训练输出中的 `training_complete.json` 表示训练完成；输出中的 `completion.json`
只有在全体评估结束且索引完整性检查通过后才生成。

## 评估含义

保持与 pilot 相同的可见方块召回/精确率和 GT 相机半格首命中口径。
审计曲线只包含固定 32 个样本；正式全体指标必须读取 `full_fit_metrics.json`。
最终推理是 20 次 Euler 更新，从随机噪声生成 latent，不以 GT latent 作为初值。
全体评估使用批量采样，种子固定为 `1234 + 全局样本索引`，不依赖 chunk 分片。
也可以用 `scripts/evaluate_multiview_persist_full.py` 单独运行 8 卡评估；此入口额外记录
每卡一个样本的批量/逐个采样差异，检查 BF16 批量计算的影响。

训练集采样在 46,200 可被 8 整除时不需要补样本；最终评估按不重复 chunk 分片，
合并后检查索引恰为 0..46,199。测试覆盖了训练/评估无遗漏无重复，以及新状态映射
保留材质、已知状态不变、未知材质拒绝等约定。
