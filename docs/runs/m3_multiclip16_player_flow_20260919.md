# M3 多片段人物 flow 小实验（2026-09-19）

单片段已可拟合，用户要求结束该项验证，进入多样本阶段。

## 实验设置

- 复用 `derived/m3_player_short_20260919` 的 16 个训练 / 8 个 val-ID 片段。
  各自覆盖 human、villager、zombie、skeleton，train 与 val-ID episode 不重叠。
- 从 `/root/rcz-runs/m3_simple_full_player_c9_from5000_20260919/step_0005500.pt`
  初始化，fresh optimizer，额外 1000 步，到 global 6500。
  不继承单片段过拟合权重；原 5500 已训练过这 16 个训练片段，val-ID 未用于梯度更新。
- 不改网络：M3 全参数可训练，Pixel VAE 冻结；主干 `1e-5`，外观路径 `5e-5`。
- GPU 7、batch 1、无梯度积累，9 帧 = 1 条件帧 + 8 预测帧。
- `full_flow + player_flow + 0.1 * player_RGB + 0.025 * player_edge`。
  人物 flow 独立按 mask 面积归一化，原有全图 mask upweight 为 0。
- 每 100 步验证、生成固定噪声 8 个 probe 视频、保存 checkpoint。
  验证 8 个窗口，视频 probe 4 train + 4 val-ID，指标需分开看。
- W&B online，完整配置和源码快照保存在 run 目录。

本轮是扩大到 16 个固定片段的可学习性与初步泛化诊断，不是全数据集正式训练。
评判重点是固定噪声视频中人物是否恢复，以及 val-ID 是否改善；不以训练 loss 单独判定成功。
GT mask 只用于 loss/评分，不输入网络。没有配对反事实数据，不能据此断言外观可控。

## 入口与路径

入口：`train_scripts/recipes/m3/train_m3_simple_multiclip_player_flow.sh`。

远端 `root@vr.turbo-ai.com:20470` run：
`/root/rcz-runs/m3_multiclip16_player_flow_from5500_20260919`

共享结果：
`/public/0_DATA/2_Avatar/zhizhou_share/rcz/plotdemo/m3_multiclip16_player_flow_20260919`

- `before/`：同一 step 5500、相同 probe / seed / 20 Euler steps 的既有基线。
- `visualizations/step_0005600/` 起：本轮采样结果。
- `config.json`、`wandb_run.json`：启动后从远端同步的元数据。
- 远端 `training.jsonl`、`validation.jsonl`、`experiment.log`：训练与验证日志。
- 远端 `step_*.pt`：完整 checkpoint，不放进 Git。

需要提前结束时，在远端 run 目录创建 `STOP` 文件。训练完成当前 optimizer step 后
跳过当轮评测，先保存 checkpoint，再结束 W&B 和分布式进程；所有 rank 协调停止。
旧正式 recipe 的默认 loss 保持不变，新增 player-flow 权重默认 0。

## 启动信息

- torchrun PID `3864511`，训练 worker PID `3866126`。
- W&B：<https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/tthq3id8>。
- 已确认全部 `461064506` 个 M3 参数可训练，warm start 无 missing/迁移参数。
- 相关 CPU 测试 60 项通过，shell recipe 语法检查通过。
- 已实际完成 optimizer step 5510；新 `player_flow` 项正常计算，
  峰值 reserved 显存约 12.11 GiB。当前只确认稳定启动，尚未产生本轮验证结论。
