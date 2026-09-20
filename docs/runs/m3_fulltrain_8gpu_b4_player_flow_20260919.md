# M3 全量训练池改为 8 GPU × batch 4（2026-09-19）

用户明确要求 8 卡并增大 batch；保持模型、loss、学习率、全训练池和终点不变。

## 接续

- 单卡通过 STOP marker 在 global 6744 保存完整 checkpoint，然后正常退出。
- 从 `/root/rcz-runs/m3_fulltrain_player_flow_from6500_c9_20260919/step_0006744.pt`
  恢复 M3 和 AdamW 状态；最终 global 7500，不重置已训练步数或 optimizer。
- 远端 `root@vr.turbo-ai.com:20470`，CUDA_VISIBLE_DEVICES `0,1,2,3,4,5,6,7`。
- 每卡 batch 4，8 个 DDP rank，梯度积累 1：global/effective batch **32**。
  训练 context 9 = 1 条件帧 + 8 预测帧；M3 全参数训练，VAE 冻结。
- 不因 batch 改变而自动扩大 LR；基础 `1e-5`、appearance `5e-5`。
- Loss：`full_flow + independent_player_flow + 0.1 * player_RGB + 0.025 * player_edge`。
- 全量训练池仍为 25,229 个有效 episode、2,294,147 个窗口。
- 从 6744 到 7500 共 756 次更新，计划取样约 24,192 个训练窗口，仍不是一个 epoch。
- 每 250 global step 保存 / 验证 / 纯噪声生成，首次 6750；固定 8 个验证窗口与
  8 个可视化 probe 保持不变，W&B online。
- 验证的随机种子改为 `seed + batch_number * world_size + rank`。
  当前 8 验证窗口 / 8 rank，每个 rank 仅一个窗口，保留原单卡 batch 1 的逐窗口种子。

## 8 卡 batch 实测

测试使用真实全量数据索引、同一 checkpoint、完整 loss 和 optimizer，W&B disabled。
测试更新不保存，也不并入正式训练。

| 每卡 batch | DDP rank | 测试步数 | 同步计时的最后一步 | 最大 peak allocated | 最大 peak reserved |
| --- | --- | --- | --- | --- | --- |
| 2 | 8/8 通过 | 2 | 3.75 s | 14.35 GiB | 16.85 GiB |
| 4 | 8/8 通过 | 3 | 4.91 s | 20.00 GiB | 20.09 GiB |

以上是短测试，不含首次加载 / 验证 / checkpoint 开销，不是长期速度保证。
测试过程中另一个 GPU 任务开始占用显存，其他进程合计约 100,000 MiB / 卡。
选择 batch 4 保留余量，不修改或停止其他任务。

测试目录：`/root/rcz-runs/m3_fulltrain_8gpu_batch_tune_20260919/batch{2,4}`。
日志 `PROFILE_STEP_TIMING` 记录所有 rank 的分阶段耗时与峰值显存。

## 路径

通用入口：`train_scripts/recipes/m3/train_m3_simple_fulltrain_player_flow.sh`，
设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、`BATCH_SIZE=4`、上述 `RESUME`，
`FINAL_STEP=7500` 和新 `OUTPUT_DIR`。

远端 run：`/root/rcz-runs/m3_fulltrain_8gpu_b4_player_flow_20260919`。
包含 `config.json`、`experiment.log`、`training.jsonl`、`validation.jsonl`、
`wandb_run.json`、`source_snapshot.tar.gz` 和 checkpoint。

共享结果：
`/public/0_DATA/2_Avatar/zhizhou_share/rcz/plotdemo/m3_fulltrain_8gpu_b4_player_flow_20260919`。

`before_6500/` 是原 6500 基线，不是切换八卡时 6744 的即时基线；6744 尚未生成视频。
`visualizations/` 是本轮新结果。需要停止时在远端 run 下创建 `STOP`，各 rank 协调，
保存当前完成的 optimizer step 后退出。

CPU renderer / loss / optimizer / monitoring / identity 测试共 60 项通过。

## 正式启动记录

- torchrun PID `1366994`，8 个训练 rank PID `1368600`–`1368607`。
- W&B：<https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/fhh63zsg>。
- 共享 `config.json` / `wandb_run.json` 已同步；`batch_tuning_summary.json` 记录测试实测值。
- 已实际执行到 global 6750，开始首轮固定验证和可视化；并非只完成初始化。

## 结束状态

用户要求试验纯人物像素 loss 后，本轮通过 STOP marker 在 **7365** 保存完整
`step_0007365.pt`（模型和 optimizer），然后正常结束并同步 W&B，未继续到原定 7500。
随后 pixels-only 与 flow + RGB10x 两轮都从该权重 warm-start，各自重置 optimizer，
保存在独立目录。详细结果分别见
`m3_fulltrain_player_pixels_only_20260919.md` 和
`m3_fulltrain_flow_rgb10x_unique_20260920.md`。
