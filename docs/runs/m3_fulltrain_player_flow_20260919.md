# M3 全训练集短程试验（2026-09-19）

用户同意扩大到完整训练集，保持网络和新 loss，不继续只拟合 16 个片段。
这是全训练池上的 1000-step pilot，不是一个完整 epoch，也不代表全量训练已完成。

## 数据

- Canonical repaired release：
  `/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/polis_two_player_fixed_skins_complete_20260917_360p`。
- 来源：`derived/m3/validated/train_c65.pt`，SHA-256
  `679f5b1f28fa9056503c70cb9f24825c02edead21bc1c03ef1af622b3af134c4`。
- 新索引：`Plot/derived/m3_fulltrain_c9_20260919/train_c9.pt`。
- 所有有效 65 帧父窗口内，以 8 帧间隔枚举 9 帧子窗口，按 episode / start / camera 去重。
  不读取或覆盖原始视频、NPZ、reference；不越过已验证时间范围。
- 25,691 个索引 episode 中，25,229 个有有效窗口；共 2,294,147 个窗口。
  这不是重新验证所有原始 9 帧组合；原 65 帧索引没覆盖的范围不额外加入。
- 有效 episode：S01 13,106，S02 2,001，S07 2,085，S08 3,289，S09 3,653，S10 1,095。
- 从所有上述窗口 shuffle 抽样，不做新的 scenario / 人物比例重加权。
- 沿用 `derived/m3_player_short_20260919/val_id_c9.pt` 的 8 个验证窗口及原 8 个 probe
  （4 train + 4 val-ID），便于与上一轮直接比较。不是完整 val-ID 评测。

## 模型、优化与预算

- 从 `/root/rcz-runs/m3_multiclip16_player_flow_from5500_20260919/step_0006500.pt`
  用 `--resume` 恢复模型和 AdamW 状态，到 global 7500，追加 1000 次更新。
- 9 帧训练、1 条件 + 8 预测、cache 32；完整 M3 可训练，Pixel VAE 冻结。
- 保持 Pixel VAE 的同一 per-channel latent normalization。
- Loss 不变：`full_flow + player_flow + 0.1 * player_RGB_L1 + 0.025 * player_edge_L1`。
  人物 flow 独立归一化；全图 latent mask upweight 为 0；每步解码 2 帧算像素项。
- 基础学习率 `1e-5`，外观 encoder / ROI projector / joint patch 外观输入列 `5e-5`。
- 远端 `root@vr.turbo-ai.com:20470`，GPU 7，batch 1，无梯度积累。
  其他 GPU 已有计算任务；不停止或修改其他任务。
- 每 250 步保存 checkpoint、固定种子验证及 20-step Euler 纯噪声 rollout 视频；W&B online。
- 在 run 目录创建 `STOP` 文件，可在完成当前 optimizer step 后保存并结束。

## 路径与复现

索引转换：`scripts/shorten_m3_window_index.py`。
启动入口：`train_scripts/recipes/m3/train_m3_simple_fulltrain_player_flow.sh`。

远端 run：`/root/rcz-runs/m3_fulltrain_player_flow_from6500_c9_20260919`。
包含 `config.json`、`training.jsonl`、`validation.jsonl`、`experiment.log`、
`wandb_run.json`、`source_snapshot.tar.gz` 与 checkpoint。

共享结果：
`/public/0_DATA/2_Avatar/zhizhou_share/rcz/plotdemo/m3_fulltrain_player_flow_20260919`。

- `before_6500/`：上一轮 global 6500 的完全相同 probe / seed 生成结果。
- `visualizations/step_0006750/` 起：本轮固定样例生成结果。
- `train_index_summary.json`：实际训练池大小和来源。
- 启动后同步 `config.json` 和 `wandb_run.json`，便于本地定位实验。

索引转换去重、边界、target 保留及旧模型 / loss / optimizer / monitoring 回归，
合计 63 项 CPU 测试通过，shell recipe 语法检查通过。

判断依据是未参与训练的固定片段是否改善，并检查人物身份、轮廓、遮挡；
仅全图变清晰或训练 loss 下降不足以证明外观条件泛化。未开展配对 reference 因果检验。

## 启动记录

- torchrun PID `1037608`，训练 worker PID `1037710`。
- W&B：<https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/rxj7tpdg>。
- 已检查 train / 固定 val episode 无重叠、完整窗口的 camera target 均在有效范围、
  全训练集 / 验证集 / 原 checkpoint 的 item vocabulary 一致。
- 已实际训练到 global 6550，完整 M3 的 461,064,506 个参数可训练，resume 无词表扩展。
  6520、6530、6540、6550 均有非零人物 flow / RGB 项；6510 无人物监督项，
  属于全数据采样中空人物 mask 的正常情况。不能把空 mask batch 的 0 误认为拟合成功。
- 启动阶段峰值 reserved 显存约 11.06 GiB。尚未到本轮首次 6750 验证点，
  因此这里只确认稳定启动，不作泛化改善结论。

## 切换到八卡

用户要求改为 8 卡并增大 batch。单卡 run 通过 STOP marker 在 global 6744 保存
`step_0006744.pt`（模型 + optimizer），然后正常结束。第一次 6750 验证尚未执行。
后续 8 卡任务以该 checkpoint 续训，沿用 global 7500 的原定终点。

每卡 batch 2 的 8-rank 两步真实训练测试通过；同步测量第 2 步约 3.73–3.75 秒，
最大峰值 allocated 14.35 GiB、reserved 16.85 GiB。测试更新不写 checkpoint，
不并入正式训练。测试日志位于远端
`/root/rcz-runs/m3_fulltrain_8gpu_batch_tune_20260919/batch2/experiment.log`。

测试期间另一个任务增加了 GPU 显存占用：其他任务从每卡约 50,000 MiB 增至
约 100,000 MiB。因此继续测试每卡 batch 4，而非按最初余量使用更大 batch；
未终止或修改其他任务。
