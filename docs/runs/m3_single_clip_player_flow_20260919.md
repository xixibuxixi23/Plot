# M3 单片段人物过拟合诊断（2026-09-19）

本实验只回答：在可重建、清晰可见的同一个人物片段上，全网络是否能学会从纯噪声生成？
不是泛化实验，也不能仅凭拟合成功断言模型使用了外观 reference。

## 设置

- 固定 `derived/m3_player_short_20260919/selection.json` 的 train 第 10 项（从 0 开始）：
  正面、全身可见的僵尸；9 帧 = 1 帧干净条件 + 8 帧预测。
- 从上一轮全网络短片段实验 `step_0005500.pt` 初始化，重置 optimizer，额外训练 1000 步。
- 全部 M3 参数可训练，Pixel VAE 冻结；不改网络结构。
- 主干等学习率 `1e-5`，reference encoder、ROI projector 和 joint patch 的 appearance 输入列 `5e-5`。
- 单卡 GPU 7 / batch 1 / 不积累梯度；不修改远端其他任务。
- 只缓存固定样本及冻结 VAE latent，不缓存任何可训练条件编码结果。
- 每一步仍随机采样噪声和时间；一个未来 8 帧块共用时间，prefix 为 0。

## Loss

`L = full_frame_flow + 1.0 * independently_normalized_player_flow + 0.1 * player_RGB_L1 + 0.025 * player_edge_L1`

人物 flow 使用 GT instance mask 的 area-downsample 覆盖率，在每个有效样本的
人物区域内独立归一化，不再用全画面 `1 + 4 * mask` 的加权平均替代。
GT mask 只作为监督，不输入模型。RGB/edge 每步解码两张预测帧。

新增项默认权重为 0，因此旧训练调用保持原行为。本实验入口是
`scripts/overfit_m3_single_clip.py`，并非修改现有正式训练 recipe 的默认设置。

记录低 / 中 / 高噪声段 `[0,1/3), [1/3,2/3), [2/3,1]` 的人物 flow；
某个统计区间没有抽到相应噪声段时，不把缺失数据记为 0 loss。

## 检验

- 已先做 GT → VAE → RGB 检查：全图 L1 `0.028401`，人物 L1 `0.030231`。
  人物脸、衣服和轮廓在重建图中可辨认；这只是重建参照，不是模型误差的严格下界。
- 训练前、每 100 步、结束时，使用相同初始噪声、20 步 Euler 生成未来 8 帧。
  推理仅编码首帧；未来 GT 仅用于评分和画面并排显示。
- 保存 H.264 视频、最后一帧并排图和 GT mask 确定的固定人物 crop，上传 W&B online。
- 结束时额外生成不同噪声种子，以及只交换 reference 的相同噪声结果。
  交换 reference 没有配对 GT，仅用于敏感性诊断，不能用原 GT 误差评价交换后的质量。
- 人物 loss 归一化、空 mask、亚像素 mask、梯度范围、单次前向和旧接口回归测试通过；
  与 renderer/monitoring/identity/full-finetune 合计 60 项测试通过。

## 路径

远端 `root@vr.turbo-ai.com:20470`：

- Run：`/root/rcz-runs/m3_single_clip_overfit_20260919`
- 日志：`/root/rcz-runs/m3_single_clip_overfit_20260919.log`
- 源码快照：`/root/rcz-runs/m3_single_clip_overfit_20260919_source.tar.gz`
- 每 500 步保存完整 checkpoint，预计为 global 6000、6500。

共享结果：
`/public/0_DATA/2_Avatar/zhizhou_share/rcz/plotdemo/m3_single_clip_overfit_20260919`

`step_0000.*` 是训练前；`step_0100.*` 是额外训练 100 步，而非 global step 100。
W&B 地址写在 `wandb_run.json`。完整逐步记录是 `training.jsonl`，
固定噪声评测是 `evaluations.jsonl`；只有全部完成才产生 `completed.json`。

## 启动记录

- 远端训练 PID `3723104`。
- W&B：<https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/eyb43w8b>。
- 训练前纯噪声采样：人物 L1 `0.082449`，全图 L1 `0.046368`；
  脸部偏棕色，细节不足。与 VAE 重建对照说明仍有明显生成误差。
- 最初 10 步约 `1.65 s/step`，进程峰值已分配显存约 `9.78 GiB`；
  backbone、appearance、joint patch 三组均有非零梯度。
- 以上只确认实验稳定启动，不代表训练已成功拟合或能泛化。

## 额外 100 步初步结果

固定样本、相同初始噪声的真实 8 帧 rollout：

| 指标 | 训练前（global 5500） | 额外 100 步（global 5600） |
| --- | ---: | ---: |
| 人物 RGB L1 | 0.082449 | 0.031713 |
| 全图 RGB L1 | 0.046368 | 0.028692 |
| 人物局部梯度幅度 / GT | 0.721527 | 0.794240 |

人物 RGB L1 下降约 61.5%。查看 `step_0100_player.png`，僵尸脸从错误的
棕色恢复为绿色，五官可辨。与 VAE 人物 L1 0.030231 数值接近，但不等于
达到严格误差下界，也不意味着所有纹理细节恢复。

训练前 30 步与第 71–100 步比较，人物 flow 的 low/mid/high 均值分别从
`0.567/0.312/0.435` 降为 `0.291/0.155/0.191`。两区间噪声随机样本不同，
仅作为训练趋势旁证；固定噪声 rollout 才是主要对照。

这支持“该样本能够拟合”，尚不能区分首帧复制、记忆和 reference 使用，
也不能把提升全部归因于新增 loss（本次同时改为固定单片段）。1000 步预算仍在执行。

## 提前结束

用户认为单样本已经验证完成，要求进入下一阶段。本轮已停止，最后完整训练记录为
local 254 / global 5754；最后一次纯噪声可视化是 local 200 / global 5700，
人物 L1 `0.022592`，全图 L1 `0.024718`，人物梯度幅度比 `0.942382`。

本轮保存间隔是 500 步，停止时尚无新 checkpoint；日志、W&B、视频、图片保留。
没有执行原定终点的换噪声 / 换 reference 测试，不能声称验证过外观条件依赖。
后续多样本实验从原有 global 5500 开始，不使用这轮单片段记忆后的权重。
