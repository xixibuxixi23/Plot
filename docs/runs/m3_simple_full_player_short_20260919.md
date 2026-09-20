# M3-Simple 人物全网络短片段微调

目标：让此前主要在单人场景预训练的 DiT 参与学习其他玩家的生成，先检查一个 8 帧 block 的纯噪声采样效果。

- 初始化：`m3_simple_player_joint_from1500_8gpu_b2_7bebdb0/step_0005000.pt`；重新建立 optimizer。
- 主干和场景/状态模块全部可训练，学习率 `1e-5`；VAE 冻结。
- 外观编码器、ROI projector、joint patch embedder 末 32 个外观输入通道使用 `5e-5`。
- 参数名和 checkpoint 模型结构保持兼容；输入通道的学习率通过缩放 AdamW 实际更新实现，而非缩放梯度。
- 每个样本 9 帧：1 帧干净前缀 + 8 帧未来，未来共享随机时间 `t ~ Uniform(0,1)`。
- Flow 权重：背景 1，人物内部最多 5（实例 mask 面积下采样，边界为连续权重）；全部样本启用。
- 总目标：区域加权 flow + `0.1 × player RGB L1 + 0.025 × player edge`；每样本解码 2 帧辅助监督。
- 数据：修复外观后的 `fixed_skins_20260917`，训练 16 个片段、val-ID 8 个片段，episode 不重叠。
- 人类、村民、僵尸、骷髅各有 4 个训练、2 个验证片段；未来每帧可见目标种类像素不少于 600，平均 2500–60000。
- 选择依据、每帧面积和输入 PNG 的 SHA256：`derived/m3_player_short_20260919/selection.json`。
- 初始化 checkpoint 是全数据训练得到的，val-ID 仅表示本轮微调不训练这些 episode，不声明相对所有历史实验都未见过。

## 本轮运行

机器：`vr.turbo-ai.com:20470`（zhizhou-avgen-js-public），共享 GPU 7，batch 1，无梯度积累。其他任务当时每卡约占 81 GiB，剩余约 59 GiB，因此本轮只占用一张卡的余量。

输出目录：`/root/rcz-runs/m3_simple_full_player_c9_from5000_20260919`。

训练额外 500 步，checkpoint 计数从 5000 到 5500；每 100 步验证，每 250 步保存和可视化。W&B online。

W&B：[ghdpl72e](https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/ghdpl72e)。启动时确认 M3 全部 `461,064,506` 个参数可训练，warm start 无缺失或迁移参数。

启动脚本：`train_scripts/recipes/m3/train_m3_simple_full_player_short.sh`。

纯噪声前后对照：`plotdemo/m3_simple_full_player_c9_20260919/{before,after}`（相对 rcz 工作区）。固定 8 个 probe（4 个训练、4 个验证），各自前后使用同一 seed、20 次去噪；只有第 0 帧 RGB 作为观察，未来 RGB 只用于评分。视频为 GT 左、生成右。播放速度 8 FPS。

日志：输出目录内 `experiment.log`；精确配置为 `config.json`；W&B 链接为 `wandb_run.json`；启动代码与子集快照为 `source_snapshot.tar.gz`。

短片段测试中的前缀可能已经包含玩家外观。因此短片段变清楚并不单独证明 reference 已被使用；后续还需固定噪声替换 reference，以及相同目标区间的真实历史/生成历史对照。

## 验证

57 项相关测试通过，覆盖完整参数更新、外观通道与独立 AdamW 参数的数值等价性、optimizer 恢复、短/长纯噪声可视化以及已有 renderer/人物监督测试。
