# M3 人物外观强化计划

## 目标

让 M3 在 65 帧 rollout 中持续使用人物四视图外观条件，优先改善玩家皮肤、衣服和身体纹理的一致性。实验先证明“换错人物参考图会让结果明确变差”，再扩大训练步数。HUD、血量和通用锐化暂不混入这一轮，以免多个目标互相掩盖。

## 已确认的问题

26000 step 基线对正确、打乱和全零参考图不敏感。在三个固定 probe 上，打乱参考图后的玩家区域 L1 甚至略低于正确参考图；正确与打乱条件产生的输出差异只有约 0.0013–0.0020。这说明旧模型能接收四视图特征，但没有被迫用它判断人物身份。

完整审计结果在 `outputs/m3_reference_condition_audit_step26000/reference_condition_audit.json`。

## 已完成的代码

1. `plot/models/player_identity.py`：人物身份编码器、可微玩家区域裁剪和对比损失。
2. `plot/data/player_identity_dataset.py`：标准四视图与真实渲染人物 crop 的配对数据。
3. `scripts/train_player_identity.py` 和 `scripts/evaluate_player_identity.py`：身份编码器训练与独立验证。
4. `scripts/audit_m3_reference_condition.py`：正确、打乱、全零参考图的条件消融。
5. `plot/data/renderer_dataset.py`：为 M3 提供逐样本身份 mask、目标帧、身份槽位和有效标记。
6. `plot/training/renderer_trainer.py`：用冻结的身份编码器计算生成 crop 与正确参考的相似度，并加入错误皮肤的排序约束。
7. `train_scripts/train_renderer.py`：接入身份 checkpoint、身份损失、玩家区域动态 flow 加权，以及末端空间块的低学习率解冻。
8. `train_scripts/recipes/m3/train_m3_identity_supervision_8gpu.sh`：可复现的 8 卡 canary 配方。

代码测试结果为 84 passed。实现提交为 `33270b9`。

## 已完成的身份编码器

身份编码器先用标准四视图预训练，再用真实游戏渲染 crop 微调。当前使用的 checkpoint 是：

`outputs/player_identity_real_finetune_h100_5gpu_20260914/step_0003300.pt`

独立真实验证集共有 160 个样本：retrieval@1 为 73.12%，正确配对余弦相似度为 0.6846，最难错误配对为 0.6141。结果记录在 `outputs/player_identity_real_finetune_h100_5gpu_20260914/real_rendered_eval_step3300.json`。

## M3 训练目标

这一阶段只保留两类训练信号：

```text
L = L_flow + 0.1 * L_identity
L_identity = (1 - sim(correct))
           + 0.5 * relu(0.2 + sim(wrong) - sim(correct))
```

`L_flow` 以 50% 概率将玩家潜空间区域加权到 4 倍。`L_identity` 在解码后的玩家 crop 上约束人物身份，并要求正确皮肤的相似度高于错误皮肤。身份编码器保持冻结；人物参考模块、最后两个 spatial block 和输出层参与训练，预训练块使用主学习率的 0.1 倍。

## 分阶段执行

### 阶段 A：26500 step canary

从完整的 26000 step checkpoint 开始，只训练到 26500。每 500 step 保存、验证并生成可视化。先用短实验判断网络是否真正开始使用人物参考，避免直接消耗到 30000 step。

启动命令：

```bash
cd /data/huangyh/hxh/Plot
mkdir -p outputs/m3_identity_supervision_canary_from26000_to26500
tmux new-session -d -s m3_identity_supervision \
  'bash train_scripts/recipes/m3/train_m3_identity_supervision_8gpu.sh 2>&1 | tee outputs/m3_identity_supervision_canary_from26000_to26500/train.log'
```

### 阶段 B：固定评估

对 26500 checkpoint 运行相同的三个 65 帧 probe，并分别输入正确、打乱和全零四视图。进入长训需同时满足：

- 正确参考的玩家身份相似度至少比打乱参考高 0.03。
- 正确参考的玩家区域 L1 优于打乱参考；相对 26000 基线不能恶化超过 2%。
- 换参考图后，玩家区域的输出变化至少是非玩家区域的 2 倍。
- 65 帧全局 L1 或 PSNR 相对 26000 基线不能恶化超过 2%。
- 人工查看无损关键帧，确认提升来自皮肤和衣服身份，而非颜色偏移或过锐伪影。

### 阶段 C：继续到 30000 step

阶段 B 通过后，从 26500 checkpoint 恢复到 30000，并在 27000、28000、29000、30000 重复固定评估。如果身份差距连续两个保存点没有扩大，停止长训并只调整一个变量：先将身份损失从 0.1 提高到 0.2；仍无效时再将解冻 spatial block 从 2 增加到 4。

### 阶段 D：再处理其他画面问题

人物身份门槛通过后，才分别恢复 HUD/血量监督和长 rollout 稳定性训练。每次只加入一类目标，并保留正确/打乱参考消融，防止人物条件再次被其他 loss 淹没。

## 当前暂停点

2026-09-14 的试跑从 26000 运行到 26420 后按要求停止。该轮每 500 step 才保存，因此 26420 没有可恢复 checkpoint；当前最后一个完整 M3 checkpoint 仍是：

`outputs/m3_h100_8gpu_entity_reference_stage1_from25000_to27000_20260914/step_0026000.pt`

试跑日志在 `outputs/m3_h100_8gpu_identity_supervision_from26000_to30000_20260914/train.log`。训练峰值显存约为每卡 44.08 GiB，没有出现 OOM。
