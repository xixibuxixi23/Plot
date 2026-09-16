# M3 多 Block 人物参考注入 Canary（2026-09-15）

## 结论

已实现一个最小的多 block 注入版本，但当前不设为默认，也不切换正在运行的
正式训练。当前选择标准只看正确参考下最终画面的人物重建与清晰度，不再使用
correct/shuffled margin 作为结构晋级指标。第一轮 100-step canary 的正确-reference
人物 L1 没有改善；真实轨迹结果也不稳定，因此当前拒绝用它替换单次注入结构。

## 结构

原结构只在 patch embedding 后注入一次：

```text
人物四视图 -> reference encoder -> 单一 reference adapter -> 12 个 DiT blocks
```

canary 保留上述输入注入，并在第 3、7、11 号 DiT block 前重复调用同一个
reference adapter：

```text
patch embedding
  + reference adapter
  -> blocks 0..2
  + gate[0] * reference adapter
  -> blocks 3..6
  + gate[1] * reference adapter
  -> blocks 7..10
  + gate[2] * reference adapter
  -> block 11
```

- 只保留一个 encoder 和一个 adapter，没有为每层复制人物分支。
- 三个新增 gate 从 0 开始，因此加载旧 checkpoint 时输出逐位不变。
- gate 使用 `tanh` 限幅；每层重新以当前 hidden state 查询同一组 reference tokens。
- 默认 `unified_reference_reinject_blocks=()`，即仍是原来的单次注入。

训练入口可通过以下参数启用：

```bash
--unified-player-reference \
--unified-reference-reinject-blocks 3 7 11
```

## Canary 协议

- 起点：正式单次注入 checkpoint `step_0032750.pt`
- 训练：8 x H200，100 steps，reference-only warm start
- S11 采样概率：0.20
- 学习率：`1e-4`
- 训练后 gate（raw）：`[0.003144, 0.006047, 0.005395]`
- checkpoint：
  `outputs/m3_multiblock_reference_canary_32750_32850_20260915/step_0032850.pt`
- 峰值显存：约 80.01 GiB/GPU

## 初步结果

### S11 固定泛化集

| 指标 | 单次注入 step 32750 | 多 block step 32850 | 变化 |
|---|---:|---:|---:|
| correct 人物 L1（低） | 0.131183 | 0.131206 | 基本不变 |

正确-reference 人物 L1 从 `0.131183` 变为 `0.131206`，约变差 0.02%。在当前
目标下，多 block 版本没有收益。打乱参考条件的结果只保留为因果路径调试信息，
不再参与模型选择。

### 五类真实验证轨迹

相对单次注入 `step 32750`：

- 全帧 L1 宏平均：`0.058040 -> 0.059186`，变差约 2.0%。
- 有可见人物的 player L1 宏平均：`0.101207 -> 0.099235`，改善约 1.95%。
- construction/four-player 的人物区域改善约 8.36%/6.14%。
- three-resident combat 的全帧和人物区域分别变差约 7.92%/5.04%。
- 身份相似度宏平均上升，但 3 个可判别 probe 的严格身份排名仍为 2/3。

因此结果是混合的，不能据此替换正式网络。

## 验证与后续判据

- 单元测试覆盖：共享单 adapter、旧 checkpoint 兼容、零 gate 精确等价、gate
  梯度、非法 block 参数。
- 完整测试：119 passed。
- 完整模型 1-step smoke 已通过，checkpoint 可正常保存和加载。
- 如果以后重试，必须跑匹配的单次注入 100-step control；除 block 列表为空外，
  其他设置与 canary 完全相同。
- 模型选择以正确-reference player ROI L1 为主；人物 ROI 的边缘/高频误差和固定
  可视化只用于防止 L1 偏好平均化的模糊输出。shuffled margin 不参与晋级。
- 只有正确-reference player ROI L1 和人物清晰度都改善，并且不损害真实轨迹
  全帧质量时，才考虑用于后续正式训练。
