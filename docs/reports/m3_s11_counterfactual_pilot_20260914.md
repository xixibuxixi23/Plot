# M3 S11 外观反事实小样本实验（2026-09-14）

## 结论

新增的 S11 固定轨迹数据可以有效检验并训练 M3 的玩家外观绑定。最终候选
checkpoint `step_0026900.pt` 在 3 条轨迹、4 个时间窗和每条轨迹的 4 个循环
外观版本上全部通过：48/48 个版本都会因玩家 reference 改变而在人物区域产生
明显变化，并且 48/48 个版本使用正确 reference 时都比循环错位 reference 更
接近真值。

这是一个小样本因果 overfit 成功，不是正式泛化结论。人物外观和主要颜色已经
受正确 reference 控制，但生成细节仍偏糊；后续应把相同的配对采样混入正式 M3
训练，并在未见轨迹和未见皮肤上复核。

## S11 数据设计

- 场景：`S11` / `appearance_counterfactual`。
- 每个 episode 有 4 个 player-shaped resident，固定在简单平台上，不发生位移。
- 轨迹由 6 个 24-frame 注视阶段组成；每位观察者依次观察另外 3 人两遍。
- 每条 base trajectory 生成 4 个 variant，保持动作、位置、相机和世界几何一致。
- 4 个视觉身份在 4 个 player slot 间循环置换，因此每个 slot 都会看到每种外观。
- 最终 pilot 为 3 条 base trajectory × 4 variant，共 12 个 episode；12/12 collection
  ledger success，12/12 validation usable。

数据位于：

- `../textagent/data/s11_counterfactual_distinct_g3_v4_20260914`
- 计划：`../textagent/data/plans/s11_counterfactual_distinct_g3_v4_20260914.jsonl`
- 采集审计：`../textagent/artifacts/s11_counterfactual_distinct_g3_v4_20260914/group_*/audit.json`

直接数据审计确认每组的动作、位置和相机轨迹精确一致，玩家在画面中有足够面积，
且真实 RGB 的人物 ROI 会随皮肤置换而明显变化。不同独立 Luanti 进程的原始 numeric
content ID 会发生偏移；配对 M3 adapter 因此复用 variant 0 的已编码 voxel condition，
但不修改原始录制数据。

## 训练捷径及修复

最初的反事实训练让同组 variant 共享噪声和 diffusion time，但普通
`x_t=(1-t)x_0+t\epsilon` 仍包含各 variant 自己的真实人物像素。模型可以从 noisy
target 直接恢复皮肤差异，而不读取 reference。该版本 loss 会下降，但 26825 的
reference shuffle 审计仍失败，确认了条件被忽略。

修复后，专用 counterfactual group 的未来帧统一使用 `t=1` 的同一纯噪声输入。
四个 sibling 的未来 noisy latent 逐元素相同，非外观条件也相同；不同目标只能由
对应的 `player_reference` 解释。回归测试直接断言 sibling future latents 相同。

网络结构未增加第二条外观分支：仍使用单次、ROI-masked、四视图 reference attention，
由 player 几何 ROI 把 reference token 路由到对应图像 token。

## 训练配置

- 机器：port 20470 节点，4 × NVIDIA H200（GPU 0--3）。
- 起点：`step_0026825.pt`；纯噪声 smoke 到 26827，正式强化到 26900 后主动停止。
- 每卡 batch：1 个完整 counterfactual group；每组 4 个 variant。
- 每步 effective view batch：4 cards × 1 group × 4 variants = 16 clips。
- 解冻：reference 模块和最后 8/12 个 DiT spatial blocks，共 154,061,216 / 461,335,888
  个可训练/总参数。
- reference LR：`1e-4`；解冻 backbone LR scale：0.25；BF16。
- 人物 latent upweight 8，player pixel L1 2，edge 0.5，counterfactual difference 4。
- 训练峰值 reserved memory：75.01 GiB；W&B disabled。
- 训练已停止，GPU 已释放。

候选 checkpoint：

`outputs/m3_s11_purenoise_focused4gpu_26827_26950_20260914/step_0026900.pt`

## 固定审计结果

审计在 starts 19/43/67/91、target observer 0、8-frame horizon、固定随机种子下，
对每个 variant 分别生成正确 reference 和循环错位 reference 的结果。

| 指标 | 结果 |
|---|---:|
| 通过的 trajectory-window 组合 | 12 / 12 |
| 对 reference 敏感的外观版本 | 48 / 48 |
| 正确 reference 优于错位 reference | 48 / 48 |
| 人物区 conditioning delta，最小 / 均值 | 0.0508 / 0.1023 |
| 错位减正确人物 L1，最小 / 均值 | 0.0117 / 0.0501 |
| 正确 reference 人物 L1 均值 | 0.1077 |
| 错位 reference 人物 L1 均值 | 0.1578 |

全部 12 份 JSON 位于：

`outputs/m3_s11_purenoise_focused4gpu_26827_26950_20260914/audit_step26900_group*_start*/audit.json`

另对 group 1 / start 67 做了 64-step denoising 复核，4/4 variant 继续通过；人物区
conditioning delta 为 0.1384--0.2566，正确 reference 的人物 L1 比错位 reference
低 0.0769--0.1064。

可视化：

`outputs/m3_s11_purenoise_focused4gpu_26827_26950_20260914/audit_step26900_group000001_start67_denoise64/all_variants_contact.jpg`

每行依次为 truth、correct reference、shuffled reference；同目录保留每个 variant
的 MP4 和单独 contact sheet。

## 验证

- Plot：116 tests passed，1 个已有 PyTorch Transformer warning。
- textagent S11/场景相关：56 tests passed。
- `git diff --check` 通过。

## 下一步

1. 保留 S11 的 group-aware sampler 和纯噪声反事实目标，但与普通 train+val 窗口混合，
   避免只记忆三条平台轨迹。
2. 新增未见 base trajectory 和未见 skin identity 的验证组；不要用训练组的 12 个
   episode 宣称泛化。
3. 分开报告 reference binding 与画质。当前实验解决了前者，后者仍需正式 M3 训练。
4. 正式训练继续保留正确/错位 reference paired audit，防止普通重建 loss 改善时再次
   忽略外观条件。

## 未见轨迹泛化与混合训练复核

后续另采 5 条完全未见 base trajectory、每条 4 个循环外观版本作为固定
`val_id`。扩大后的密封场地避免海洋 seed 淹没采集区域；20/20 episode usable。
审计窗口统一按每个 episode 的 `model_start_observation` 取相对 offsets
0/24/48/72，从而排除并行启动造成的初始化帧数差异。

旧的四视图 UI 贴图直接投影方案已排除：训练和以下审计均只使用统一的
ROI-masked reference attention，不启用 `--view-aware-appearance`。

| checkpoint / 配置 | 正确 reference 更优 | 严格通过窗口 | reference-sensitive | correct player L1 |
|---|---:|---:|---:|---:|
| 26900，未见轨迹基线 | 72 / 80 | 12 / 20 | 20 / 20 | 0.1577 |
| 26925，20% step、解冻 8 层 | 72 / 80 | 13 / 20 | 20 / 20 | 0.1542 |
| 26950，20% step、解冻 8 层 | 69 / 80 | 10 / 20 | 20 / 20 | 0.1671 |
| 26950，reference-only 温和版 | 75 / 80 | 15 / 20 | 20 / 20 | 0.1550 |
| 27000，reference-only 温和版 | 75 / 80 | 15 / 20 | 20 / 20 | 0.1538 |

激进版在 50 steps 已出现退化。温和版从相同 26900 起点只训练 reference
encoder/adapter（803,168 / 461,335,888 参数），LR `1e-5`，反事实差分权重
`0.1`；100 steps 后同时改善因果正确率、严格窗口数和人物 L1，因此选择
`step_0027000.pt` 作为正式续训起点。

正式训练等待 100 个完整 group（400 episodes）采集完毕后启动。由于每个 S11
item 会展开成 4 个 sibling clips，正式配置使用 5% S11 optimizer steps；在 8 卡
batch 1 下约为 17% 实际 S11 clips，而不是把 20% step 错当成 20% 样本。自动启动
recipe 为：

`train_scripts/recipes/m3/train_m3_mixed_s11_referenceonly_8gpu.sh`
