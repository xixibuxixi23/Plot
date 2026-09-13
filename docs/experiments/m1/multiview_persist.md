# M1：真实相机输入的多人 PERSIST 拟合实验

实现：`experiments/m1/multiview_voxel_dit.py`。
训练：`experiments/m1/train_multiview_voxel_dit.py`。

## 结构

每名玩家的 360×640 图片由同一个冻结的 PERSIST Pixel VAE 编码为
16×36×64 latent。逐像素拼接相机射线起点 xyz 与方向 xyz，得到 22 通道，
按 2×2 patch 编码为每人 576 个图片 token。所有人的 token 合并为同一个
交叉注意力上下文，没有玩家顺序编码或玩家间因果遮罩。

一个共享的 48×12×12×12 体素 latent 按 2³ patch 编码为 216 个三维 token。
DiT 有 12 层，每层为三维 self-attention/MLP 和图片 cross-attention/MLP，
宽度 1024、16 heads。输出 flow velocity；20 次 Euler 更新后，用冻结的
PERSIST 体素 Decoder 解码为 48³ 离散方块。

静态 DiT 的 457,619,840 个参数全部从 PERSIST-S 对应参数严格迁移。
时间注意力、动作及全局单相机调制模块被移除；相机信息通过每张图片的射线输入。
这是 PERSIST 的静态多人适配，不是原始完整 world-model 的原样推理。

体素 Encoder 和 Pixel VAE 只用于离线准备缓存，不参与反向传播；Decoder 也冻结。
训练只用 normalized latent 的 flow-velocity MSE，无实例 mask、可见性加权损失、
深度修正器或场景 ID 输入。评分会使用 GT 射线计算可见区域，这不是训练监督。

## 数据和相机约定

- 使用 S01 原始 `node ID + param2`，映射到 PERSIST 2,138 类，禁止丢失状态后直接补 0。
- 保留原始 ENU 朝向，不做 canonical yaw；相机都相对同一个目标 resident 体素窗口。
- 相机位置为 `(cam_pos - (voxel_center - 0.5))/48`，方向和 FOV 使用数据提供的真值。
- 输入图片通过已有 `m1_rgb_agent*.jpg` 缓存读取；不会把真实体素作为网络输入。
- 当前 pilot 从原 val_id 中等距选取 8 个 episode，每个只选 resident 0 的窗口，
  每个窗口同时使用 2 个玩家的图像。原 val_id 在本任务中已经允许拟合，成绩不是泛化成绩。
- 模型支持有效视图标记和任意输入人数；训练入口默认两人，`--num-views` 可选择其他人数，
  但只能选数据确实提供的人数。每个场景至少需要一名有效玩家。

## 启动

在 PLOT 仓库根目录中：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/m1/train_multiview_voxel_dit.py \
  --output outputs/m1_multiview_persist_gtcam_fit8_v1 --samples 8 --steps 1000

CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/m1/train_multiview_voxel_dit.py \
  --output outputs/m1_multiview_persist_gtcam_fit8_v1 --samples 8 --steps 5000 \
  --eval-every 500 --resume outputs/m1_multiview_persist_gtcam_fit8_v1/checkpoint_latest.pt
```

权重默认来自相邻 PERSIST 的 `data/checkpoints`；DiT 使用
`PERSIST-team/persist-voxel-denoiser-s` 的缓存/Hub 文件。
`cache.pt` 保存原始 GT、原始输入、预编码 latent 和相机信息。
`manifest.json` 保存权重路径、迁移数量与样本路径；`run_history.jsonl` 保存续训信息。
原子写入 `checkpoint_latest.pt`，内含模型和 AdamW 状态；新断点同时保存随机数状态。
初次 1,000 步断点产生时尚未保存随机数状态，因此这一次续训重置了采样 RNG，
模型及优化器状态仍连续，不应声称与不中断运行逐位等价。

## 验收

主指标是 GT 可见方块原始 ID 的精确召回、可见区域精确率和 GT 相机半格首命中。
前两项使用两名玩家可见区域的并集；首命中排除画面底部 18%。几何使用完整立方体，
这些数值不等于引擎 RGB 渲染相似度，也不证明未知相机输入可以达到同样成绩。

`scripts/evaluate_multiview_voxel_dit.py` 提供无需训练的检查：
只保留第一人的输入、替换其他场景的图片特征（相机不变）、同时替换图片/相机条件、
改用另一组采样噪声。本轮实际执行的是单人输入、只替换图片和新采样噪声三个对照。
只给一个人时仍用原来两名玩家的 GT 视野评分，保证评分目标不变。

`scripts/visualize_multiview_voxel_dit.py` 输出每名玩家的输入/GT深度/预测深度与拟合曲线。
树木评分仅作额外诊断，不参与训练或 checkpoint 选择。

本轮结果见 `outputs/m1_multiview_persist_gtcam_fit8_v1/experiment_report.md`。

5,000 步实验已完成：8 场景可见精确召回/精确率均为 99.7723%，GT 相机半格首命中
100%。4 个有树场景的完整窗口树木位置＋材质召回宏平均 99.6589%。这是固定小样本
记忆验证；未完成全量 S01 训练或相机预测接入。

用户随后授权了全量拟合，入口和本轮进度文件口径见
[全量多人 PERSIST 训练](multiview_persist_full.md)。全量结果以该输出目录的
`completion.json` / `full_fit_metrics.json` 为准，不能用上面的 8 场景成绩代替。
