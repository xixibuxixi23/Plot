# M3：开启QK并降学习率的权重续训

目的：从外部run `9v021ve5` 的完整检查点继续改善质量，同时开启DiT主干QK
RMSNorm、将LR从1e-4降为3e-5。这是联合改动的续训，不是此前提出的单变量
从零对照；结果不能用于单独判断QK的收益。

源码基于该run的启动提交 `9373a6791c365d6c29a8732b9aceaf85b9529983`，
分支 `experiment/m3-qk-lr-continuation-20260919`。没有混入后续预训练加载、
人物像素loss或冻结主干的改动，也不读取PERSIST/2daction的DiT权重。

## 续训语义

- 加载指定父checkpoint的全部旧模型tensor；主干每层空间/时间注意力的
  Q/K归一化gain新增为1。配置、词表或其他tensor不匹配直接报错。
- 继承父checkpoint的累计step。例如parent15500、STEPS40000意味着再训24500步。
- 重新创建AdamW，LR固定3e-5，beta=(0.9,0.999)、weight_decay=0.01。
  **不恢复父optimizer、RNG或dataloader进度**；沿用旧trainer的warm-start行为。
- 除QK外，renderer配置必须与父checkpoint一致，包括65/32/8与reference配置。
  标准化由原有codec加载逻辑继承，并拒绝冲突设置。Pixel VAE继续冻结。
- 正式recipe维持7卡×batch4、全网flow、均匀块级t、无EMA和warmup。
- gain=1不代表新增归一化是恒等变换。开始后的loss可能跳变，要看后续趋势
  和同probe画质，不能把新旧训练曲线的连接点当成连续无扰动恢复。

## 接收版本

将 `m3_qk_lr_continue_20260919.bundle` 复制到远端任意位置。该bundle是基于
`9373a67`的增量包，接收仓库需要已有该提交；原训练仓库满足这个条件。

```bash
cd /pfs/pfs-7jnepv/zkw/project/polis-v1-bundle-20260911/Plot
git bundle verify /path/to/m3_qk_lr_continue_20260919.bundle
git fetch /path/to/m3_qk_lr_continue_20260919.bundle \
  experiment/m3-qk-lr-continuation-20260919
git worktree add --detach ../Plot-qk-lr-continue-20260919 FETCH_HEAD
cd ../Plot-qk-lr-continue-20260919
git rev-parse HEAD
```

这些命令不切换原训练工作区。新worktree需能访问原环境和冻结VAE；例如在确认
源VAE目录存在且目标不存在后，为 `checkpoints/pixel_vae` 创建指向原工作区
同名目录的符号链接。不要复制或替换正在运行任务的环境。

## 启动

先在远端确认parent是run `9v021ve5` 最近完整落盘的checkpoint，保存SHA-256和
parent step。不要使用正在写入的文件。不要使用人物微调run的checkpoint。
保留原run与检查点，资源不足时排队，不默认停止基线。

```bash
export PYTHON_BIN=/path/to/the/original/training/environment/bin/python
export PLOT_DATASET_ROOT=/pfs/pfs-7jnepv/zkw/project/polis-fixed-skins-20260917-data/polis_two_player_fixed_skins_complete_20260917_360p
export WARM_START=/path/to/9v021ve5/step_NNNNNNN.pt
export OUTPUT_DIR=/pfs/pfs-7jnepv/zkw/project/polis-fixed-skins-20260917-output/m3-simple/qk_lr_continue_20260919
export PLOT_CHECKPOINT_STAGING_DIR=/pfs/pfs-7jnepv/zkw/project/polis-fixed-skins-20260917-checkpoint-staging/qk_lr_continue_20260919
export CUDA_VISIBLE_DEVICES='REPLACE_WITH_7_FREE_GPU_IDS'
export WANDB_NAME=m3-qk-lr3e5-fromPARENT-7gpu-b4
export STEPS=40000
bash train_scripts/recipes/m3/train_m3_qk_lr_continue_7gpu.sh
```

替换占位内容，GPU编号用逗号分隔。`OUTPUT_DIR`必须全新或为空，recipe会拒绝
已有内容的目录。词表和window index复用原文件并记录哈希，不能顺便换release。
学习率在专用recipe中固定为3e-5，不会被旧环境中的`LR`覆盖。

正式训练前，用另一组临时输出/staging目录和`WANDB_MODE=disabled`做短smoke：
将STEPS设为parent_step+5，SAVE_EVERY和VALIDATE_EVERY也设为该累计目标，
VISUALIZE_EVERY=0，只在结束时保存和验证一次，确认实际大模型
在远端能够加载、反向、保存和验证。smoke完成后，正式任务仍从原parent开始，
不使用smoke输出；恢复STEPS=40000、保存/验证频率500/1000及VISUALIZE_EVERY=1000。
不将本地CPU小模型通过等同于远端大模型GPU smoke通过。

启动日志应包含：

```text
"migration": "enable_qk_rms_norm"
"optimizer_state": "reset"
"rng_and_data_iterator": "reset"
"parent_step": <实际父检查点步数>
"lr": 3e-05
```

12层主干应新增48个gamma tensor（每层空间/时间×Q/K），其他旧tensor全部迁移。
W&B和config.json会保存`qk_migration`报告。检查
`renderer.qk_rms_norm=true`、`training.lr=3e-5`，且`resume`和`backbone_checkpoint`
为空，`warm_start`准确指向父checkpoint。

## 后续恢复与评估

这条新run保存的checkpoint已经含QK参数。后续中断恢复使用普通`--resume`，
保持QK和其他架构参数一致，**不要再次传`--warm-start-enable-qk-rms-norm`**。
原recipe专门用于第一次迁移，不能直接重复运行来充当strict resume。

第一轮看续训后的1000–2000步：固定probe生成、人物/背景质量和flow趋势，
而不是只看改变架构后的第一个loss。原验证覆盖可能偏S01，人物probe需单独看。
当前训练配方不改变验证选样，以便保留旧run的对照；全面离线评估同时作用于
新旧checkpoint。记录新commit、run URL、parent路径/哈希、输出、环境和GPU分配。

## 背景精修重启

若固定验证与背景probe仍在改善、但希望降低成熟阶段的更新幅度，使用
`train_m3_qk_background_refine_7gpu.sh`从本run最近完整checkpoint普通恢复。
该recipe保留AdamW moments，将恢复后的实际LR显式覆盖为`1e-5`，并为训练
sampler使用独立seed，避免新进程再次从`seed=0`的相同窗口排列开头重放；验证
和固定probe仍使用原`seed=0`，可与前一run直接比较。输出目录和W&B run必须新建。

```bash
export RESUME=/path/to/qk_lr_continue/step_NNNNNNN.pt
export OUTPUT_DIR=/path/to/new/qk_background_refine_lr1e5
export PLOT_CHECKPOINT_STAGING_DIR=/path/to/new/staging
export CUDA_VISIBLE_DEVICES='REPLACE_WITH_7_FREE_GPU_IDS'
export STEPS=40000
bash train_scripts/recipes/m3/train_m3_qk_background_refine_7gpu.sh
```

启动日志必须同时报告原checkpoint LR为`3e-5`、active LR为`1e-5`。W&B新增
`train/total_loss_ema_099`用于观察趋势；即时loss仍保留。不要对已含QK参数的
checkpoint再次使用`--warm-start-enable-qk-rms-norm`。

本地验证命令：

```bash
python -m pytest tests/test_renderer.py -q
bash -n train_scripts/recipes/m3/train_m3_qk_lr_continue_7gpu.sh
python train_scripts/train_renderer.py --help
```

本版本的兼容性测试包含旧tensor精确复制、归一化gain初始化、前向/反向和
优化器更新、新格式strict恢复，以及缺失/多余/形状异常tensor、配置或词表变化
的拒绝路径。实际远端checkpoint和GPU训练仍须接收端验证。
