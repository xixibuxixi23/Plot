# M3-Simple 训练交接

当前正式目标是从零训练 M3-Simple 40,000 step，不加载或覆盖旧 M3 的 40k
checkpoint。网络只有三条条件路线：融合场景编码、目标玩家状态 AdaLN、一次玩家外观
reference attention。

## 交给目标机器 Codex 的指令

把下面整段交给目标机器上的 Codex：

> 请在目标机器部署并训练 PLOT 的 M3-Simple。固定使用交付方指定的 Git commit，
> 不要切回旧的 M3 resume recipe。数据集是
> `polis_two_player_fixed_skins_complete_20260917_360p`，必须包含 27,043 个 episode、
> `derived/m3/validated/train_c65.pt` 和 `val_id_c65.pt`。同一集群若已挂载 `/public`
> 就直接使用共享路径，不要重复复制 388 GB 数据。
>
> 确认 `checkpoints/pixel_vae/model.safetensors` 存在；它是冻结的 Pixel-VAE，Git
> 不包含这个 868 MB 文件。运行 `scripts/check_m3_simple_ready.py` 和
> `scripts/check_m3_environment.py`。检查失败时先修复输入或环境，不要带病启动正式训练。
>
> 使用 W&B online，entity 为 `ckx23-tsinghua-university`，project 为 `plot-m3`。
> 凭据应由 `wandb login` 写入目标机器的 `~/.netrc`，不要把 API Key 提交到 Git。
> 先查看 `nvidia-smi`，只选择空闲 GPU。先把 `STEPS=3` 跑通；随后用同一 commit、
> 同一数据和同一 recipe 启动 40k 正式训练。运行中记录 PID、日志、W&B URL、输出目录、
> checkpoint 目录、step 时间、峰值显存和预计完成时间。不要杀死或共享别人的 GPU 进程。

## 需要传递的三部分

1. **代码**：GitHub 分支/commit。不要传 `outputs/`、W&B 目录、数据或 checkpoint。
2. **冻结资产**：`checkpoints/pixel_vae/model.safetensors`，精确大小 909,781,080 bytes，
   SHA-256 为 `eb634803c94aeea980046961382f2ab67157aa71e92c5a183a35e4f61f8cbc36`。
3. **数据**：388 GB 的 canonical 360p release。共享 `/public` 时只传路径；没有共享盘时
   使用 `rsync -aH --info=progress2`，不要通过 GitHub。

W&B API Key 不是项目文件。目标机器运行一次 `wandb login --relogin`，或者通过安全渠道
复制用户级 `~/.netrc` 并设置权限 `chmod 600 ~/.netrc`。

## 当前集群的标准路径

```bash
export PLOT_DATASET_ROOT=/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/polis_two_player_fixed_skins_complete_20260917_360p
export PLOT_CHECKPOINT_STAGING_DIR=/tmp/plot_m3_simple_checkpoints
export WANDB_MODE=online
export WANDB_ENTITY=ckx23-tsinghua-university
export WANDB_PROJECT=plot-m3
```

## 目标机器验收

```bash
source .venv/bin/activate

python scripts/check_m3_simple_ready.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --check-pixel-vae-hash \
  --check-wandb

python scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir /tmp/plot_m3_environment_check \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"
```

若目标机器没有共享数据盘，可从源机器复制：

```bash
rsync -aH --info=progress2 \
  SOURCE_HOST:/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/polis_two_player_fixed_skins_complete_20260917_360p/ \
  /fast/data/polis_two_player_fixed_skins_complete_20260917_360p/

mkdir -p checkpoints/pixel_vae
rsync -ah --info=progress2 \
  SOURCE_HOST:/public/0_DATA/2_Avatar/zhizhou_share/rcz/Plot/checkpoints/pixel_vae/model.safetensors \
  checkpoints/pixel_vae/model.safetensors
```

## 3-step smoke

下面示例使用物理 GPU 4–7。目标机器必须根据实际空闲卡调整：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export OUTPUT_DIR=/tmp/plot_m3_simple_smoke
export STEPS=3
export SAVE_EVERY=500
export VALIDATE_EVERY=1000
export VISUALIZE_EVERY=1000
export WANDB_NAME=m3-simple-smoke

bash train_scripts/recipes/m3/train_m3_simple_4gpu.sh
```

当前 H200 实测为约 1.54 秒/step、单卡峰值显存 23.3–23.8 GiB。smoke 必须看到
forward、combined loss、backward 和 optimizer step 全部完成。

## 40k 正式训练

```bash
export OUTPUT_DIR=/fast/outputs/m3_simple_fixed_skins_40k
export STEPS=40000
export SAVE_EVERY=500
export VALIDATE_EVERY=1000
export VISUALIZE_EVERY=1000
export VISUALIZATION_DENOISING_STEPS=20
export WANDB_NAME=m3-simple-fixed-skins-40k

mkdir -p "$OUTPUT_DIR"
nohup bash train_scripts/recipes/m3/train_m3_simple_4gpu.sh \
  >"$OUTPUT_DIR.launch.log" 2>&1 &
echo $! | tee "$OUTPUT_DIR.launch.pid"
```

默认是 4 卡、每卡 batch 1、无梯度积累、BF16、65 帧上下文、32 帧 KV cache、8 帧
causal block。按本机实测，40k 约需 17.1 小时；换机器后以 smoke 实测重新估算。

安全停止时只终止 `launch.pid` 对应的本次训练进程组，绝不能按 Python 名称批量 kill。
