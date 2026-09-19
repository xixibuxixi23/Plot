本分支（2026-09-19）用于远程 `9v021ve5` 开启QK并降LR至3e-5的权重续训。
入口、接收方式和优化器重置语义见 [续训交接](docs/runs/m3_qk_lr_continuation_20260919.md)。
使用专用 `train_m3_qk_lr_continue_7gpu.sh`；下方保留基线版本的历史说明。
新机器从零安装环境、下载数据和VAE见 [新机器部署](docs/m3_new_machine.md)。

# M3-Simple 实验说明

## 实验目标

训练 M3-Simple 40,000 step。

固定实验配置：

- 使用目标机器上确认空闲的 GPU；每卡 batch 4，effective batch 为 `4 × GPU 数`；
- 不使用梯度积累；
- 只使用均匀的 latent flow loss，不使用区域加权或 decoded-pixel 辅助损失；
- VAE latent 使用固定逐通道标准化 `(z - mean) / std`，解码前还原；统计随代码
  提供，沿用同一 Pixel VAE 的旧 2daction 统计，不需额外下载。新实验从头训练；
- BF16、65 帧上下文、32 帧 KV cache、8 帧 causal block；
- M3-Simple 不使用额外的 `condition_mask` embedding；首帧和已完成历史以 `τ=0`
  表示，保留 block-causal attention mask 和动作窗口起点标记；
- 每窗口一个目标视角；
- W&B online，entity 为 `ckx23-tsinghua-university`，project 为 `plot-m3`；
- 先运行 3-step smoke，再启动 40,000-step 正式训练。

## 外部机器路径配置

所有目录都必须设置为目标机器的实际高速本地盘或 PFS，不依赖原集群的 `/public`
路径。下面的 `/fast` 只是示例：

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export PLOT_DATASET_ROOT=/fast/data/polis_two_player_fixed_skins_complete_20260917_360p
export PLOT_CHECKPOINT_STAGING_DIR=/fast/checkpoints/plot-m3-simple
export PLOT_RUN_ROOT=/fast/outputs/plot-m3-simple

# 先检查显存、利用率和计算进程，再填写全部确认空闲的 GPU。
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
export CUDA_VISIBLE_DEVICES=<comma-separated-free-gpu-indices>
export NPROC_PER_NODE=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export BATCH_SIZE=4

export WANDB_MODE=online
export WANDB_ENTITY=ckx23-tsinghua-university
export WANDB_PROJECT=plot-m3
# WANDB_API_KEY 由委托人单独提供，不写入 Git 仓库。
: "${WANDB_API_KEY:?Set the W&B API key provided by the owner}"
```

创建输出目录，并确认数据盘与 checkpoint 盘空间充足：

```bash
mkdir -p "$PLOT_CHECKPOINT_STAGING_DIR" "$PLOT_RUN_ROOT"
```

## 输入验收

必须存在：

- `checkpoints/pixel_vae/model.safetensors`；
- `$PLOT_DATASET_ROOT/COMPLETE.json`；
- `$PLOT_DATASET_ROOT/derived/m3/validated/train_c65.pt`；
- `$PLOT_DATASET_ROOT/derived/m3/validated/val_id_c65.pt`。

执行完整校验：

```bash
"$PYTHON_BIN" scripts/check_m3_simple_ready.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --check-pixel-vae-hash \
  --check-wandb

"$PYTHON_BIN" scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir "$PLOT_RUN_ROOT/environment_check" \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"
```

期望数据规模：27,043 episodes，其中 train 25,691、val-ID 1,352；训练索引
1,888,161 windows，验证索引 99,062 windows。数据必须报告 missing 0、error 0。

## 3-step smoke

只使用确认空闲的 GPU，不得杀死或共享其他任务的进程：

```bash
export OUTPUT_DIR="$PLOT_RUN_ROOT/smoke"
export STEPS=3
export SAVE_EVERY=500
export VALIDATE_EVERY=1000
export VISUALIZE_EVERY=1000
export VISUALIZATION_DENOISING_STEPS=2
export WANDB_NAME=m3-simple-smoke

bash train_scripts/recipes/m3/train_m3_simple.sh
```

smoke 必须以每卡 batch 4 完成数据加载、forward、combined loss、backward 和
optimizer step，且无 OOM、NaN 或 rank 异常退出。根据 smoke 的实测 step 时间重新估算
正式训练耗时。

## 40k 正式训练

smoke 成功后使用完全相同的 commit、数据、GPU、batch 和模型参数：

```bash
export OUTPUT_DIR="$PLOT_RUN_ROOT/formal_40k"
export STEPS=40000
export SAVE_EVERY=500
export VALIDATE_EVERY=1000
export VISUALIZE_EVERY=1000
export VISUALIZATION_DENOISING_STEPS=20
export WANDB_NAME=m3-simple-fixed-skins-40k

mkdir -p "$OUTPUT_DIR"
nohup bash train_scripts/recipes/m3/train_m3_simple.sh \
  >"$OUTPUT_DIR.launch.log" 2>&1 &
echo $! | tee "$OUTPUT_DIR.launch.pid"
```

正式启动后报告 Git commit、数据校验、GPU、per-GPU batch、effective batch、环境检查、
smoke、W&B URL、PID、日志、checkpoint、当前 step、loss、显存、step 时间和预计完成时间。

安全停止时只终止 `launch.pid` 对应的本次训练进程组，绝不能按 Python 名称批量 kill。
