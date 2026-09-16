# PLOT

PLOT 是 **Multi-Inhabitant World Models with Writable 3D Memory** 的主代码仓库，
包含 M1–M4、数据适配器、训练入口、索引和固定的上游权重。大规模数据、训练输出和
新 checkpoint 不进入 Git。

| 模块 | 入口 | 作用 |
|---|---|---|
| M1 | `experiments/m1/train_multiview_persist_full.py` | 从视觉证据补全 48³ 世界记忆 |
| M2 | `train_scripts/train_transition_full.py` | 预测运动、交互目标和稀疏写入 |
| M3 | `train_scripts/train_renderer.py` | 65 帧 causal diffusion-forcing 渲染器 |
| M4 | `train_scripts/train_policy.py` | 根据完成的 M3 观察预测 8 步动作 |

本文重点是一台新机器如何从零准备并启动标准 M3 八卡训练。若要审阅完整代码和
理解项目设计，请先阅读仓库根目录的
**[`PLOT 代码与项目逻辑说明`](PROJECT_LOGIC_ZH.md)**；其中给出了 M1–M4 数据流、核心文件
地图、闭环执行顺序、当前实现边界和推荐审阅顺序。


## 先明确几个概念

- `BATCH_SIZE` 是**每张 GPU** 的 source-window batch。
- 默认 `--target-views-per-window 2`，所以每个 source window 会展开成两个玩家视角。
- 训练 `WORKERS=4` 是**每个 DDP rank** 四个 DataLoader worker。八卡共 32 个
  CPU worker；它不改变 batch，只负责并行读取和预处理未来 batch。
- chunk 转换命令的 `--workers 2` 是**整台机器总共两个转换进程**，不乘 GPU 数。
- chunk8 是可删除、可重建的本地缓存，不是发布数据，也不应上传或跨机器传输。
- 原始数据仍然必须保留：视频、外观 PNG、事件等仍从原始 episode 目录读取；chunk8
  只优化大型 NPZ 时序数组的读取。

八卡、每卡 batch 为 `B`、每个 window 两个视角、无梯度累积时：

```text
每步 source windows = B × 8
每步实际 view 数     = B × 2 × 8
```

例如 `B=2` 时，每步是 16 个 source windows、32 个玩家视角。

## 1. 准备代码、环境和路径

进入仓库并记录当前代码版本。正式训练开始后不要在同一个工作树中 `git pull`：

```bash
cd /path/to/Plot
git status --short
git rev-parse HEAD
```

同一集群且环境兼容时可以尝试仓库中的 `.venv`：

```bash
source .venv/bin/activate
```

换机器、换驱动或换 CUDA 时，推荐新建 Python 3.11 环境，并按
`requirements/m3-b200.txt` 安装。B200 必须使用包含 `sm_100`、CUDA 12.8 支持的
PyTorch；普通 CUDA wheel 不一定可用。

设置高速本地盘或 PFS 路径。checkpoint staging 不要放在 OSS FUSE 上：

```bash
export PLOT_DATASET_ROOT=/fast/data/polis_v1_20260909_360p
export PLOT_M3_CACHE_ROOT=/fast/cache/polis_m3_chunk8
export PLOT_CHECKPOINT_STAGING_DIR=/fast/checkpoints/plot
export PLOT_RUN_ROOT=/fast/outputs/plot
mkdir -p "$PLOT_M3_CACHE_ROOT" "$PLOT_CHECKPOINT_STAGING_DIR" "$PLOT_RUN_ROOT"
```

如果只使用部分 GPU，要同时设置两项，二者数量必须一致：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
```

## 2. 下载并校验原始数据

训练包的标准数据是 Hugging Face 数据集 `xixibuxixi/polis-v1` 中的
`polis_v1_20260909_360p`。项目脚本会逐 shard 下载并解压 train 与 val-ID：

```bash
python scripts/download_dataset.py \
  --output "$PLOT_DATASET_ROOT" \
  --splits train val_id
```

若需要 Hugging Face token，只通过当前命令的环境变量或项目外的权限文件提供；不要把
明文 token 写入 README、Git 或训练日志。

随后校验数据和固定 checkpoint：

```bash
python scripts/validate_bundle.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --check-checkpoint-hashes
```

标准索引应存在：

```text
derived/m3/validated/train_c65.pt
derived/m3/validated/val_id_c65.pt
derived/common/block_vocabulary.json
```

这些索引中的 episode 路径是相对路径，可以随数据目录迁移。若改用 fixed-skin 等其他
数据版本，必须使用与该版本配套生成的 train/val index，不能把不同 release 的索引和
episode 混用。

## 3. 检查 GPU、CUDA 和 checkpoint 落盘

```bash
nvidia-smi
python scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir "$PLOT_RUN_ROOT/environment_check" \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"
```

必须确认输出中以下项目没有 error：

- PyTorch 能看到预期 GPU；
- B200 对应 `sm_100`；
- BF16 可用；
- nvdiffrast CUDA context 创建成功；
- checkpoint 能先写 staging，再完整复制到输出目录。

## 4. 在目标机器本地生成 chunk8

不在源机器预先生成或上传 chunk8。原始数据下载完成后，在目标机器自己的高速盘执行。
train 和 val 都要转换到同一个 cache root：

```bash
python scripts/materialize_m3_chunk_cache.py \
  --window-index derived/m3/validated/train_c65.pt \
  --source-root "$PLOT_DATASET_ROOT" \
  --output-root "$PLOT_M3_CACHE_ROOT" \
  --chunk-frames 8 \
  --workers 2

python scripts/materialize_m3_chunk_cache.py \
  --window-index derived/m3/validated/val_id_c65.pt \
  --source-root "$PLOT_DATASET_ROOT" \
  --output-root "$PLOT_M3_CACHE_ROOT" \
  --chunk-frames 8 \
  --workers 2
```

转换器具有以下性质：

- 不修改原始 NPZ；
- 先写临时文件，完成后原子替换；
- 重启时校验并复用已完成缓存；
- 生成只含相对路径的新索引，不携带源机器的 `/public/...` 等绝对路径。

默认会生成：

```text
$PLOT_M3_CACHE_ROOT/train_c65_chunk8.pt
$PLOT_M3_CACHE_ROOT/val_id_c65_chunk8.pt
```

把它们交给八卡 recipe：

```bash
export M3_WINDOW_INDEX="$PLOT_M3_CACHE_ROOT/train_c65_chunk8.pt"
export M3_VAL_WINDOW_INDEX="$PLOT_M3_CACHE_ROOT/val_id_c65_chunk8.pt"
export M3_CHUNK_CACHE_ROOT="$PLOT_M3_CACHE_ROOT"
```

如需临时回退到原始 NPZ，取消 `M3_CHUNK_CACHE_ROOT`，并把两个 index 恢复为
`derived/m3/validated/*.pt` 即可。

## 5. 自动测试每卡 batch 4/2/1

训练默认每卡四个 DataLoader worker：

```bash
export WORKERS=4
```

八卡时即全机 32 个 worker。当前 `prefetch_factor=2`，因此应同时观察主机内存和磁盘
吞吐；若出现主机 OOM 或严重 I/O 争抢，再降到 `WORKERS=2`，而不是先改变 GPU batch。

按 4、2、1 顺序执行真实多卡 forward/backward，选择能够稳定完成的最大每卡 batch：

```bash
export PLOT_BATCH_TUNE_ROOT="$PLOT_RUN_ROOT/m3_batch_tuning"
bash scripts/tune_m3_batch.sh
source "$PLOT_BATCH_TUNE_ROOT/selected_batch.env"
echo "selected per-GPU BATCH_SIZE=$BATCH_SIZE"
```

不要仅凭 `nvidia-smi` 中的空闲显存猜 batch。`batch=4` 失败后脚本会自动尝试 2，再尝试
1；若三者都失败，查看 `$PLOT_BATCH_TUNE_ROOT/batch_*.log`，修复后重新测试。

## 6. 做一次两步八卡 smoke test

```bash
export OUTPUT_DIR="$PLOT_RUN_ROOT/m3_smoke"
export STEPS=2
export SAVE_EVERY=2
export VALIDATE_EVERY=2
export VISUALIZE_EVERY=0
export WANDB_MODE=disabled

bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh
```

smoke 至少需要确认：

- 所有 rank 都启动且没有 NCCL hang；
- 两个 optimizer step 都完成；
- loss 为有限值；
- checkpoint 能写入并重新读取；
- 各卡显存没有持续增长；
- 日志中使用的是选中的 `BATCH_SIZE`、`WORKERS=4` 和 chunk-cache index。

标准 `train_m3_b200_8gpu.sh` 从固定 M3 backbone 初始化一条新训练，不会自动续接任意
实验 checkpoint。续训必须使用与 checkpoint 完全一致的结构参数和对应 recipe，不能只把
一个旧 checkpoint 路径塞给标准 recipe。

## 7. 启动正式训练

smoke 通过后，固定当前 commit，再启动 10,000 step：

```bash
export OUTPUT_DIR="$PLOT_RUN_ROOT/m3_formal"
export STEPS=10000
export SAVE_EVERY=1000
export VALIDATE_EVERY=1000
export VISUALIZE_EVERY=1000
export VISUALIZATION_DENOISING_STEPS=20
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-plot-m3}
export WANDB_NAME=${WANDB_NAME:-m3-formal-b200}

mkdir -p "$OUTPUT_DIR"
nohup bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh \
  >"$OUTPUT_DIR.launch.log" 2>&1 &
echo $! >"$OUTPUT_DIR.launch.pid"
```

如果没有 W&B 凭据：

```bash
export WANDB_MODE=offline
```

不要因为 W&B 未登录而阻塞训练。离线记录之后可以再同步。

默认 `LOSS_MODE=combined`，即 flow loss 加当前配置的像素辅助 loss。如果只想训练扩散
flow loss：

```bash
export LOSS_MODE=flow
```

`--latent-player-region-upweight` 和 `--latent-entity-region-upweight` 属于 flow loss 内部的
空间权重，不是 decoded-pixel 辅助 loss。

## 8. 启动后检查与最终报告

查看进程、日志和 GPU：

```bash
cat "$OUTPUT_DIR.launch.pid"
ps -fp "$(cat "$OUTPUT_DIR.launch.pid")"
tail -f "$OUTPUT_DIR.launch.log"
nvidia-smi
```

查看输出：

```bash
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort | tail -50
```

前 100 step 要持续检查：

- 八张卡利用率和显存；
- DataLoader 是否让 GPU 长时间等待；
- loss 是否下降且无 NaN/Inf；
- step 用时与吞吐；
- checkpoint、validation 和 W&B 是否写到预期位置。

最终交接至少报告：

- Git commit；
- 数据路径及校验结果；
- 使用的 GPU 和 `NPROC_PER_NODE`；
- 每卡 batch、每卡 worker、effective source/view batch；
- chunk cache 与 portable index 路径；
- smoke 结果；
- W&B URL 或 offline 目录；
- 正式训练 PID、日志、输出和 checkpoint 路径；
- 当前 step、step/s 或 s/step、预计完成时间。

## Git 与大文件规则

首次部署可使用 Hugging Face 训练包；后续代码通过 GitHub 增量更新。服务器修复放在
`b200/<topic>` 分支，commit、push 后通过 PR 合入 main。正式训练固定一个 commit，运行
期间不 pull。

不要提交以下内容：

- 原始数据与 chunk8 cache；
- `.venv`；
- checkpoint；
- `outputs/` 与 W&B 文件；
- 含本机绝对路径的大型派生索引；
- token、密码或其他凭据。

完整的目标机器自动化交接文字仍保留在 [`HANDOFF_README.md`](HANDOFF_README.md)。
