# PLOT：交给目标机器 Codex 的唯一指令

师姐不需要研究项目，也不需要手动配置。请把下面整段原样交给目标机器上的 Codex：

> 你现在全权负责部署并训练 PLOT。请直接执行全部工作，不要停在解释、命令建议或
> 等待人工逐项确认。训练包和数据公开位于 Hugging Face 数据集
> `xixibuxixi/polis-v1`。先从 `bundles/` 下载以下三个文件：
> `plot_training_bundle_20260911.tar.zst`、同名 `.sha256` 和
> `plot_training_bundle_20260911_README.md`；校验 SHA-256 后解压并进入 `Plot`。
>
> 查看 `nvidia-smi`，自动选择当前空闲的全部 B200，并设置 `CUDA_VISIBLE_DEVICES`
> 与 `NPROC_PER_NODE`。选择高速本地盘或 PFS 作为工作目录、数据目录、训练输出和
> checkpoint staging；不要直接向 OSS FUSE 序列化 checkpoint。磁盘容量已确认
> 足够，不要因此询问或停工。
>
> 优先使用包内 `.venv`。若与本机驱动或系统不兼容，依据
> `requirements/m3-b200.txt` 新建 Python 3.11 环境。运行
> `scripts/download_dataset.py --output <高速盘>/polis_v1_20260909_360p
> --splits train val_id`，让脚本下载并逐 shard 解压数据。运行
> `scripts/validate_bundle.py --dataset-root <数据目录>
> --check-checkpoint-hashes`，随后运行 `scripts/check_m3_environment.py`，确认
> PyTorch、CUDA、B200 `sm_100`、BF16、nvdiffrast 和 checkpoint 原子落盘都正常。
>
> 先运行 `scripts/tune_m3_batch.sh`。它按每卡 batch 4、2、1 的顺序做真实
> forward/backward/DDP 测试并选择能稳定完成的最大 batch；不要仅凭显存容量猜测。
> 使用选出的 batch 再做一次 2-step 多卡 smoke run，启用两步 denoising 的五类
> 可视化，确认 65 帧 causal flow loss、随机两个玩家、`batch × player`、32 帧
> KV cache、每 chunk 8 帧以及总计 64 帧 rollout 都通过。若 batch 4 或 2 OOM，
> 自动降一级继续，不要停下来询问。
>
> smoke 通过后启动 `train_scripts/recipes/m3/train_m3_b200_8gpu.sh` 的 10000-step
> 正式训练，使用选出的每卡 batch。每 1000 step 保存、验证并生成 construction、
> four-player、PvE、three-resident combat、mixed build/combat 五组 W&B 视频。
> 如果没有 `WANDB_API_KEY`，使用 `WANDB_MODE=offline` 继续，不要阻塞训练。持续观察
> 前 100 step 的 loss、各卡显存、GPU 利用率、数据吞吐和错误日志。发生问题时定位
> 并修复代码，重新做最小测试后恢复训练。
>
> 如果提供了 `PLOT_GITHUB_URL`，在首次 smoke 通过后执行
> `bash scripts/connect_git_remote.sh "$PLOT_GITHUB_URL" main`。以后所有代码修复都在
> `b200/<topic>` 分支提交并 push，通过 PR 合入 main；不要只留下服务器本地修改。
> 正式训练固定一个 commit，运行过程中不要 pull。绝对不要执行 `git clean -fdx`，
> 不要提交数据、`.venv`、checkpoint、派生大索引、W&B 或训练输出。
>
> 完成后只向委托人报告：数据路径和校验结果、GPU 列表、环境检查、最终每卡 batch、
> effective batch、smoke 结果、五组视频结果、Git commit、W&B URL或离线目录、正式
> 训练 PID、日志、输出/checkpoint 路径、当前 step、吞吐和预计完成时间。不要使用、
> 保存或打印任何历史对话中出现过的 token。

## Codex 可采用的命令骨架

```bash
# 下载训练包（也可以由人提前复制到机器）
hf download xixibuxixi/polis-v1 \
  bundles/plot_training_bundle_20260911.tar.zst \
  bundles/plot_training_bundle_20260911.tar.zst.sha256 \
  bundles/plot_training_bundle_20260911_README.md \
  --repo-type dataset --local-dir plot-bootstrap

cd plot-bootstrap
(cd bundles && sha256sum -c plot_training_bundle_20260911.tar.zst.sha256)
tar --zstd -xf bundles/plot_training_bundle_20260911.tar.zst
cd Plot
source .venv/bin/activate

export PLOT_DATASET_ROOT=/fast/data/polis_v1_20260909_360p
export PLOT_CHECKPOINT_STAGING_DIR=/fast/checkpoints/plot
export NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
python scripts/download_dataset.py --output "$PLOT_DATASET_ROOT" --splits train val_id
python scripts/validate_bundle.py \
  --dataset-root "$PLOT_DATASET_ROOT" --check-checkpoint-hashes

python scripts/check_m3_environment.py --device cuda:0 \
  --output-dir /fast/outputs/plot_environment_check \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"

export PLOT_BATCH_TUNE_ROOT=/fast/outputs/plot_m3_batch_tuning
bash scripts/tune_m3_batch.sh
source "$PLOT_BATCH_TUNE_ROOT/selected_batch.env"

export OUTPUT_DIR=/fast/outputs/plot_m3_smoke
export STEPS=2 SAVE_EVERY=2 VALIDATE_EVERY=2 VISUALIZE_EVERY=2
export VISUALIZATION_DENOISING_STEPS=2
export WANDB_MODE=${WANDB_MODE:-offline}
bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh

export OUTPUT_DIR=/fast/outputs/plot_m3_formal
export STEPS=10000 SAVE_EVERY=1000 VALIDATE_EVERY=1000 VISUALIZE_EVERY=1000
export VISUALIZATION_DENOISING_STEPS=20
mkdir -p "$OUTPUT_DIR"
nohup bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh \
  >"$OUTPUT_DIR.launch.log" 2>&1 &
echo $! >"$OUTPUT_DIR.launch.pid"
```

## 后续代码传代

第一次使用 Hugging Face 大包初始化。之后代码只从同一个 GitHub 仓库增量更新；
数据、环境、checkpoint 和训练结果一直保留在目标机器。连接一次：

```bash
export PLOT_GITHUB_URL=https://github.com/xixibuxixi23/Plot.git
bash scripts/connect_git_remote.sh "$PLOT_GITHUB_URL" main
```

以后目标 Codex 在没有训练占用该工作树时运行：

```bash
git switch main
git pull --ff-only
```

若目标 Codex 修复 B200 问题，应创建 `b200/<topic>` 分支、提交并 push。完整规则见
`docs/collaboration.md`。
