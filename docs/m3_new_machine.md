# 新机器部署M3：代码、环境、数据与权重

核对日期：2026-09-19。以下准备步骤不启动训练。适用于目前QK＋LR续训分支；
训练实现提交为`66dfc7c`，后续文档提交不改变该训练实现。
新机器从零安装，不代表模型需要从零训练。

## 1. 代码与环境

在有足够空间的工作目录执行：

```bash
git clone --branch experiment/m3-qk-lr-continuation-20260919 \
  https://github.com/xixibuxixi23/Plot.git Plot
cd Plot
git rev-parse HEAD

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r requirements/m3-b200.txt
python -m pip install -e .
python -m pip install huggingface_hub
python -m pip freeze > environment-new-machine.txt
```

系统需要可用的NVIDIA驱动，以及Git、C/C++编译工具、CUDA toolkit（包含nvcc）；
优先使用CUDA12.8开发环境，编译nvdiffrast需要toolkit，只有PyTorch wheel不够。
Python使用3.11.8或更新的3.11补丁版本，以支持下载脚本的tar安全解包接口。
`requirements/m3-b200.txt`固定utils3d和nvdiffrast的Git提交，不要换成同名无关包。
不要从旧机器直接搬`.venv`。GitHub若要求认证，使用执行端自己的仓库访问方式。

上面的PyTorch版本组合来自[官方安装表](https://pytorch.org/get-started/previous-versions/#v2-7-1)。
硬件型号不同不等于必须换全部软件栈，但需要在目标GPU核实架构和BF16支持；
未知显存的机器不能直接假设旧batch可用。该文档未在新机器实际安装或训练。

## 2. 下载训练数据与索引

已核实公开数据仓库：`xixibuxixi/polis-v1`。
使用`fixed_skins_20260917`子目录，固定revision：
`ddbc73d73075a6de1392746824cb61dff3cdf242`。

- train：49个tar，约367.68 GiB；val_id：3个tar，约19.29 GiB。
- 合计约386.97 GiB；最大单个tar约7.72 GiB。这是tar分片总字节统计，
  实际文件系统占用还会受小文件和元数据影响。
- 数据盘建议至少500 GiB可用空间，另为频繁保存的模型/优化器checkpoint
  预留独立空间。此数字不是包含长期训练输出的总容量预算。
- 下载器逐片校验大小和SHA-256、解压并删除该分片缓存；中断后用同命令继续。
  不要把不同release/revision下载进同一个已有目录。

```bash
export PLOT_DATASET_ROOT=/path/on/data-disk/polis_two_player_fixed_skins_complete_20260917_360p
python scripts/download_dataset.py \
  --output "$PLOT_DATASET_ROOT" \
  --version fixed_skins_20260917 \
  --revision ddbc73d73075a6de1392746824cb61dff3cdf242 \
  --splits train val_id
```

请替换数据盘路径。脚本会同时下载release元数据与：

```text
derived/m3/validated/train_c65.pt
derived/m3/validated/val_id_c65.pt
```

不需要重新采集、转码、建立窗口索引或生成额外chunk cache。当前基线并未使用
`chunk_cache_root`；Git中包含词表。公开索引LFS SHA-256与本地已核实索引一致：

```text
train_c65.pt  679f5b1f28fa9056503c70cb9f24825c02edead21bc1c03ef1af622b3af134c4
val_id_c65.pt 43d25d375ad7d853b5019267b0d1c626123be355a5e6d350871c025d185a816e
```

数据页：[固定release](https://huggingface.co/datasets/xixibuxixi/polis-v1/tree/ddbc73d73075a6de1392746824cb61dff3cdf242/fixed_skins_20260917)。

## 3. 下载冻结Pixel VAE

不用搬旧工程里的VAE：已核实PERSIST公开文件与项目manifest的字节数、SHA-256
完全一致。这里只下载VAE，不下载PERSIST DiT，也不加载M3 backbone。

```bash
hf download PERSIST-team/persist-pixel-vae model.safetensors \
  --revision ac2c8ed622e3acaf071790eb4a021f0a367d82ae \
  --local-dir checkpoints/pixel_vae
```

文件909,781,080 bytes，SHA-256：
`eb634803c94aeea980046961382f2ab67157aa71e92c5a183a35e4f61f8cbc36`。
公开来源：[Pixel VAE](https://huggingface.co/PERSIST-team/persist-pixel-vae/tree/ac2c8ed622e3acaf071790eb4a021f0a367d82ae)。

## 4. 决定是否需要旧M3 checkpoint

| 新机器用途 | 需要的M3权重 |
|---|---|
| 接着`9v021ve5`开启QK＋降LR训练 | 从旧机器复制该run完整`.pt`，核对两端SHA-256，设为WARM_START |
| M3从随机初始化开始 | 不需要旧M3检查点；冻结Pixel VAE仍需要 |

不能把“新机器安装环境”当成“随机初始化模型”。新机器若继续本轮任务，照旧
继承父checkpoint步数并重建优化器，详见[续训说明](runs/m3_qk_lr_continuation_20260919.md)。
原run的checkpoint尚无本次核实的公开下载地址；W&B在线曲线不代表权重文件已上传。
目前`xixibuxixi/plot-checkpoints`是私有仓库，其中已列出的legacy M3权重不是这个父链，
不可拿来替代。复制哪个parent由旧机器完整落盘的文件确定，记录来源与哈希。

本分支的`train_m3_qk_lr_continue_7gpu.sh`专用于权重续训，要求WARM_START。
若另做随机初始化实验，应使用无warm-start/resume/backbone的命令；本分支继承的
旧`train_m3_simple.sh`默认也没有打开QK，不可误称调用它就进行了QK从零对照。

## 5. 新机器检查与W&B

首次部署检查编译扩展、GPU和文件，不额外安排长时间训练smoke。
选择一张已确认可用的GPU；环境检查默认使用可见设备的cuda:0。

```bash
export CUDA_VISIBLE_DEVICES='REPLACE_WITH_ONE_FREE_GPU_ID'
export PLOT_CHECKPOINT_STAGING_DIR=/path/on/local-ssd-or-pfs/checkpoint-staging
python scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir outputs/new_machine_preflight \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"

python scripts/check_m3_simple_ready.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --check-pixel-vae-hash

wandb login
```

W&B凭据由执行者在新机配置，不从聊天、Git或旧日志复制。正式启动前把
CUDA_VISIBLE_DEVICES改为实际训练GPU列表，不能沿用上面检查时的一张卡。
专用续训recipe保留7卡×batch4；不足7卡或硬件显存不够时需要独立调整启动配置，
不能硬套。尚未核实原机器所用的具体7张卡，新机应按自己的可用卡编号选择。

准备完成后回报：机器GPU型号/数量/显存、代码commit、数据revision及路径、环境
检查结果、VAE哈希，以及续训parent路径/哈希。要继续本轮训练，再按续训说明
设置新的输出目录和W&B run，目标为累计40000步。
