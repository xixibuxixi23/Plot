# H20合作伙伴：直接下载环境包

这份说明面向另一位从零配置机器的合作伙伴。仅部署代码、环境和数据，不默认
接续`9v021ve5`，不绑定7卡，也不启动训练。训练初始化方式另行确定。

## 环境包

包内包含Python 3.11.14、PyTorch 2.7.1+cu128、torchvision 0.22.1、NumPy、
OpenCV、utils3d、W&B、Hugging Face客户端及已编译的nvdiffrast sm_90内核。
不用逐项pip安装，也不需要为这些现成算子重新下载源码或编译CUDA扩展。
不含NVIDIA驱动、项目代码、数据、模型权重或账户凭据。

适用范围：Linux x86_64，按Ubuntu22.04/glibc2.35或兼容的更新系统准备；
需要系统libstdc++与可运行CUDA12.8用户态库的NVIDIA驱动。H20属于sm_90，
本包的nvdiffrast原生内核针对sm_90，不应直接拿去当作其他架构的环境包。
不需要Docker或root来解压和使用；系统驱动仍由机器管理员提供。

已验证在不同目录中运行：所有关键模块从移动后的包内加载，H200(sm_90)上
BF16矩阵运算、nvdiffrast光栅化前向/反向通过，renderer测试48项通过。
这不是H20实机验证，接收端仍需运行一次环境检查。

已发布：[环境包直接下载](https://huggingface.co/datasets/xixibuxixi/polis-v1/resolve/eb46cbdcb29f91ce6ba7493c70e909db2bb3615d/runtimes/m3_h20_py311_cu128_20260919/plot-env-py311-cu128-hopper-20260919.tar.gz?download=true)，4,403,381,254 bytes（约4.10GiB）。
SHA-256：`f9c5c455893e850b7c1b8da776bbc6dc883ae532d40d047793bdcda4bbf0b730`。
[发布清单和验证记录](https://huggingface.co/datasets/xixibuxixi/polis-v1/tree/eb46cbdcb29f91ce6ba7493c70e909db2bb3615d/runtimes/m3_h20_py311_cu128_20260919)。

## 下载和解压

在计划存放环境的目录执行。下载地址指向环境发布目录，支持中断后重新运行下载。

```bash
curl --fail --location --retry 5 --continue-at - \
  'https://huggingface.co/datasets/xixibuxixi/polis-v1/resolve/eb46cbdcb29f91ce6ba7493c70e909db2bb3615d/runtimes/m3_h20_py311_cu128_20260919/plot-env-py311-cu128-hopper-20260919.tar.gz?download=true' \
  --output plot-env-py311-cu128-hopper-20260919.tar.gz

printf '%s  %s\n' 'f9c5c455893e850b7c1b8da776bbc6dc883ae532d40d047793bdcda4bbf0b730' \
  plot-env-py311-cu128-hopper-20260919.tar.gz | sha256sum -c -
tar -xzf plot-env-py311-cu128-hopper-20260919.tar.gz
bash plot-env-py311-cu128/setup.sh
source plot-env-py311-cu128/activate.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

`setup.sh`只修复解压位置对应的Python路径，不联网安装包。以后移动环境目录时
重新运行setup.sh。每个新shell使用`source /实际路径/plot-env-py311-cu128/activate.sh`。
无需预装Python；setup.sh使用包内的解释器。
环境解压约7.4GiB，下载和解压同时保留时需为两者之和留足磁盘空间。

## 代码和新机检查

```bash
git clone --depth 1 --branch experiment/m3-qk-lr-continuation-20260919 \
  https://github.com/xixibuxixi23/Plot.git Plot
cd Plot
git rev-parse HEAD

# 选择本机一张确认空闲的GPU，只做环境检查。
export CUDA_VISIBLE_DEVICES='REPLACE_WITH_ONE_FREE_GPU_ID'
export PLOT_CHECKPOINT_STAGING_DIR=/path/on/local-disk/plot-checkpoint-staging
python scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir outputs/h20_environment_check \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"
```

不用再运行`pip install -r ...`或复制原工程的`.venv`。PLOT训练入口会把项目根目录
加入Python路径，不需要为启动训练执行editable安装。该分支提供QK开关和续训
功能，克隆它不代表必须使用续训recipe。正式训练前按实际机器重新设置GPU列表。

## 数据、VAE与账户

环境准备完后可直接使用包内的Hugging Face客户端：

```bash
export PLOT_DATASET_ROOT=/path/on/data-disk/polis_two_player_fixed_skins_complete_20260917_360p
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=60
python scripts/download_dataset.py \
  --output "$PLOT_DATASET_ROOT" \
  --version fixed_skins_20260917 \
  --revision ddbc73d73075a6de1392746824cb61dff3cdf242 \
  --splits train val_id

hf download PERSIST-team/persist-pixel-vae model.safetensors \
  --revision ac2c8ed622e3acaf071790eb4a021f0a367d82ae \
  --local-dir checkpoints/pixel_vae

python scripts/check_m3_simple_ready.py \
  --dataset-root "$PLOT_DATASET_ROOT" --check-pixel-vae-hash
wandb login
```

数据仍为52个tar分片、约386.97GiB，下载器逐片校验、解压、清理下载缓存，
可以中断后继续。下载环境包减少分散的依赖下载，不会减少训练数据的体积。
数据盘建议至少500GiB可用空间，训练检查点另留空间；已上传的窗口索引随数据
一起下载，不需要重新采集、转码或构建缓存。

合作伙伴使用自己的W&B凭据，并确保有目标project权限。完成部署后回报GPU
数量/显存、系统版本、环境检查结果、代码commit和数据路径。若随后选择从零
训练M3，不加载M3检查点；冻结Pixel VAE仍需要。若另行选择续训，再明确父检查点。
