# 多节点 M1 flow 启动模板

这是通用多节点模板，不记录具体机器的 SSH 入口、IP 或凭据。
`MASTER_ADDR` 必须是所有节点可达的 rank 0 私网地址，不能使用 SSH
网关或端口转发地址。正式启动前必须通过本地预检、节点间 TCP 互通和
NCCL smoke test。

## 准备材料

每台机器需要同一个 PLOT commit 的独立运行工作树、兼容的 CUDA/
驱动与同版本 Python 环境（勿直接跨机复制 `.venv`），以及：

- 完整 `outputs/m1_multiview_persist_s01_cache`，包含metadata.json、ready.npy和全部字段npy；允许本地副本。
- PERSIST的 `data/checkpoints/voxel_decoder/model.safetensors`。
- 同一个不可变的完整checkpoint，包含model/optimizer/epoch/step且batch_in_epoch=0。
  复制现有checkpoint到专门路径，确认后各节点使用该副本；不要使用持续被覆盖的live checkpoint。
- 输出目录使用新路径，不覆盖当前主线。共享挂载时各节点使用同一目录；非共享时可用各机相同字符串的本地目录。
  全局rank0写checkpoint/汇总，各机各rank写自己的预测；后续完整评估须分发rank0权重并按需要汇集文件。

## 操作顺序（未来需要启动时执行）

在每台机器分别进入固定 commit 的 PLOT 运行目录：

```bash
cp experiments/m1/recipes/multinode/cluster.env.example experiments/m1/recipes/multinode/cluster.env
# 编辑cluster.env，填内网MASTER_ADDR及本机路径，核实GPU和网卡。
# 以下rank 0在另外四台分别改为1、2、3、4。
bash experiments/m1/recipes/multinode/run_node.sh experiments/m1/recipes/multinode/cluster.env 0 dry-run
bash experiments/m1/recipes/multinode/run_node.sh experiments/m1/recipes/multinode/cluster.env 0 check
```

check仅本地读取，不联系其他机器；加载checkpoint元数据、检查GPU，并对全部缓存/权重作SHA256，因此可能需要数分钟且占用磁盘带宽。先不要在现有训练最忙时无意执行。

各台同时在终端启动smoke（不训练、不写模型），确认rank0输出PASS，再启动train：

```bash
bash experiments/m1/recipes/multinode/run_node.sh experiments/m1/recipes/multinode/cluster.env 0 smoke
# 五台smoke全部退出后，再在各台启动：
bash experiments/m1/recipes/multinode/run_node.sh experiments/m1/recipes/multinode/cluster.env 0 train
```

train也会先做跨节点数据、模型、代码、关键配置一致性校验及NCCL all-reduce验证。
这些命令在前台运行，建议放在各机tmux会话中。不会自动SSH、不停止任何已有作业、不自动重启失败训练。
不要在相同GPU上与原全量任务直接重叠启动；正式迁移时另行安排资源和原任务停机。

## batch、步数和恢复语义

默认5节点×8GPU×每卡8样本=全局batch320，无梯度累积，学习率沿用checkpoint，不自动乘5。
这不是与原batch64完全等价的续训：同一总step意味着更多曝光，但更新数的学习效果不能线性替代。
MAX_STEPS=1000000包含checkpoint已有步数，不是额外再跑100万步。
扩大world后每轮更新数会改变；仅接受epoch边界checkpoint，避免旧batch offset在新分片下跳错样本。
恢复优化器动量，但随机数/样本序列随world改变，不是逐位恢复。

默认AUDIT_SAMPLES=80，必须>=总GPU数且<=数据量；保持原32样本会让部分rank拿到空输入。
80窗口评估不能与旧32窗口绝对分数直接比较。FIXED_LOSS_SAMPLES=128。
现有DistributedSampler在46200不能整除40时会补齐到46240条，含40条重复样本/轮，约0.087%。

## 本次验证范围

已做bash语法、五rank命令生成、参数拒绝路径、Python静态检查与编译。
后续已完成五机SSH/TCP测试；没有实际五机NCCL测试，不能据此确认训练吞吐。

参考：[PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run)、[多节点DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode)。
