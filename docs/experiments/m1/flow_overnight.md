# 原 flow 全量长训练：1000k 上限

用户授权全量窗口长时间训练，保留原 flow 架构，不使用投影连接或直接预测。

- 输出：`outputs/m1_flow_full_1000k_v1`。
- 输入：全体46,200窗口，两名玩家图片＋GT相机；原train/val_id/test_id均已授权参与记忆训练。
- 起点：原全量flow第20轮checkpoint，step14,440；恢复AdamW状态，lr保持1e-4。
- 上限：累计1,000,000步（包含已有14,440步），等效全局batch64，8卡。
- 每轮722次更新，覆盖全部窗口一次；每轮保存模型和优化器、固定128窗口loss、固定32窗口几何审计。
- 按当前步数预算推算总训练到约第1386轮；最后一轮在精确步数上限停止，记录batch_in_epoch。
- 后台独立session，关闭当前对话不会主动停止进程。没有设置按时钟自动停止，1000k是上限而非一晚跑完的承诺。
- `launch.json`记录PID和启动时间；`run.log`为实时输出。训练结束以`training_complete.json`为准，不生成虚假的完成报告。

启动入口（已启动，不要重复执行）：

```bash
bash experiments/m1/recipes/flow_overnight.sh
```

检查最近训练记录：

```bash
tail -n 5 outputs/m1_flow_full_1000k_v1/train.jsonl
tail -n 5 outputs/m1_flow_full_1000k_v1/epoch_loss.jsonl
```

需要在未来恢复时，在入口后附加
`--resume outputs/m1_flow_full_1000k_v1/checkpoint_latest.pt`，会覆盖原始resume路径。
若checkpoint在轮中，按保存的batch偏移继续数据顺序；没有保存每卡随机数状态，不保证逐位重放噪声。
`--max-steps`覆盖`--epochs`，包含checkpoint已有步数。

验证：6项M1全量入口回归测试通过。另用原checkpoint只续训1步的8卡实跑，准确停在14,441步，
保存epoch20、batch_in_epoch1。代码与启动脚本快照见输出目录source_snapshot。
