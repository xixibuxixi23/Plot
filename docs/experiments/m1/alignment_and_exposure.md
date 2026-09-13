# M1 对齐核验与1024窗口充分训练

用户授权2026-09-11同时测试两项，原百万步训练保持运行。

1. 独立CPU世界坐标DDA投影、原始data.npz/视频与缓存比对已完成8窗口16视角。见[对齐报告](../../../outputs/m1_raw_alignment_v1/experiment_report.md)。未发现整体坐标/FOV或明显帧错配；原生图片缓存为320×180，输入VAE前放大到640×360。小样本核验不等于全量正确，特殊植物的整方块代理也不同于真实渲染。
2. `outputs/m1_flow1024_exposure1250_v1`：原flow、固定1024窗口、batch64、20,000更新、额外曝光1250次，从与旧1024实验相同的全量第20轮checkpoint开始，AdamW重置、lr1e-4。仅延长训练预算，不改图片缓存、网络或loss。

训练入口`experiments/m1/train_projected_flow_compare.py --mode flow --steps 20000`。
launch.json记录后台PID/命令。每1000步在全部1024窗口评估并保存预测，最后写completion.json。
后台`scripts/finish_m1_flow1024_exposure.py`等待训练完成，再计算visible_tree_metrics.json并生成experiment_report.md。失败写monitor_error.txt或monitor.log；不以启动成功代替实验完成。

256窗口对照也是每窗口1250次曝光，但窗口集合和随机种子不同。因此本轮可以判断延长1024训练是否明显改善，不能将两种数据规模的差异全部归因于单一机制。
