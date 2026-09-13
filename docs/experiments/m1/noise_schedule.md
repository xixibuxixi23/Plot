# 原 flow：推理步数与高噪声采样对照

本次已完成。推理步数增加无收益。3轮对照的64窗口纯噪声召回：原分布37.60%，
高噪声41.87%；但可见树干召回2.25%→2.04%，未改善关键树木问题。保留主线原配置。
见 [完整报告](../../../outputs/m1_noise_schedule_compare_v1/experiment_report.md)。

输出 `outputs/m1_noise_schedule_compare_v1`；共同来源保存在 `source_checkpoint.pt`，
包括模型与优化器。原百万步长训练独立继续，实验不覆盖其权重。

先用同一快照在64个固定窗口上比较20/50/100步Euler，噪声种子1234+全局索引。
结果分别40.54%/40.36%/40.27%召回，暂不支持增加推理步数。

续训对照：全部46,200窗口，两组各3轮；保持batch64、lr1e-4、Adam状态与原flow结构。
原t分布sigmoid-normal；高噪声组每样本50%概率替换为Uniform(0.8,1)。
替换随机数独立于全局随机数，因此两组体素噪声和基础时间采样相同。

训练入口新增 `--time-sampling original/high_noise`，默认仍original，不影响已在运行的长训练。
共同resume路径是实验source_checkpoint，目标epochs=518；output分别original与high_noise。
每轮固定128窗口速度误差诊断、固定32窗口几何审计。不同采样分布的训练loss不直接比较。

最终用 `scripts/evaluate_noise_schedule.py --output <组目录>` 评估同一64窗口的纯噪声生成，
以及t=0.5带噪真值还原；后者仅为诊断，不是从照片重建。另记录可见树干和树叶分项。
`scripts/report_noise_schedule.py` 汇总报告，要求两组训练和最终评估均完成。

本轮只做短期单种子验证；无论结果如何，不直接改变原百万步任务。
