# M2 实现与训练

更新：2026-09-10。结构讨论见 [transition_design.md](transition_design.md)。
S01 编辑模型已经完成八卡、10,000 step 的 ordered-event 正式训练；结果见
[`m2_s01_ordered_slots_v8/experiment_report.md`](../outputs/m2_s01_ordered_slots_v8/experiment_report.md)。

## 实现

`plot/models/transition.py` 包含共享 3D CNN、时间/居民双向注意力、局部记忆 cross-attention、
运动残差、统一地址和 typed payload。无 KV cache，不读取块内未来状态或其他 query
已经解码的结果。2、3、4 人可以混 batch，null 地址随 padding 后的 A 正确重映射。

编辑事件采用窗口级计数和有序槽：每位居民先预测八步窗口内有几个成功编辑，再由
固定槽分别预测时间类别、体素地址、dig/place 和结果方块。槽按真实发生时间监督，
每个槽只能落在一个 transition 上；这取代逐帧独立 occurrence 作为正式推理路径，
避免同一成功编辑在相邻帧重复输出。时间 CE 给相邻 transition 少量软标签，因此总
loss 有非零下限，评估同时报告严格同帧和 +/-1 transition 的完整编辑指标。

玩家动力学另有隔离分支，只读初始玩家/相机状态、居民类型、手持物和八步联合动作，
通过时间与居民间注意力预测八步位置、yaw/pitch、相机和手持物。玩家专训不读取
RGB/体素，也不更新事件编码器与事件头。S02–S10 的八卡训练结果见
[`m2_s02_s10_player_v1/experiment_report.md`](../outputs/m2_s02_s10_player_v1/experiment_report.md)。
独立测试包含 human-like、villager、zombie 和 skeleton，并分别报告误差。

附加输出为 camera-relative 位移、相机方向和辅助下一帧 HP。辅助 HP 只算 loss；
实际 HP 只由写事件累加，避免重复扣血。`plot/pipelines/transition_pipeline.py` 提供稳定
slot 顺序提交及每步快照 callback，后写体素覆盖先写，攻击 payload 对同一目标求和。
`WorldMemory.read_region` 支持奇数尺寸 13³，原有 M1 的 48³ tile 接口保持原约定。

初版视觉支路直接对最后完成的 RGB 做小 CNN 并保留 4×4 tokens；**尚未切换到
设计中建议的冻结 VAE/M3 特征**。默认网络 256 宽、6 层；小规模 smoke 使用 64 宽、
2 层、4 heads。粗运动步长与鼠标 gain 是显式可调参数，尚未完成采集协议校准。
当前不实现通用库存变化模型。固定 loadout 的 held item adapter 支持选物；drop
之后标为不支持，必须查看真实标签一致性，不能直接宣称任意库存下闭环正确。

## 真实数据的处理

- 输入为一个 observation 的各居民 13³ 体素/最小状态/图像，加后续八步
  `action_continuous`；目标为之后八个 observation 和对应有效事件。
- 事件 Lua XYZ 转数组 ENU 使用 `[x,z,y]`，与当前 TextAgent collector 一致。
  不使用其他项目中的 `[x,-z,y]` 约定。
- 正式 slot 顺序、事件时序取 manifest 与 `transition_index/observation_frame`；
  不将 attack attempt/contact 当成成功伤害。
- 事件 `damage` 可能是名义伤害。例如记录 damage=2 时，HP 可能只减少 1。
  单来源事件用当步观测的净 HP 差作为 payload，**这依赖同帧没有其他 HP 因素的假设**；
  多来源同目标事件只保留地址监督，屏蔽不可辨识的 payload。
- 同居民同一步多写、未知来源、候选越界、无观测扣血等有单独计数与有效标签 mask。
  它们不自动变成 null。原始日志不修改。
- block payload 仅在该 transition 该位置只有一次已记录写入、且下一 observation
  覆盖该位置时由真实类别监督；否则屏蔽 payload，避免把最后一次写冒充每次写。
- 旧 item 字典缺少 iron sword；固定战斗 loadout 用 episode `agent_weapons` 元数据
  补全。观察标签仍来自 observation-aligned entity table。
- 缓存审计还统计扣除可归因攻击后的 HP 变化。本次有 13 个有效窗口居民步出现
  正 HP 差，攻击写入无法解释；它们可能涉及恢复或记录时序，需要独立处理。
  auxiliary HP 能接受这些标签，但当前 authoritative HP 提交路径尚不覆盖这类变化。
- 缓存字典保留传入 M1 字典的 class 顺序并追加新 raw IDs。与 M1/M3 联动前必须
  统一使用这个扩展后的字典，不能直接混用旧 embedding/checkpoint 的类别范围。

公共数据读取一次后缓存小窗口，训练不重复解压巨大 `data.npz`。缓存只接受 usable
train episode。pilot 按每个场景的最后一个选中 episode 做内部留出，不把相邻窗口
随机拆成训练/验证；这不是最终 world-disjoint benchmark。

## 使用

```bash
cd /path/to/Plot
.venv/bin/python dataset_toolkits/build_transition_cache.py \
  --root ../textagent/data/batches/s02_s10_v1_mojang_skins_20260908/train \
  --output outputs/m2_pilot/train_cache_v2.pt \
  --vocabulary derived/common/block_vocabulary.json \
  --episodes-per-scenario 2 --windows-per-episode 8

.venv/bin/python experiments/m2/train_transition.py \
  --cache outputs/m2_pilot/train_cache_v2.pt \
  --output-dir outputs/m2_pilot/run_20steps_v2 \
  --width 64 --depth 2 --heads 4 --steps 20 --batch-size 2 \
  --save-every 10 --device cuda:4 --wandb-mode offline

.venv/bin/python -m pytest -q tests
```

默认缓存场景为 S02/S06/S08/S09/S10，各最多两个 episode。窗口采样有意保留非空
事件与普通动作；样本统计不是全量数据分布估计。脚本提供 W&B online/offline/disabled，
默认 offline；本次试跑不上传云端。日志同时写入 `metrics.jsonl`，每次验证保存轨迹、
HP 和写类型对比 PNG 及精确 pose/address NPZ，并保存 checkpoint/config。

只看 address accuracy 会被 null 淹没，必须同时看非空事件 precision/recall。
单项 loss 是逐 batch 有效标签均值，验证中的 loss 汇总为 batch 均值；事件指标则
累加样本计数后求比率。HP 图中的虚线是 auxiliary head，不是 ledger 推演的 HP。

## 范围和后续

数据目录的只读事件盘点在 `outputs/m2_pilot/collection_event_inventory.json`；
缓存监督审计在对应 `train_cache_v2.json`；最终本轮结论见 `outputs/m2_pilot/pilot_report.md`。
最初 `run_20steps` 使用了错误的事件坐标符号，已标为 INVALIDATED，不作为有效结果。

正式训练前还需要：扩大有效攻击/方块写样本；校准粗运动；处理 null 不平衡；
审计环境 HP 与记录时延；解决 held item 与观测的残余不一致；接入冻结视觉特征；
在完整多块 rollout 中连接 M1/M2/M3。当前提交器与快照接口已经实现，但并未宣称
完整闭环、正式质量或更大网络显存已验证。

## S01 edit-only pilot

Use `--objective edit` to optimize address CE plus candidate-conditioned block CE.
Other state heads remain present but receive no direct loss; their outputs are not
trained motion/HP predictions. This mode rejects caches containing valid resident
writes. `--null-weight` controls null-query weighting (default 0.1); report exact
non-null address recall and joint address+block `edit_recall`, not only overall
address accuracy. `block_payload_accuracy` uses the ground-truth address and is
therefore a conditional diagnostic, not end-to-end editing accuracy.

```bash
.venv/bin/python dataset_toolkits/build_transition_cache.py \
  --root ../textagent/data/batches/s01_v1_mask_20260907/train \
  --output outputs/m2_s01_edit/train_cache.pt --scenarios S01 \
  --episodes-per-scenario 8 --windows-per-episode 16 \
  --vocabulary derived/common/block_vocabulary.json
.venv/bin/python experiments/m2/train_transition.py \
  --cache outputs/m2_s01_edit/train_cache.pt \
  --output-dir outputs/m2_s01_edit/run_100steps \
  --objective edit --null-weight 0.02 \
  --width 64 --depth 2 --heads 4 --batch-size 2 \
  --steps 100 --save-every 50 --device cuda:4 --wandb-mode offline
```

The S01 source is the existing construction collection, distinct from the newly
collected S02–S10 `noregen` batch. Windows with contradictory voxel observations
across residents are excluded and counted as `skipped_inconsistent_voxel_windows`;
no arbitrary resident wins. Evaluation holds out a complete episode from the
training source, with no claim of world-disjoint generalization. W&B is offline;
local plots show edit addresses and initial RGB for this objective.


## Full S01 joint training without HP

The 10,000-step run uses `train_scripts/train_transition_full.py`, all 21,000 accepted training episodes and independent `val_id` (1,050 episodes), with lazy per-episode caches. It supervises geometry, edits and held items; HP/damage heads are frozen. See [run configuration and evaluation policy](../outputs/m2_s01_full_nohp_10k/README.md).
