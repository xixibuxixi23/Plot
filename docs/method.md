# PLOT 最新方法规范

> 状态：当前唯一有效的方法口径
> 更新：2026-09-08
> 代码数据口径：`../textagent`
> 历史 `method_v*.bak`、旧三网络说明和旧中心推断规则均不再作为实现依据。

## 1. 目标与系统边界

PLOT 是一个多居民的可编辑体素世界模型。真人、文字建造者、村民、僵尸和骷髅都占用
相同的居民槽位，拥有自己的第一人称画面、动作流和状态行。引擎只用于采集数据和重放
评测；纯模型推理从外部提供的首帧 RGB 与必要元数据开始，此后不调用引擎。

系统由一个无界稀疏世界记忆和四个分别训练的模型组成：

| 模型 | 名称 | 作用 | 空间/时间口径 |
|---|---|---|---|
| M1 | Fill Network | 首次填充记忆中尚不存在的体素 | 48³ 模型窗口 |
| M2 | Write/Transition Network | 根据联合动作预测居民运动与稀疏写事件 | 居民局部 13³，8 步双向块 |
| M3 | Renderer | 从记忆和角色状态生成每个居民的第一人称画面 | 48³，8 帧因果输出，长历史缓存 |
| M4 | Inhabitant Policy | 从居民自己的画面产生未来动作 | 最近 8 个已完成画面，输出未来 8 步 |

四个模型编号以后不再改动。论文宏、训练配置和实验表都必须沿用这一编号。

## 2. 世界记忆与最小状态

世界记忆在全局坐标中保存：

```text
WorldMemory
  voxels : 无界稀疏体素映射；不存在与 air 是两个不同状态
  chars  : resident_id -> CharRow
  ledger : 按提交顺序追加的 WriteEvent
```

`CharRow` 只保存推理确实需要且不能从其他字段推出的状态：

```text
position_xyz, velocity_xyz, yaw, pitch, hp, held_item, appearance_id/type
```

- 居民始终位于地面运动，本轮不单独预测飞行状态。
- `velocity_xyz` 保存最近一次 transition 的世界坐标位移；M2 在块边界直接读取，
  并随每帧状态提交显式更新，不由下游模块临时回看位置历史重算。
- `hp > 0` 已经表达存活，不再重复保存或预测 `alive`。
- `active_agent_mask` 只表示槽位是否存在/是否为 padding，不表示存活。
- 不保存 attack cooldown、last-attack time 或 knockback 状态作为模型输入。
- 不设置显式击退输出；攻击写入只负责目标和血量效应，后续位置由运动预测隐式吸收。
- `held_item` 必须保存；武器纹理不作为单独条件输入。

每个非空写事件为：

```text
(transition_index, within_transition_order, source, target_kind, target, payload)
```

目标是体素时 payload 为新 block id；目标是居民时 payload 为 `delta_hp`。没有产生世界写入的
动作对应 null。原始数据保留同一 transition 内的全部事件；“每居民每帧至多一次写”是 M2
的建模约束，不得反过来删除采集真值。

## 3. 初始化、无界扩展与坐标约定

推理时外部提供首帧 RGB 及其必要的居民/相机元数据；采集到的 49³ 引擎观测是 M1 的训练真值，
不默认作为模型输入。这样可与 γ-World、Solaris 使用相同的画面初始化权限。另设
state-initialized 赛道时可以直接给体素初态，但必须单列结果。

初始化时，M1 填充所有居民 48³ 窗口的并集。居民移动后，只对第一次进入范围、在全局记忆
中仍为 `UNFILLED` 的区域调用 M1；已有体素只能由 M2 的写事件修改。全局记忆没有固定边界。

采集与模型裁剪必须分开：

- 公共原始数据逐帧保存每个居民的 49³ 体素观测及真实 `obs_voxel_center`。
- 48³、13³ 和 8 帧窗口均在后处理阶段派生，不写死进公共采集格式。
- `obs_voxel_center` 仅用于把数组索引映射回全局体素坐标，不作为任何模型的语义输入。
- 不从连续 player XYZ 猜测引擎 crop center，也不使用经验 hysteresis。
- 若需要共享 crop，只能从当帧真实使用的各 `obs_voxel_center` 构造，不能先平均连续玩家坐标。

纯模型 rollout 每个 8 步块开始时选择一个全局 crop anchor，并在这 8 步内固定；居民可以在
窗口内部移动。块边界需要换 anchor 时，从无界全局记忆重新裁剪，重叠内容按全局坐标复制，
不得通过平移画面或重置局部网格实现。

相机位置与玩家位置是不同量，但不手工加入“眼睛高 1.6 格”的常数。模型使用
`camera_world - player_world` 的连续相对编码，或直接预测世界坐标相机；整数 crop center
不参与相机还原。这样 center 相差一格不会导致整幅画面跳变。

## 4. M1：Fill Network

M1 只处理从未生成过的体素，不负责玩家挖放后的修改。

**输入**

```text
voxel_context : 48³，已有 block/air 与 UNFILLED 明确区分
fill_mask     : 本次必须首次生成的位置
first/recent frames and player-relative camera poses of relevant residents
```

初始化时相关居民为全部居民；移动触发局部扩展时只使用能观察到该新区域的居民。

**输出与训练**

M1 对 `fill_mask` 中的体素直接输出类别 logits，用交叉熵训练。本方案不用离散扩散。训练 mask
包含初始全空、轨迹首次发现和已知区域边界扩展三类。一次生成后写入全局记忆，之后禁止 M1
覆盖同一体素。

## 5. M2：Write/Transition Network

M2 的输入输出均按 8 步块组织，但每个居民在每个 transition 只有一个 query 和至多一次写。

**输入**

```text
local_voxels        : 以本块居民参考位置映射出的固定 13³ 区域
initial_char_rows   : 所有相关居民的最小状态
joint_actions       : 8 × A × 23；整个 8 步动作块均已知
previous_frame_feat : 每个居民在块开始前最后一个已完成画面的特征
held_item           : 已包含在居民状态中
```

13³ 始终位于采集的 49³ 观测或当前全局记忆窗口中，直接按真实全局坐标映射，不增加所谓
“几何余量”。M2 不需要当前 transition 尚未提交的其他居民预测结果；所有 query 从同一已提交
状态读取，再按固定居民顺序仲裁冲突写入。

**结构与输出**

- 8 步 query 之间使用双向注意力，因为未来 8 步动作在调用 M2 时都已知。
- M2 不使用 KV cache。
- 位置和朝向先由按键产生一个粗略运动学 proposal，主干只预测 residual。
- Player dynamics 分支额外读取每个居民自己的 `7³` 紧凑局部体素，并在每层通过
  cross-attention 注入；长 rollout 中体素世界坐标必须相对预测位置重新对齐。
- 几何分支从零影响初始化并独立微调，已收敛的速度/运动主干保持冻结，避免短期几何
  拟合破坏长程动力学。
- proposal 只包含运动学，不手写碰撞、攻击、击退或 cooldown 规则。
- 每个 query 输出连续 self-update residual、统一地址 pointer 和按目标类型选择的 payload。
- 地址候选固定为 13³ 体素、可见/邻近居民以及 null；事件坐标随后映射回全局记忆。
- 方块 payload 用类别交叉熵；居民 payload 只监督血量变化；本方案不使用离散扩散。

同一步多个居民的结果并行预测，提交阶段才按稳定 slot 顺序处理竞争写。M2 永远不能读取
“同一 transition 中排在自己前面的模型输出”，否则训练会依赖任意执行顺序。

## 6. M3：Renderer

M3 为每个居民独立渲染，但所有居民读取同一份世界记忆。

**条件**

```text
48³ memory crop / camera-space voxel projection
target resident position and player-relative camera
target action and held_item
other residents' projected positions, hp, appearance and event cues
target resident's own causal frame history
```

居民外观输入使用 episode 中 `players/agentN/{front,back,left,right}.png` 四张标准视图。entity-ID
mask、武器 mask 等只用于局部监督与评测，不作为 M3 条件。武器外观由 RGB、`held_item` 和
动作共同学习，不输入武器纹理图。

M3 是因果视频模型，推理一次输出 8 帧并保留 KV cache。训练使用最长约 65 帧的连续因果
片段，在其中选择多个锚点监督未来 8 帧；每个目标帧只能看到它之前的画面。推理时每轮只
新生成 8 帧，但可以通过缓存读取更长历史。训练不能让某个未来帧看到块内更晚的真值画面。

## 7. M4：Inhabitant Policy

M4 与 M3 的视觉分支耦合，读取 M3 后部的只读画面特征，不修改渲染 token。每次调用使用
最近 8 张已经完成的第一人称画面和对应历史动作，预测未来 8 步统一 23 维动作。训练样本在
连续轨迹上按每个合法锚点构造，不要求原始数据预先切成 8 帧文件。

具体实现沿用 2DAction：在冻结 M3 的最后四个 DiT block 后插入 actor-token 分支，依次读取
该帧全部 detached spatial patches、做8帧 causal temporal attention、读取 frozen T5
current text，再经 MLP 更新。最后一个 actor token 通过按键 BCE、none+9 hotbar categorical
以及水平/垂直各17档 mouse categorical，一次输出整个8步 chunk。推理完整执行这8步后才
再次调用 M4；M4 不维护 KV cache。训练时 M3 对最多65帧完成历史做 causal forward，M4 仅
在其中由 `policy_indices` 指定的连续8帧上建立 token。推理时直接使用 M3 KV commit 保存的
最后四层空间特征，不把最近8帧脱离长历史重新编码。

M4 只使用居民在部署时可得到的信息。文字建造者额外读取总任务、角色任务和当前子任务；
环境居民不读取为审计生成的 behavior text，也不读取全局体素或其他角色真值表。

策略参数按行为能力分族，族内再使用 Profile：

| 参数族 | Profile | 角色与约束 |
|---|---|---|
| `language_builder` | `language_builder` | 文字条件；允许移动、选物、挖放 |
| `villager` | `villager_peaceful` | 游荡、观察、等待、逃跑；禁止挖放和攻击 |
| `combat` | `zombie_melee` | 直接追击；剑或斧 |
| `combat` | `skeleton_swordsman` | 持剑，近战横移，过近后退 |
| `combat` | `villager_defender` | 守卫区域、拦截 hostile、返回 |

僵尸和骷髅共享 combat 参数，通过 Profile/type modulation 区分；和平村民使用另一套参数；
防守村民按能力路由到 combat。只有在共享参数出现明确负迁移时，才增加小 adapter/LoRA，
第一版不为每种外观复制一套完整网络。

`behavior_routes` 只在 episode manifest 记录一次，作为审计和后处理路由；不在逐帧 tensor
重复。角色外观、物理种类、控制来源和行为能力必须分开记录。

## 8. 居民实现口径

- 僵尸、骷髅、村民都是 player-shaped agent，不使用无相机、无动作流的原生 engine mob。
- 骷髅与僵尸具有相同数据接口，但 mesh、纹理和动画区间不同。
- 骷髅固定持剑；僵尸的剑/斧负载按 episode seed 独立平衡。
- 和平村民绝不产生 dig/place/attack；validator 同时检查动作流和事件流。
- 每个 episode 为每个居民保存正面、背面、左侧、右侧四张外观图；不保存展开 UV 图作为
  模型输入，也不保存武器纹理图。
- entity-ID render pass 使用稳定 `uint16` 实例 ID。mask 是监督信号，可用于角色/武器区域
  加权以减轻模糊，但不进入模型输入。

## 9. 公共数据与时间对齐

公共数据保持连续、同步和模型无关：

```text
observations: 0 ... T       共 T+1 个
actions:      0 ... T-1     共 T 个
transition t: observation_t -> action_t -> events/state_{t+1} -> observation_{t+1}
```

采集引擎已经同步多个客户端，不额外增加“同步层”。每个事件必须有 `transition_index=t`、
`observation_frame=t+1`、server tick 和 tick 内顺序。初始化事件单独标为 observation 0。

每个 observation 保存所有居民的 RGB、49³ 体素与真实 center、连续 player/camera 状态、HP、
库存/held item、实体表和实例 mask。动作保存语义按键与连续鼠标。训练所需的 48³/13³、
8 帧块和约65帧因果片段全部由发布后的后处理 adapter 构造。

数据不重复表达同一事实：HP 不再配 alive；center 不进入相机编码；武器已有 held item 就不再
复制纹理条件；mask 已有整数 ID 就不从预览 PNG 恢复标签。

## 10. 闭环推理

```text
0. 外部提供首帧 RGB 与居民/相机元数据；M1 填充初始 48³ 窗口并集。
1. M4 为模型控制的槽位依据最近 8 个已完成画面生成未来 8 步动作；真人动作外部提供。
2. M2 读取当前记忆、上一画面特征和完整 8 步联合动作，预测 8 步 self-update 与稀疏写。
3. 每步按固定 slot 顺序提交写入；更新全局记忆和账本。
4. 若新 48³ 范围出现 UNFILLED，M1 只填充这些新体素。
5. M3 因果生成每个居民未来 8 帧并更新缓存。
6. 进入下一轮；无论 NPC 是否出现在真人画面内，都不能冻结它。
```

回放评测使用记录动作，只测 M1/M2/M3；生成评测让选定槽位由 M4 接管。生成动作必须保存，
并从同一首帧在引擎中重放，分别报告模型内部后果与真实可执行后果。

## 11. 已定案与尚未定案

已定案：四模型编号；48/13 空间口径；M2 与 M3 八步；M2 双向无 cache；M3 因果有 cache；
首帧 RGB/必要元数据外部提供；无界稀疏记忆；camera-player 相对编码；center 非模型输入；运动学 proposal 加
残差；无显式 knockback/cooldown/alive；直接分类而非离散扩散；公共数据连续 49³；mask 仅
监督；四视图皮肤输入；全部 NPC player-shaped；按能力族路由 M4。

尚未定案的内容不得在论文中写成事实：各模型宽度/层数、约65帧的最终训练上限、损失权重、
M1 三种 mask 的最终采样比例、M4 是否需要 adapter、正式数据规模与所有结果数字。
