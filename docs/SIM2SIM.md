# 执行器约束与跨引擎迁移

当前目标是无真机条件下的可检验迁移研究。Genesis 用于训练与原有评价；独立 MuJoCo 用于模型外检验。跨引擎成功不等于真实机器人成功。现有 PPO 尚未获得可靠行走，本轮小网络烟测只验收接口。

## 已实现的边界

| 模块 | 实现 | 限制 |
|---|---|---|
| 执行器 | 可选显式 PD，读取随机化后的实际增益，使用原力矩上限 | 不是已辨识的 G1 电机模型 |
| 命令延迟 | 整数物理步延迟；每机器人共享延迟；每次 reset 重新采样 | 不包含传感延迟、抖动、丢包、带宽动态 |
| 速度包络 | 可选 `no_load_speed_rad_s`，随绝对转速线性收缩对称力矩范围 | 保守压力测试，包括制动；不等同真实四象限电机曲线，默认关闭 |
| 观测 | `proprioceptive` 删除 actor 的基座线速度真值，平地 96 维；critic 保留 109 维 | 重力方向仍来自仿真姿态；尚无真实 IMU 姿态/速度估计器 |
| 控制契约 | 关节顺序、home、增益、限幅、观测顺序、坐标系、周期、策略哈希 | flat H1/H2/H3；CPU portable runtime；无 SDK2 控制器 |
| 共用控制 | Genesis 和 MuJoCo 使用同一 `process_action`，保留反馈残差和 shield | shield 仍只是名义 PD 代理，存在延迟时不保证实际未来力矩或状态安全 |
| MuJoCo | 固定同一源 MJCF 哈希、显式力矩控制、真实外力、完整 episode 与失败记录 | 不支持 rough terrain 或物理 DR；显式拒绝，禁止静默降级 |

`timing.physics_dt_s` 是每次外部 PD 更新/Genesis scene.step 周期，`substeps` 是求解器内部子步数。延迟不按内部子步计数。`control_dt_s` 是策略周期，必须是物理周期整数倍。旧配置保持 20 ms / 20 ms / 2 substeps；新配置使用 2 ms / 20 ms / 1 substep，不能将两者结果直接归因于执行器类型。

外力在每个外部物理步重新施加，避免 Genesis 清除外力后，decimation > 1 时实际冲量缩小。`ghrl smoke transfer_push` 检查 2 ms 物理周期、20 ms 控制周期下的物理步累计冲量。

GenesisLab 在 reset 时也会 write actuator；本项目以实际物理步计数器门控延迟队列，避免 reset 消耗延迟步、部分 reset 干扰其他环境。显式 PD 读取 live gain，因此启动时的 per-env DR 对显式控制同样生效。

## 运行

在项目独立环境增加可选依赖：`pip install -e '.[sim2sim,test]'`。第三方模型仍按 NOTICE 保留其许可，不将网格复制进本仓库。

```bash
# 新显式执行器 + 0–10 ms 延迟 + per-env gain DR 的真实接口烟测
ghrl smoke transfer --output outputs/transfer_smoke.json

# 无延迟显式 PD，实际 Genesis 完整 episode
ghrl evaluate --task H1 --task-config configs/tasks/h1_transfer.yaml \
  --agent zero --episodes 8 --num-envs 2 --device cpu --seed 1200 \
  --output outputs/transfer_genesis

# 同一控制配置，独立 MuJoCo，禁止自动调参
ghrl sim2sim --task-config configs/tasks/h1_transfer.yaml \
  --episodes 8 --seed 1200 --output outputs/transfer_mujoco

# 新观测配置必须重新训练，不能直接加载旧 99 维 actor
ghrl train --task H1 --task-config configs/tasks/h1_transfer.yaml \
  --ppo-config configs/algorithms/ppo_smoke.yaml --num-envs 2 --device cpu \
  --output outputs/transfer_training
ghrl export --checkpoint outputs/transfer_training/model_final.pt \
  --output outputs/transfer_export
ghrl sim2sim --bundle outputs/transfer_export --episodes 8 \
  --output outputs/transfer_export_mujoco

# 仅生成 36 项 PD 诊断计划，不自动启动昂贵实验
ghrl transfer_plan --task H1 --episodes 8 --seed 1200 --output outputs/transfer_plan
ghrl transfer_summary --plan outputs/transfer_plan/plan.json --output outputs/transfer_summary.json
```

MuJoCo 默认读取 GenesisLab 的固定模型缓存；可用 `--mjcf` 指定已下载原 XML。模型哈希必须与训练资产一致。原 XML 内部 actuator 被移除，通过纯外加关节力矩驱动；设置审核快照的 armature，添加地面，积分器为 implicitfast。**显式 PD 控制不等于显式积分器**：更换 PD 实现与更换积分器必须分开研究。

导出仍包含原始 `policy.pt`（内含归一化），平地任务额外生成 `contract.json`；必须通过 PortableController 做动作处理。不能将 actor 单独发送到硬件并声称部署一致。H4 导出保留旧行为，元数据明确不提供 flat portable contract。

## 评价可比性

Genesis 评价协议升级为 v3，哈希包含执行器类型、延迟、速度包络、物理周期与 decimation，防止将不同物理条件误判为配对实验。原始 episode 时间序列、失败与版本均保留。

当前两引擎的接触求解器、摩擦语义、几何导入与 reset RNG 不完全相同；MuJoCo 按 episode 顺序采样初始关节姿态，Genesis 按 batch 采样。相同 seed **不表示相同初始状态**。因此 MuJoCo 输出暂不送入 `ghrl compare` 的严格配对统计；只能作描述性比较。下一阶段需要冻结初始姿态/速度及指令库，再按 episode ID 重放。MuJoCo 也不是真实世界的替代真值。

MuJoCo 饱和占比采样全部外部执行器更新；Genesis 新增 `actuator_last_step_saturation` 仅采样每个策略步最后一次执行器更新，不应直接比较这两项统计。

## 研究主线：可迁移增益区域与策略协同设计

**假设 H1**：相较单一模型上选取高回报增益，在延迟和执行器约束下筛选增益区域，可以减少固定策略跨模型性能退化。当前只是待验证假设。

固定姿态 PD 的位置目标始终等于 home；这里的命令传输延迟不会延迟内层 PD 的本地状态反馈。因此，仅用 zero agent 扫描延迟不能估计闭环策略的延迟容忍边界。它用于验证接口和静态基线；动态延迟实验必须使用反馈位置控制器、动作激励或收敛策略。

`transfer_summary` 根据完整计划筛选离散经验增益候选：任何缺失/失败工况都不能通过，不在离散增益之间插值。这个筛选器是研究工具，不作为新算法或安全证书。

1. 固定 50 Hz 策略周期，扫描物理周期 2/5/10 ms、固定延迟 0/10/20 ms、刚度倍率 8/16，阻尼倍率取刚度倍率平方根。两引擎共 36 组；先用 PD 作诊断，再评价收敛策略。
2. 单独做隐式/显式驱动对照，固定物理周期、内部子步与力矩约束；再单独做积分收敛检查。不得同时改步长、驱动、增益后将变化归因于单一机制。
3. 在开发模型上寻找满足任务与执行器约束的**经验可行区域**，将策略训练限制在该区域并对区域内误差随机化。区域不是全局稳定性或不跌倒证书；只在验证过的范围成立。
4. 对照：充分调参的固定 PD + PPO、普通 DR、历史观测适应、提出的区域约束方法。统一网络规模、训练预算、5 个独立训练种子与未参与调试的测试集；先获得收敛行走，再比较行走能力。
5. 指标：成功率、速度误差、跌倒率、能耗、饱和、延迟容忍边界、跨引擎下降；区分策略间差异和物理模型差异。小样本站立不能外推到抗扰、地形和行走。

| 论断 | 需要的证据 | 不成立时如何处理 |
|---|---|---|
| 高增益收益可迁移 | 固定约束下的延迟/步长收敛及独立引擎实验 | 若只在一种数值配置成立，记录仿真敏感性 |
| 区域约束优于普通 DR | 未见模型、匹配训练预算、独立种子与消融 | 不优于充分调参基线，则不宣称方法创新 |
| 对 sim-to-real 有帮助 | 后续真实执行器辨识与封存真机测试 | 目前只报告 sim-to-sim/部署准备 |

### 现有方法与创新边界

| 来源 | 已有核心思想 | 对本项目的意义 |
|---|---|---|
| [Isaac Lab actuators](https://isaac-sim.github.io/IsaacLab/main/source/concepts/actuators.html) | 延迟、转矩—转速等执行器建模 | 基础能力，不作为创新 |
| [mjlab](https://github.com/mujocolab/mjlab) | MuJoCo-Warp 训练与机器人任务 | 第二引擎/成熟基线参考 |
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) | 训练、MuJoCo 与硬件部署链 | 后续 SDK2 接口参考 |
| [RMA, RSS 2021](https://www.roboticsproceedings.org/rss17/p011.html) | 基础策略与在线适应模块 | 历史编码适应不是新思想 |
| [ASAP, RSS 2025](https://arxiv.org/abs/2502.01143) | 真实数据学习动作残差，再在校正仿真中微调 | 本项目的反馈残差不等同 ASAP；无真实数据不能复现其 sim-to-real 闭环 |
| [SimOpt, ICRA 2019](https://arxiv.org/abs/1810.05687) | 用真实体验调整随机化分布 | 自适应随机化不作为独立新贡献 |

目前尚不能断言“增益区域 + PPO”有足够新颖性。投稿前需进一步排查 gain scheduling、执行器/策略协同设计及可辨识性工作，并证明区域构造本身具有可推广的方法价值。
