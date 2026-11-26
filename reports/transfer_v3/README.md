# V3 执行器与跨引擎验收

2026-11-26（北京时间），CPU 环境；实际 Genesis 1.3.3 / MuJoCo 4.0.1。代码保留 V2 隐式 PD 默认值；新增机制全部为显式可选配置。

## 验收结果

- 全量 pytest：44 passed；ruff lint/format 通过。
- Genesis 显式 PD + per-env gain DR + 随机命令延迟：真实 80 个物理步完成；actor96 / critic109，观测组装与 portable runtime 一致。
- 延迟队列不会被 reset/write 消耗；部分 reset 不影响其他环境历史。
- 多物理步外力烟测：2 ms 物理 / 20 ms 控制周期，10 N × 0.04 s；物理步累计冲量与控制步累计均为 0.4 Ns。
- 新 96 维 actor 的 PPO 两轮烟测与导出完成；JIT roundtrip 检查通过。该烟测 actor 在 MuJoCo 为 0/2，平均约 1.01 s，不能作为收敛策略。

## 实际执行的静态 PD 工况

物理周期均为 2 ms，控制周期为 20 ms，求解器内部子步1；刚度倍率8/16，阻尼倍率取平方根；执行器原力矩上限不变。每组2个初始关节扰动 episode，开发 seed1200。

| 工况 | 成功数 | 平均持续时间/s |
|---|---:|---:|
| kp16_dt2ms_delay0ms_genesis | 2/2 | 20.00 |
| kp16_dt2ms_delay0ms_mujoco | 2/2 | 20.00 |
| kp16_dt2ms_delay10ms_genesis | 2/2 | 20.00 |
| kp16_dt2ms_delay10ms_mujoco | 2/2 | 20.00 |
| kp8_dt2ms_delay0ms_genesis | 2/2 | 20.00 |
| kp8_dt2ms_delay0ms_mujoco | 2/2 | 20.00 |
| kp8_dt2ms_delay10ms_genesis | 2/2 | 20.00 |
| kp8_dt2ms_delay10ms_mujoco | 2/2 | 20.00 |

八个工况共16个完整 episode，均达到20 s。静态目标不随状态变化，命令延迟在这些实验中没有持续激励；不能据此推断行走或反馈策略的延迟容忍度。两个引擎的 reset RNG 和接触参数也未严格配对，不能做配对显著性结论。

计划总计36项，本次执行8项；其余28项未执行。筛选器不会因已执行工况成功就接受任何完整矩阵下的增益候选；不声称已找到可迁移安全区域。

## 数据与复现

各工况保留 summary、episodes、commands、manifest、status；manifest 含源代码快照哈希、实际配置、依赖版本、策略/资产哈希（适用时）。development_plan.json 是本次机器的原始命令记录；换机器请重新运行 ghrl transfer_plan，不直接执行旧绝对路径。

全部结果是仿真开发验收，不是真机 sim-to-real、稳定性证书或算法创新验证。完整接口、对照条件、失败准则与文献边界见 [研究说明](../../docs/SIM2SIM.md)。
