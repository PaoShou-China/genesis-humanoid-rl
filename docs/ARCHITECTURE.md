# 技术架构

V0.1 使用薄适配层，避免重复实现 GenesisLab 的机器人控制和张量环境。

| 层 | 实现及职责 |
|---|---|
| Physics | Genesis 1.3.3，刚体/重力/接触，真实CPU或GPU后端 |
| Robot | GenesisLab G1_CFG，29关节名称映射、隐式PD、模型缓存；humanoid/robots.py负责运行时核验 |
| Task | humanoid/environments.py配置上游工厂；H1增加静止约束奖励，H2继承速度跟踪 |
| Algorithm | algorithms/ppo.py配置RSL-RL 5.4.1；不自行实现PPO |
| Evaluation | evaluation/metrics.py物理量；scripts/evaluate.py首次完整episode和终止状态采样 |
| Visualization | 原生离屏G1相机、rollout.mp4、trace.json、scripts/plot.py |

环境reset返回(obs_dict, extras)，step返回(obs_dict,reward,terminated,truncated,extras)。训练包装为RSL-RL TensorDict。训练auto reset，评估manual reset，防止把reset后的状态混入终止指标。

机器人配置YAML是审计清单；实际执行器参数导入固定SHA的G1_CFG，避免两份PD参数漂移。运行时导出的robot.json是加载后的真实顺序。资产XML校验和、关节限位和惯量保存在configs/robots/g1_29dof.yaml。

当前控制接口是关节位置残差+隐式PD。速度/力矩控制由上游实体支持但本项目没有独立任务验收，不能标为已支持。H3/H4、DR、反馈残差和PD目标shield现已实现并完成真实物理烟测，详见EXTENSIONS.md。MBRL与H5跟踪策略仍未实现。
