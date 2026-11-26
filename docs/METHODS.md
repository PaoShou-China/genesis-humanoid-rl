# 方法

V0.1仅实现RSL-RL PPO。Actor/Critic为独立MLP，默认512/256/128，rollout24步，5个epoch、4个minibatch，gamma0.99、lambda0.95、clip0.2、lr1e-3、entropy0.005。完整解析配置保存在每个run的runner.json。

动作q_target=q_default+0.5*clip(a,-1,1)，由GenesisLab执行器应用PD和effort limit。此残差是对默认姿态的参数化，**不等于**已经实现基于传统运动控制器的Residual RL。

ppo_smoke.yaml使用64/64网络、8步rollout和2个iteration，仅验证更新/序列化闭环，不是收敛实验，不能与完整训练混用为性能结论。

V0.2：Nominal PPO vs PPO+DR，在同分布和预先封存的OOD分布评价；V0.3：奖励soft penalty与动作shield分开对照，独立报告约束违规；V0.4：Residual RL、MBRL、模仿。当前不宣称任何形式化安全保证。

## 当前扩展

已接入PPO+DR、soft joint-limit penalty、PD目标shield、踝关节反馈位置目标与RL残差；见 [EXTENSIONS.md](EXTENSIONS.md)。它们具备真实可执行路径，但没有据接口烟测宣称算法优越性。默认站姿PD控制器已进行固定增益扫描，所有effort limit保持不变。


## 显式残差RL机制

新的冻结基准、有界rad修正、零均值初始化、正则与消融流程见 [RESIDUAL_RL.md](RESIDUAL_RL.md)。旧 controller.residual_scale 配置保留原语义；新 residual 字段不允许与旧缩放混用。
