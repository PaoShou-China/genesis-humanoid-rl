# 有界位置残差强化学习

本项目实现的是**控制残差**：冻结基础控制器，PPO 只学习对关节目标的有界修正。它不是动力学残差模型，也不是根据真实数据拟合的 sim-to-real 校正器。当前可用的基础控制器是已标定的定姿 PD，以及可选的解析踝关节反馈；尚不提供预训练行走策略作为基础策略的加载器。

## 控制机制

对每个控制周期：

\[
q_t^{\mathrm{ref}}=q_t^{\mathrm{base}}+\Delta q_t,
\quad \Delta q_t=M\,b\,\operatorname{clip}(\pi_\theta(o_t),-1,1).
\]

其中 `b=max_position_rad` 为每关节修正上限（默认0.05 rad，约2.86°），`M` 是可选关节选择掩码。`q_base` 由冻结的解析控制器给出；默认是经过验证的站立 home 姿态。

最终目标经过动作限幅和可选 shield，再通过同一执行器实现：

\[
\tau_t=\operatorname{sat}\left[K_p(q_{t-d}^{\mathrm{ref}}-q_t)-K_d\dot q_t\right].
\]

这是位置残差。没有延迟和饱和时，可展开为基础 PD 力矩加 `Kp Δq`；存在延迟/饱和时不能将两者独立相加。位置修正有界也不意味着稳定性或不跌倒保证。

`humanoid.control.process_action` 同时用于 Genesis 和 MuJoCo。保留三项诊断：基准控制器单独经过同样投影后的动作、请求残差、加入残差后相对该基准的有效变化；因此能够区分“网络输出了修正”和“修正实际通过了过滤”。最后一项是同一时刻的反事实动作差，不是独立闭环基准轨迹差。

## 训练机制

| 环节 | 实现 |
|---|---|
| 基础控制器 | `controller.type: nominal_pose`，PD刚度×16、阻尼×4，原力矩上限保持 |
| 可选解析反馈 | `controller.type: ankle_feedback`；不能同时再使用旧 `residual_scale` 缩放 |
| 动作范围 | 策略归一化 `[-1,1]`；残差幅度直接用rad配置，不随 action_scale 改变 |
| 初始策略 | `zero_actor_output: true` 清零 actor 最后一层，使确定性均值从零修正开始 |
| 探索 | 高斯标准差初始0.05；默认0.05rad预算对应小幅位置探索，采样值仍被硬限幅 |
| 正则项 | `residual_position` 为请求残差平方和(rad²)；`residual_rate` 为变化率平方和(rad²/s²) |
| reset | 清空当前环境的残差历史，避免跨episode产生虚假变化率；不清除其他环境 |
| 观测 | 平地actor96维；保留IMU相关量、关节状态、指令及上一处理动作；critic109维 |

零初始化保证的是初始确定性策略等于基础控制器；训练中存在随机探索，不能写成“训练始终不影响基准”。恢复 checkpoint 时先构建模型再加载权重，不会将已训练模型重新清零。

H1/H3新配置分别为 `configs/tasks/h1_residual_rl.yaml` 和 `h3_residual_rl.yaml`；H3在躯干施加真实外力。它们使用2ms外部执行器更新、20ms策略更新；延迟默认0，可显式配置。H1/H3机制相同，任务与物理扰动不同。配置文件存在不代表已经训练/验收H3策略。

`configs/tasks/h1_residual.yaml` 是历史低增益踝反馈实现，保留以复现旧结果。新实验使用带 `_residual_rl` 后缀的配置。

## 训练、消融与跨引擎

```bash
# 开发用100轮小网络训练；完整研究需要独立多seed与足够训练预算
ghrl train --task H1 --task-config configs/tasks/h1_residual_rl.yaml \
  --ppo-config configs/algorithms/ppo_residual.yaml --num-envs 8 --device cpu \
  --seed 0 --output outputs/residual_training

# 学习到的残差与同一基础控制器：使用同一checkpoint、物理条件和随机种子
ghrl evaluate --task H1 --checkpoint outputs/residual_training/model_final.pt \
  --episodes 64 --num-envs 16 --seed 1300 --device cpu --output outputs/residual_on
ghrl evaluate --task H1 --checkpoint outputs/residual_training/model_final.pt \
  --disable-residual --episodes 64 --num-envs 16 --seed 1300 --device cpu \
  --output outputs/residual_off
ghrl compare outputs/residual_on outputs/residual_off \
  --labels residual base_only --output outputs/residual_comparison.json

# 未训练基准可直接用zero agent；zero是零修正，不是关电机
ghrl evaluate --task H1 --task-config configs/tasks/h1_residual_rl.yaml \
  --agent zero --episodes 8 --num-envs 2 --device cpu --output outputs/base_reference

ghrl export --checkpoint outputs/residual_training/model_final.pt --output outputs/residual_export
ghrl sim2sim --bundle outputs/residual_export --episodes 8 --output outputs/residual_mujoco
ghrl sim2sim --bundle outputs/residual_export --disable-residual --episodes 8 \
  --output outputs/base_mujoco
```

导出的 policy.pt 只包含归一化与网络；contract.json 保存基础控制器、残差预算/掩码及后处理信息。跨引擎时通过 PortableController 执行完整控制逻辑，不能遗漏基础控制器或重复叠加残差。

评价新增 `mean_residual_requested_rad`、`mean_residual_effective_rad`（每步关节最大幅值的episode平均）；trace记录基准和请求残差动作向量。完整动作诊断单位是归一化 action，乘 action_scale 才是rad。`--disable-residual`仅改变方法开关，不改变物理配对哈希；单训练种子的消融不作跨训练种子的显著性推断。

## 验收目标与边界

优先检查：零修正时严格退化为相同基础控制器、任意网络输出不越过修正预算、部分reset隔离、训练能更新残差、checkpoint/导出保留机制、独立episode评价能看出恶化或改善。

目前目标是站立/抗扰的可解释残差控制底座。尚无真机数据、稳定性证书或可靠行走基础策略；不能将这个机制本身作为新颖性贡献。是否优于基础控制器，必须看同条件独立评价，不以训练奖励上涨替代。
