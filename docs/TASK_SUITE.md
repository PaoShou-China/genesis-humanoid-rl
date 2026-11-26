# 人形机器人强化学习任务集

统一使用29 DoF浮动基座G1、Genesis真实接触和RSL-RL PPO。任务独立于算法；`ghrl train --method residual` 可选择冻结基础控制器与有界残差。环境可运行不等于策略已收敛。

## MDP定义

共同状态包含基座姿态/速度、关节位置/速度、接触和执行器状态。Actor为99维含真值速度观测或96维本体观测，地形任务另有24维高度扫描；critic可用特权信息。新目标同时加入actor/critic。默认动作是29维关节位置偏移；动态任务不套用站立高增益标定；H8/H11位置动作尺度为1.0rad、H10为1.5rad，以提供蹲起/起身所需关节行程，其他新平地任务为0.5rad。

| ID | 任务 | 新目标观测 | 主奖励 | 成功判据 |
|---|---|---|---|---|
| H1 | 稳定站立 | 无 | 漂移、姿态、关节速度 | 完整episode，倾角/位移峰值满足阈值 |
| H2 | 平地速度跟踪 | 已含vx/vy/wz | 速度跟踪 | 完整episode，平均跟踪误差合格 |
| H3 | 外力抗扰恢复 | 外力不直接作为actor输入 | 站立与稳定保持 | 确有推扰、恢复且完整episode未失败 |
| H4 | 混合粗糙地形 | 高度扫描 | 速度跟踪、接触约束 | 完整episode跟踪误差合格 |
| H5 | 参考关节动作跟踪 | 位置/速度误差，各29维 | 关节角RMS误差 | 末尾RMS≤0.2rad，保持0.5s；需真实NPZ |
| H6 | 朝向控制 | 相对yaw的sin/cos，2维 | 包裹角误差 | 末尾误差≤0.15rad，保持0.5s |
| H7 | 航点到达 | 身体系水平目标，2维 | 目标距离 | 进入0.2m区域，保持0.5s，未失败 |
| H8 | 蹲起/基座高度 | 目标、高度误差、垂直速度，3维 | 高度误差 | 末尾误差≤0.04m，保持0.5s |
| H9 | 双手触达 | 两腕身体系误差，6维 | 两腕最大位置误差 | 末尾两腕均在0.05m内，保持0.5s |
| H10 | 俯卧/仰卧起身 | 地面相对高度，1维 | 高度不足+倾角 | 末尾高度≥0.65m、误差≤0.2、速度<0.3m/s，保持1s |
| H11 | 跳跃与落地 | 时变目标、高度误差、垂直速度，3维 | 起跳/落地高度跟踪 | 双脚离地≥0.04s且基座≥0.84m，随后落地稳定0.5s |
| H12 | 楼梯行走 | 高度扫描 | 前向速度跟踪 | 8s完整episode、误差合格、水平位移≥1.8m |
| H13 | 坡面行走 | 高度扫描 | 前向速度跟踪 | 同H12；真实0.15坡度地形 |
| H14 | 接触推动物体 | 目标误差2维+物体相对位置3维 | 物体目标距离+接近物体 | 距离≤0.1m且物体速度<0.1m/s，保持0.5s |

目标奖励为 `exp(-(error/reward_width)^2)`，另加动作变化、姿态等正则。起身的综合误差是混合单位工程塑形值，高度条件独立检查，不称为物理距离。成功与奖励分开，不能用累计回报替代。

H5/H6/H8/H9/H10要求episode末尾仍保持目标；H7/H11/H14记录获取事件，但最终物理失败覆盖成功。保持时间只按实际控制步累计，reset时不增加。

## 场景与终止

- 平地通常以倾角>1rad、基座高度<0.4m或超时终止；失败与超时同时发生算失败。
- 起身reset为基座0.25m、俯卧/仰卧，取消通用低高度/大倾角终止，否则初始即被错误截断；保留超时。
- 跳跃必须测到实际双脚离地和足够高度，静态站立不算跳跃。默认0.8–1.6s为目标起跳阶段。
- 楼梯/坡面使用真实程序地形和高度扫描，保留基座接触终止；12×12m固定形貌，速度0.3–0.5m/s，位移门槛避免只站在平台上。
- 推动物体使用0.25m自由刚体盒、密度100kg/m³（约1.56kg），靠真实接触移动，不附着、不瞬移。允许身体任意部位推动，不称为抓取。

## 运行

```bash
ghrl tasks
ghrl task_smoke --task H10 --ppo --output outputs/getup_smoke

# 相同任务选择纯PPO或残差RL；残差默认使用零均值初始化与0.05rad预算
ghrl train --task H6 --method ppo --num-envs 4096 --device cuda:0 --output outputs/heading_ppo
ghrl train --task H6 --method residual --num-envs 4096 --device cuda:0 --output outputs/heading_residual

ghrl evaluate --task H6 --checkpoint outputs/heading_residual/model_final.pt \
  --episodes 64 --num-envs 16 --device cuda:0 --output outputs/heading_eval
ghrl evaluate --task H6 --checkpoint outputs/heading_residual/model_final.pt \
  --disable-residual --episodes 64 --num-envs 16 --device cuda:0 --output outputs/heading_base
```

目标任务关闭随机episode时钟初始化，避免跳跃时间表与实际初始状态错位。每环境独立reset目标/事件记忆；部分reset不改变其他环境。评价在terminal reset前采样，写入goals.json、目标计划哈希、目标误差和成功事件。物理协议v4加入任务目标，跨目标不能配对。

残差模式只改变方法。站立PD本身没有行走/起身技能，不能假定小残差能补全缺失技能；正式研究需提供任务适用的基础控制器。

## 数据与覆盖边界

H5默认路径 `data/motions/g1_reference.npz` 没有附带数据。需按 `ghrl validate_motion` 规范准备实际授权、已重定向的命名G1数据，并通过task YAML指定路径；缺失会在构建场景前报错。实现是关节动作跟踪，不冒充完整root/全身刚体姿态模仿。

H7不包含地图规划与障碍感知；H8不包含座椅接触；H9不包含抓取；H14不包含抓持与搬运。抓取放置/搬运仍需可动手指或夹爪资产、对应动作维度、抓持接触奖励、物体掉落/放置判据。现有29 DoF模型无这些执行器，因此不注册假抓取环境。视觉避障也需要额外前视传感和障碍场景。

高度/跳跃/起身等新增观测使用仿真测量的高度或速度，真机需要估计器；不能因基础profile名为proprioceptive就认为所有目标观测都可直接实机获得。

新目标尚无MuJoCo/SDK2观测适配器；可导出raw actor，但元数据明确 `control_contract_available: false`，拒绝复用96/99维平地部署契约。H1–H4原有部署行为保留。

范围参考官方实现：[Isaac Lab人形任务](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/humanoids_imitation.html)、[mjlab速度与跟踪](https://github.com/mujocolab/mjlab)。本实现不宣称复现其任务性能。
