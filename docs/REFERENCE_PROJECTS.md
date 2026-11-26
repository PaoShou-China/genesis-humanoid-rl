# 参考项目代码审计

访问日期：2026-11-26。三个核心仓库均实际 shallow clone，阅读代码后记录如下；未修改上游文件。

## 固定版本

| 项目 | commit | 许可证/复用决定 |
|---|---|---|
| [GenesisLab](https://github.com/genesislab-project/GenesisLab) | `e06f383af31773fc748b1a72b5268bd84b26eb2a` | Apache-2.0；通过依赖和工厂配置复用 |
| [SafeHumanoidsPolicy](https://github.com/aCodeDog/SafeHumanoidsPolicy) | `1895284029912150df18a532714ad1133663c8a0` | README License 为占位文字，未发现明确授权；不复制代码/策略/资产 |
| [AXIBO](https://github.com/axiboai/axibo_rl_challenge) | `39a889ffab4fc5b58b9690c239d509d15cfa581f` | 无许可证；仅参考任务说明，不复制实现 |

所有源码链接均可将下述路径接到对应仓库 `/blob/<commit>/` 后核查。

## GenesisLab：V0.1 主依赖

关键代码：

- `pyproject.toml`：Genesis `>=1.3.1,<1.4.0`，RSL-RL `>=5.4.0`；`uv.lock` 锁定 RSL-RL 5.4.1。
- `src/genesislab/assets/g1.py`：29 个动作关节的正则选择器、站姿、六组隐式 PD 执行器；关节顺序由实际模型解析决定。
- `src/genesislab/assets/_fetch.py`：从 mjlab `0cdc56246999409b83622764f5b38edb660cf16e` 下载 G1 XML 和 35 个 STL；资产不在 Genesis 包内。
- `src/genesislab/tasks/velocity/velocity_env_cfg.py`：通用任务、观测、奖励、reset、push、地形。
- `src/genesislab/tasks/velocity/config/g1/env_cfgs.py`：G1 参数覆盖，目标高度 0.74 m，跌倒高度 0.4 m，倾角终止 1 rad。
- `src/genesislab/tasks/velocity/config/g1/rl_cfg.py`：PPO 24 步 rollout、512/256/128 网络、3000 iterations、lr=1e-3、entropy=0.005。
- `src/genesislab/envs/manager_based_rl_env.py`：五元 step 返回值、异步终止、`auto_reset=False` 保留终止状态、局部 reset。
- `src/genesislab/rl/vecenv_wrapper.py`：TensorDict actor/critic 输入；五元环境输出转四元 VecEnv；无限时域 time_outs 自举。
- `src/genesislab/rl/runner.py`：checkpoint 保存 optimizer/网络/iteration/common_step_counter；并不保存完整物理状态或 RNG。
- `src/genesislab/scripts/train.py`、`play.py`：训练、resume、加载和视频包装器。

### 实际接口

| 项 | 核验结果 |
|---|---|
| 引擎/算法 | Genesis / RSL-RL PPO；manager-based 原生张量并行环境 |
| 模型 | floating-base G1 29 DoF，双腿12、腰3、双臂14；非12 DoF资产 |
| 控制 | q_target=q_default+0.5a；隐式PD；本项目将策略动作 clip 到[-1,1]，不构成安全保证 |
| 频率 | sim.dt=0.02 s、substeps=2、decimation=1；50 Hz控制、100 Hz内部积分 |
| actor观测 | base linear vel(3)、angular vel(3)、gravity(3)、command(3)、q_rel(29)、dq(29)、last action(29)：99维 |
| critic观测 | actor99 + contact(2) + air time(2) + contact force(6)：109维，无观测噪声 |
| 奖励 | 水平/偏航跟踪exp、竖直速度、横滚俯仰角速度、高度、关节偏移、动作变化率、足离地时间、G1姿态惩罚；按dt积分 |
| reset | 默认 root pose/速度 + 关节位置±0.1 rad、关节速度0；各环境独立计时 |
| 地形/接触 | flat为平面+双脚接触传感；rough工厂增加地形/高度扫描/摩擦与质量随机化 |
| 训练指标 | reward、episode length、命令跟踪等训练统计；没有本项目定义的完整成功验收 |

复用范围：模型加载、关节PD、场景、接触、reset/step、张量管理器、PPO runner。扩展方式：独立工厂配置、少量H1奖励项、独立评估脚本。上游 flat 默认含速度 kick，因此本项目 nominal PPO 明确移除此 event。原生 rough 实现存在不代表本项目已经验收 H4。

局限：代码接口具备不代表已有收敛G1模型；CPU测试不验证4096环境GPU吞吐；资产需联网缓存；API对Genesis/RSL-RL版本敏感。

## SafeHumanoidsPolicy：感知和经验安全参考

关键代码：`envs/g1_with_inspire_config.py`、`envs/g1_with_hand_robot.py`、`sim2sim.py`、`requirements.txt`、LiDAR/IMU文件。

| 项 | 代码审计结果 |
|---|---|
| 定位 | 带感知的G1已训练策略部署/sim2sim；根目录入口加载 `assets/policy.jit` |
| 引擎/版本 | README安装Genesis 0.2.1；与GenesisLab 1.3.x不兼容，不能直接拼装 |
| 机器人/动作 | G1 29维关节位置残差动作，scale=0.5；显式dof名称/indices转换 |
| 控制频率 | decimation=4；base config的sim.dt=0.005 s，推得50 Hz；需运行才能确认实际执行 |
| 观测 | 配置声明 `3*4+3*29+225=324`；实际 `get_observations()` 拼接angvel3+gravity3+command3+q29+dq29+action29+visual36=132。README维数不能直接作为接口 |
| 感知 | Warp LiDAR，`theta_feature()`分为36角度箱；配置 enable_imu=False，不能因存在IMU文件就声称当前策略启用IMU |
| 指令 | `get_observations()`中强制 `commands[:,0]=0.5`，键盘指令存在覆盖 |
| 奖励 | 配置有速度/姿态/高度/关节限位等soft penalty；公开部署循环未提供可独立复现的完整PPO训练入口 |
| DR | 代码含摩擦、base mass、velocity push分支；存在分支不等于发布策略已验证各分布 |
| 路径 | 配置含作者机器绝对URDF路径；直接运行需修正资源解析 |
| 安全/评价 | 可参考避障感知及奖励；未发现可据此认证的CBF/CLF证明或完整动作shield验收，不宣称形式化保证 |
| 训练与统计 | README提及外部Genesis Legged Gym；未在此仓库复现训练或多seed结果 |

借鉴：V0.3设计感知/安全性能指标时参考，不导入旧版环境，不复制没有明确许可证的代码、资产或策略。不把公开JIT演示当作本项目训练结果。

## AXIBO：任务规格，不是baseline实现

检查完整文件树：仓库提供 README，未包含Python训练实现、PPO超参数、控制器代码或机器人资产。不能声称已从AXIBO复现行走。

| 项 | 已核验内容 |
|---|---|
| 引擎/模型 | 要求Genesis，G1 12 DoF（双腿12），外部Google Drive资产 |
| 任务 | vx/vy/wz速度跟踪、自然步态奖励设计、BeyondMimic动态运动跟踪 |
| 算法 | 允许RSL-RL；推荐Genesis Go2示例，未公开完整G1 PPO实现 |
| 观测/奖励/动作 | 三维速度指令、12关节；具体观测拼接/奖励权重/控制接口未公开 |
| 频率 | 给定运动文件名80fps，但说明metadata为50fps；这是参考轨迹采样频率，不等于已核实控制频率 |
| 数据 | 描述607帧、12关节13个body的数据；未下载验证外部文件，禁止填造运动数据 |
| 评价 | 稳定性、自然性、跟踪保真度和工程质量；没有可直接执行的统一成功率协议 |
| 复用与局限 | 借鉴任务分层；12 DoF配置只标记blocked，不复用29维checkpoint |

## 其他参考的边界

Genesis与RSL-RL已作为实际安装依赖并阅读所调用API。PHC、HumanoidBench、Whole Body Tracking/BeyondMimic尚未做源码审计或集成，不据其名称推断接口；后续H5阶段再固定版本和审核资产许可证。用户提供的BeyondMimic与Whole Body Tracking链接指向同一仓库，不应计成两套独立baseline。
