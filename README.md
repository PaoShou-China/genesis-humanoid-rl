# Genesis Humanoid RL Benchmark

Unitree G1 的强化学习与控制实验平台：**Genesis · GenesisLab · RSL-RL PPO**。

当前是可运行的开发版。H1/H3默认采用已验证的固定PD标定（刚度×16、阻尼×4，力矩上限不变）；历史低增益配置仍可显式加载。训练、推扰、地形、随机化、动作处理和独立评价共用一条控制链；运动能力以真实完整 episode 验收，接口通过不代表策略收敛。任务集验收见 [V5 报告](reports/tasks_v5/README.md)，残差RL结果见 [V4 验收](reports/residual_v4/README.md)，跨引擎接口见 [V3 验收](reports/transfer_v3/README.md)，站立标定见 [V2 实测](reports/V2_PROGRESS.md)，历史失败保留在 [初版验收](reports/ACCEPTANCE.md)。

## 快速运行

先按 [安装说明](docs/INSTALLATION.md) 创建独立环境，不修改其他项目的 PyTorch/CUDA。安装后统一使用 `ghrl`；也可用 `python -m humanoid`。

```bash
ghrl doctor
ghrl assets --output outputs/assets.json
ghrl tasks

# 已校准的 PD 站立参考；zero 表示零姿态残差，并非关闭电机
ghrl evaluate --task H1 --task-config configs/tasks/h1_stiff_pd.yaml \
  --agent zero --episodes 8 --num-envs 2 --seed 201 --device cpu \
  --video --output outputs/pd_standing

# 真实 PPO 最小闭环，不用烟测权重宣称行走收敛
ghrl train --task H2 --num-envs 2 --device cpu \
  --ppo-config configs/algorithms/ppo_smoke.yaml --output outputs/ppo_smoke
```

所有输出目录必须不存在，防止覆盖已有实验。`ghrl <命令> --help` 查看完整参数。

## 任务与方法

已注册H1–H14，完整MDP、成功判据与数据依赖见 [任务集](docs/TASK_SUITE.md)。

| 任务 | 环境范围 |
|---|---|
| H1–H4 | 站立、速度跟踪、真实推扰恢复、粗糙地形 |
| H5 | 参考关节动作跟踪；需要真实命名NPZ，缺失明确报错 |
| H6–H9 | 朝向、航点、蹲起高度、双手触达 |
| H10–H11 | 俯卧/仰卧起身、跳跃与落地 |
| H12–H14 | 真实楼梯、坡面、接触推动自由刚体 |

新增九类无外部数据任务完成真实步进及两轮PPO烟测；不宣称策略学会任务。抓取/搬运仍需可动夹爪资产，不以刚性附着冒充。目标观测尚未接入MuJoCo/实机，raw actor导出会明确标记。

| 对照/方法 | 配置或入口 |
|---|---|
| Nominal PPO | `configs/tasks/h2_velocity.yaml` |
| PPO + DR | `configs/tasks/h2_domain_randomized.yaml`：摩擦、躯干附加质量、PD参数 |
| Soft safety penalty | `configs/tasks/h2_soft_safety.yaml` |
| PD target shield | `configs/tasks/h2_shield.yaml`；区间不可行时明确记录 |
| Residual RL | `configs/tasks/h1_residual_rl.yaml` / `h3_residual_rl.yaml`；冻结PD基准 + 有界rad修正、零均值初始化、正则与消融 |
| 历史 Residual feedback | `configs/tasks/h1_residual.yaml`；旧低增益踝反馈配置，保留复现 |
| 校准 PD 参考 | `h1_stable_pd.yaml` / `h1_stiff_pd.yaml`；不提高力矩上限 |

Shield 约束的是关节目标和名义 PD 力矩代理，不保证不跌倒，也不构成形式化安全证明。新增可选显式 PD、物理步命令延迟、96维本体观测与 MuJoCo 独立评价入口，见 [跨引擎迁移](docs/SIM2SIM.md)。真实执行器辨识、传感时延、独立惯量随机化、MBRL和完整全身运动模仿尚未实现。

训练入口支持 `--method ppo` / `--method residual`，目标任务、物理条件与评价协议独立于方法。

## 残差 RL

新机制明确使用 `q_ref = q_base + Δq_RL`，默认每关节修正上限0.05 rad。PPO只训练修正网络；基础PD控制器固定。提供H1站立/H3抗扰配置、零均值初始化、小幅探索、残差与变化率惩罚，以及 `ghrl evaluate --disable-residual` 同条件消融。完整机制与命令见 [残差RL说明](docs/RESIDUAL_RL.md)。

## 成组实验

```bash
# 顺序执行五个训练种子，逐次保存日志、配置、checkpoint、状态和源码快照
ghrl sweep --task H2 --task-config configs/tasks/h2_domain_randomized.yaml \
  --seeds 0 1 2 3 4 --device cuda:0 --num-envs 4096 --output outputs/dr_full

# 所有方法用同一份物理测试条件；训练奖励、shield与残差属于方法，不改变配对身份
ghrl evaluate --task H2 --checkpoint <run>/model_final.pt \
  --evaluation-config configs/evaluation/nominal.yaml --output outputs/test
```

`configs/evaluation/shifted.yaml` 提供分布偏移接口；这是开发用配置，正式研究必须另行封存未用于调试的测试集。五seed短烟测与五seed收敛实验严格区分。

## 结果与可视化

每次评价输出 `episodes.json`、`summary.json`、`trace.json`、完整 `commands.json` 和协议/命令序列哈希。终止状态在 reset 前采样，提前跌倒的环境不会贡献第二个 episode。

- `ghrl plot`：质心、姿态、关节、接触、机械功率和真实训练曲线。
- `ghrl videos`：相同测试条件的视频并排比较；短视频结束后明确标注冻结帧。
- `ghrl compare`：验证物理协议与命令序列；按独立训练seed汇总，拒绝重复seed伪增样本。
- `ghrl export`：导出原始actor与归一化；平地任务另存控制契约，残差/动作过滤由共用 PortableController 执行。
- `ghrl task_smoke`：逐任务验证reset、部分reset、真实步进和可选两轮PPO。
- `ghrl sim2sim`：在真实 MuJoCo 中评价显式 PD 或导出策略；不自动调参，不宣称真机迁移。
- `ghrl transfer_plan`：生成固定控制周期的增益/步长/延迟诊断矩阵；不自动启动长实验。

## 文档

[架构](docs/ARCHITECTURE.md) · [源码审计](docs/REFERENCE_PROJECTS.md) · [任务](docs/TASKS.md) · [方法](docs/METHODS.md) · [指标](docs/METRICS.md) · [实验协议](docs/EXPERIMENT_PROTOCOL.md) · [路线图](docs/ROADMAP.md) · [跨引擎与研究方案](docs/SIM2SIM.md)

原始代码 Apache-2.0；第三方依赖、策略与机器人资产各自保留原许可证，见 [NOTICE](NOTICE)。未复制授权不明确的参考仓库实现。
