# Genesis Humanoid RL Benchmark

Unitree G1 的强化学习与控制实验平台：**Genesis · GenesisLab · RSL-RL PPO**。

当前是可运行的开发版。H1/H3默认采用已验证的固定PD标定（刚度×16、阻尼×4，力矩上限不变）；历史低增益配置仍可显式加载。训练、推扰、地形、随机化、动作处理和独立评价共用一条控制链；运动能力以真实完整 episode 验收，接口通过不代表策略收敛。最新结果见 [本轮实测](reports/V2_PROGRESS.md)，历史失败保留在 [初版验收](reports/ACCEPTANCE.md)。

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

| 任务 | 实现范围 | 验证边界 |
|---|---|---|
| H1 Standing | 29 DoF、站立姿态、独立完整 episode 评价 | PD 增益扫描及初始扰动评价；PPO另行验收 |
| H2 Velocity | vx / vy / wz 指令、PPO训练和评估 | 已执行训练；尚未获得可靠行走基线 |
| H3 Push recovery | 躯干 CoM 处施加世界系真实外力，记录冲量与恢复 | 没受到推扰的 episode 不计算为恢复成功 |
| H4 Rough terrain | 上游真实地形、课程、123维 actor 观测 | 真实步进与PPO烟测通过；不宣称地形通过率 |
| H5 Motion | 真实 NPZ 的频率、关节/刚体命名、形状和四元数校验 | 尚未实现动作跟踪策略，不生成假示教数据 |

| 对照/方法 | 配置或入口 |
|---|---|
| Nominal PPO | `configs/tasks/h2_velocity.yaml` |
| PPO + DR | `configs/tasks/h2_domain_randomized.yaml`：摩擦、躯干附加质量、PD参数 |
| Soft safety penalty | `configs/tasks/h2_soft_safety.yaml` |
| PD target shield | `configs/tasks/h2_shield.yaml`；区间不可行时明确记录 |
| Residual feedback | `configs/tasks/h1_residual.yaml`；反馈位置目标 + 有界RL残差 |
| 校准 PD 参考 | `h1_stable_pd.yaml` / `h1_stiff_pd.yaml`；不提高力矩上限 |

Shield 约束的是关节目标和名义 PD 力矩代理，不保证不跌倒，也不构成形式化安全证明。执行器时延、独立惯量随机化、MBRL和全身模仿尚未实现。

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
- `ghrl export`：导出原始actor与归一化；残差/动作过滤需按元数据继续执行。

## 文档

[架构](docs/ARCHITECTURE.md) · [源码审计](docs/REFERENCE_PROJECTS.md) · [任务](docs/TASKS.md) · [方法](docs/METHODS.md) · [指标](docs/METRICS.md) · [实验协议](docs/EXPERIMENT_PROTOCOL.md) · [路线图](docs/ROADMAP.md)

原始代码 Apache-2.0；第三方依赖、策略与机器人资产各自保留原许可证，见 [NOTICE](NOTICE)。未复制授权不明确的参考仓库实现。
