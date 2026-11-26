# Genesis Humanoid RL Benchmark

基于 **Genesis + GenesisLab + RSL-RL PPO** 的 Unitree G1 人形机器人科研实验工程。

**当前状态：V0.1 开发版。真实仿真、PPO烟测、模型保存/加载和独立视频评估已运行；尚未证明站立/行走收敛，不能标记 V0.1 验收完成。** 运行结果与失败证据见 [reports](reports/)。

## 已实现

- 29 DoF G1，明确关节顺序、限位、PD和effort limit；复用固定版本GenesisLab，不修改上游。
- H1站立、H2速度跟踪，50 Hz控制；nominal训练不混入随机推扰。
- RSL-RL PPO训练、TensorBoard、checkpoint、恢复训练、JIT导出。
- 独立评估：首次完整episode、终止状态采样、成功/跌倒/跟踪误差/机械能/CoM/接触滑移代理指标。
- 真实三维运动视频、关节/接触/轨迹数据与科学绘图。
- 参考仓库源码审计、五seed实验协议、真实物理测试。

12 DoF资产、DR、推扰恢复、粗糙地形、安全shield、Residual RL、MBRL与动作模仿尚未验收，见 [路线图](docs/ROADMAP.md)。动作限幅和奖励惩罚不代表形式化安全保证。

## 快速开始

先看 [安装说明](docs/INSTALLATION.md)，新建环境并检查GPU，不修改其他科研项目的依赖。

```bash
python scripts/doctor.py
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/pytest -q

# 小规模CPU真实PPO烟测（不代表收敛）
.venv/bin/python scripts/train.py --task H2 --num-envs 2 --device cpu \
  --ppo-config configs/algorithms/ppo_smoke.yaml --output outputs/h2_smoke

# 独立环境执行并录制真实视频
.venv/bin/python scripts/evaluate.py --task H2 --device cpu --episodes 2 --num-envs 2 \
  --checkpoint outputs/h2_smoke/model_final.pt --video --output outputs/h2_eval

# 输出物理轨迹和实际训练日志图
.venv/bin/python scripts/plot.py --evaluation outputs/h2_eval \
  --training outputs/h2_smoke --output outputs/h2_eval/diagnostics.png
```

完整GPU训练：安装与驱动匹配的torch CUDA wheel后，运行 `python scripts/train.py --task H2 --device cuda:0 --num-envs 4096 --seed 0`。本次运行环境没有GPU，GPU吞吐和五seed收敛尚未验证。

## 结果与复现边界

短训练模型只能作为可执行性证据；初轮默认站姿控制和2-iteration PPO均在2个完整测试episode中跌倒，成功率0%。不能用奖励日志正常来声称已经学会行走。完整验收状态将记录在 [运行报告](reports/ACCEPTANCE.md)。

每个训练输出解析后的task/runner配置、机器人清单、commit、依赖版本、随机seed和日志。评价另存checkpoint SHA256、原始episode表、trace和视频。resume恢复optimizer和训练进度，重新初始化物理/RNG，不是逐位精确续跑。

## 文档

| 文档 | 内容 |
|---|---|
| [参考审计](docs/REFERENCE_PROJECTS.md) | 三个仓库真实代码与版本、可复用接口和差异 |
| [架构](docs/ARCHITECTURE.md) | 物理、机器人、任务、算法与独立评价 |
| [机器人配置](configs/robots/g1_29dof.yaml) | 29关节、PD、限位、质量惯量、XML校验和 |
| [任务](docs/TASKS.md) | H1/H2定义、成功与跌倒判定 |
| [方法](docs/METHODS.md) | PPO参数与后续方法边界 |
| [指标](docs/METRICS.md) | 单位、物理定义与统计局限 |
| [协议](docs/EXPERIMENT_PROTOCOL.md) | 五seed、配对条件、数据划分 |

原始代码Apache-2.0；第三方依赖和机器人资产保留各自许可证，见 [NOTICE](NOTICE)。未复制许可证不明确的SafeHumanoidsPolicy/AXIBO实现或资产。
