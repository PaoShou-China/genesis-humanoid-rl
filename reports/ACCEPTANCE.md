# V0.1 实际运行与验收

日期：2026-11-26。状态：**开发版，未达到 V0.1 运动能力验收。** 所有物理轨迹来自真实Genesis仿真，不包含合成实验结果。

## 工程验证

| 项目 | 结果 | 证据 |
|---|---|---|
| 三个参考项目源码审计 | 通过 | docs/REFERENCE_PROJECTS.md、reference_versions.json |
| G1实际加载与步进 | 通过 | 29关节/29动作；actor99、critic109；physics_smoke.log |
| 真实重力/足地接触、关节限位、PD、局部reset | 通过 | pytest.log、pytest.xml；11项测试通过 |
| PPO参数更新 | 通过 | actor更新最大绝对权重差0.001001358；checkpoint_verification.json |
| checkpoint和optimizer保存、加载 | 通过 | 两次烟测checkpoint与独立策略运行 |
| 恢复训练 | 通过 | common_step_counter从16到24；不宣称物理状态/RNG逐位续跑 |
| JIT导出与实际文件重载 | 通过 | 与原RSL-RL actor输出最大误差0 |
| 独立evaluation和视频 | 通过 | 下列episode表、manifest、真实mp4 |
| 可靠站立/速度跟踪 | **失败** | 完整episode成功率0，见下表 |
| GPU、五seed收敛、鲁棒/安全方法对照 | 未验收 | 本次无可见GPU；不把CPU烟测当完整科研实验 |

## 本次训练预算及性能

训练seed均为0，评价seed从1000开始；每batch2个环境，逐slot采首个完整episode。默认任务时限20s；失败与time out同时发生按失败。

| 策略 | 训练环境×rollout×iteration | 评价episode数 | 成功率 | 跌倒率 | 平均存活时间 |
|---|---|---:|---:|---:|---:|
| H1零残差/default-pose PD | 无训练 | 2 | 0% | 100% | 1.290s |
| H1 PPO | 16×24×200=76,800 transitions | 8 | 0% | 100% | 1.010s |
| H2 PPO烟测 | 2×8×2=32 transitions | 2 | 0% | 100% | 1.210s |
| H2 PPO烟测，配对条件追加评价 | 相同2轮checkpoint | 8 | 0% | 100% | 1.285s |
| H2 PPO | 16×24×500=192,000 transitions | 8 | 0% | 100% | 0.3675s |

这些结果只表明当前配置和有限CPU预算下没有获得可靠运动控制。尤其不能把短episode的较低平均速度误差解读为性能提升。表中不同任务和不同样本量不是方法优劣统计检验；未进行多seed收敛实验，也不据此否定PPO或上游项目。

`h2_comparison.json`及CSV仅对同task、seed、num_envs、8个episode的两个PPO训练预算做描述性对照，不把训练预算差异称为算法创新或显著性结果。

CPU训练的batch规模远小于上游4096环境设置。H1/H2分别采用默认512/256/128网络，烟测采用64/64。CPU collection/learning耗时在原始TensorBoard及日志中，不推断GPU性能。

## 可查看的真实证据

- [H1训练与终止轨迹图](h1_training_diagnostics.png)
- [H2两轮烟测轨迹与loss](h2_smoke_diagnostics.png)
- [H2五百轮训练诊断](h2_training_diagnostics.png)
- [H1默认站姿失败视频](h1_zero/rollout.mp4)
- [H1训练后失败视频](h1_trained_eval/rollout.mp4)
- [H2烟测策略视频](h2_eval/rollout.mp4)
- [H2五百轮策略视频](h2_trained_eval/rollout.mp4)

summary.json、episodes.json、manifest.json保留具体数值、配置和checkpoint SHA256。训练产物包另包含实际checkpoint、对应runner/task/robot/system、TensorBoard、原始trace与视频。checkpoint与配置必须一起移动。

## 运行过程与复现边界

本次使用独立CPU venv，Genesis1.3.3、GenesisLab固定SHA、RSL-RL5.4.1、torch2.15.0+cpu。没有升级或降级用户原有环境。尝试并行构建多个Genesis场景时出现明显内存压力；终止尚未开始训练的并行H2构建和原先需要物理场景的导出进程，改为顺序运行和直接导出RSL-RL actor。最终H2完整500轮已真实运行结束。

开发期运行的system.json如实记录git_dirty；本次提供解析配置、阶段commit、原始产物与版本，并不承诺跨硬件逐位复现。当前训练入口进一步保存source_snapshot.zip及其SHA256，用于保留以后每次运行的未提交源码。没有伪装为正式盲测：seed1000已经用于工程调试，正式研究需重新封存测试种子。

下一项验收是先获得可在完整20s测试中成功站立/跟踪的基线，再跑五训练seed；之后才进入DR、推扰和安全机制。当前不添加未验证的新算法来掩盖基线失败。
