# 任务和验收

| 任务 | 当前范围 | episode成功定义 |
|---|---|---|
| H1站立 | 29 DoF、20 s、零速度指令；关节reset扰动±0.1 rad | 活到time limit且未跌倒；全episode最大倾角≤0.35rad、最大root水平位移≤0.25m |
| H2速度跟踪 | 20 s；vx∈[-1,1]、vy∈[-0.5,0.5]m/s、wz∈[-0.5,0.5]rad/s | 活到time limit且未跌倒；episode平均水平速度误差≤0.3m/s、偏航角速度误差≤0.3rad/s |
| H3推扰恢复 | 已实现真实躯干外力与冲量记录 | 仅在实际受到力的episode中计算恢复率，无暴露则null |
| H4粗糙地形 | 真实地形、观测、PPO烟测通过 | 完整任务成功率仍待收敛实验 |
| H5动作跟踪 | V0.4待实现 | 不提供虚构参考动作；先核验真实数据和许可证 |

H1/H2跌倒终止：倾角>1.0 rad或root高度<0.4 m；time limit与失败同时发生时算失败。阈值是本benchmark实验约定，不是理论稳定域。episode成功率按完整episode计数，不能按每步平均“存活”代替。

nominal任务移除上游flat的随机push；未开启质量、摩擦等DR。H1去掉feet air-time收益并惩罚水平漂移和关节速度。控制周期0.02s，solver substep0.01s。

当前控制/随机化/恢复协议见 [扩展说明](EXTENSIONS.md)。原始任务配置保持兼容；校准后的固定PD参考使用显式*_stable_pd或*_stiff_pd配置，不覆盖历史checkpoint的控制参数。

H1/H3当前registry默认加载h1_stiff_pd/h3_stiff_pd；旧h1_standing/h3_push保留用于复现实验，checkpoint始终加载自身task.json而非替换为新默认配置。
