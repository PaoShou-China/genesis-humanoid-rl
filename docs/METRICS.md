# 独立物理指标

所有指标在物理步进后、terminal reset前采样，单位显式记录。reward不是成功指标。

| 字段 | 定义 |
|---|---|
| mean_linear_error_m_s | mean_t ||v_xy^body-command_xy||₂ |
| mean_yaw_error_rad_s | mean_t |w_z^body-command_wz| |
| max_tilt_rad | max_t acos(-projected_gravity_z) |
| max_root_displacement_m | root相对reset位置的最大水平位移，不称为CoM |
| max_com_displacement_m | 全身CoM相对初始CoM最大水平位移 |
| com_x/y/z_m | native link COM用各link惯性质量加权；非link origin加权 |
| energy_j | sum_t sum_j |tau_j*dq_j|*dt，绝对机械功；不是电耗，也不计驱动器效率 |
| mean_joint_speed_rad_s | mean_t mean_j |dq_j| |
| mean_slip_speed_m_s | 接触脚的水平link速度均值；无接触时0，属于足端滑移代理指标 |
| success/failure_rate | 完整episode满足任务成功条件/其补集 |
| fall_rate | 任一物理失败终止；与任务跟踪失败率分开 |

按episode报告mean/sample std/median；n=1时std=null。五个训练seed的独立结果才适于方法层面的不确定性分析，不能把同一策略的64个episode当成64个独立训练seed。当前compare输出逐run数据，不生成伪统计显著性。

局限：机械能为控制周期末端矩形积分；精细电机能效和接触点真实滑移需更高频记录。计算成本用原始PPO日志的collection/learning时间与system.json；CPU烟测不代表GPU吞吐。
