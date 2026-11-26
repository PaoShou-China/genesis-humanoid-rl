# 可复现实验协议

预注册配置：configs/experiments/baseline.yaml。train seeds0–4；validation100–102；test1000–1004。使用validation确定超参数；封存测试集后才能正式评价。当前工程烟测使用seed1000验证接口，正式科研评估应重新封存未使用测试种子，不能把这次调试当盲测。

每个checkpoint保存task.json、runner.json、robot.json、system.json、invocation.json、TensorBoard和状态文件。独立评价保存checkpoint SHA256、task、系统版本、完整episode表、轨迹、汇总与可选视频。

评估指定总episode数，按固定环境slot采样首个完整episode；某slot提前跌倒后，其后episode不进入样本。已经结束slot可重置以继续推进其他slot，但active mask不再计入统计。最后一个batch在开始时就固定需要的slot，避免选取最先结束的episode造成长度偏差。

命令每4s更新，使用独立NumPy RNG预生成每个batch、每个slot的命令序列；H2每段10%零指令。不同策略即使终止时刻不同，也不会改变其他slot的后续命令序列。比较必须保持task配置、seed、num_envs和episode数一致。训练中的命令采样使用上游实现。

resume恢复网络、optimizer、iteration与common_step_counter，重新初始化物理状态/RNG；不是逐位精确续跑。最终checkpoint额外显式保存。失败与time out同时发生优先记失败。外推H3/H4、DR/OOD、五seed收敛和理论安全证明未完成前，不发布相应性能结论。
