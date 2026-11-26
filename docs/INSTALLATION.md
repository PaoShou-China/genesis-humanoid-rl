# 安装与环境核验

## 已实测环境

2026-11-26，Linux x86_64、Python3.12.14、**无可见GPU**。在独立venv内安装，没有修改用户原有环境。实测组合：Genesis1.3.3、GenesisLab固定SHA、RSL-RL5.4.1、PyTorch2.15.0+cpu。完整版本见reports运行记录。RSL-RL6.0.0不是本项目目标，已依据GenesisLab uv.lock约束至5.4.1。

不要直接套用SafeHumanoidsPolicy的Genesis0.2.1，也不要把用户其他项目的Genesis1.4.4原地降级。为本项目新建环境。

## 先检查再安装

```bash
git clone https://github.com/PaoShou-China/genesis-humanoid-rl.git
cd genesis-humanoid-rl
python scripts/doctor.py
nvidia-smi  # 有NVIDIA GPU时检查驱动；没有则执行CPU烟测
uv venv --python 3.12 .venv
```

CPU安装（本次验证路线）：

```bash
uv pip install --python .venv/bin/python 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python scripts/doctor.py
```

GPU环境：根据实际驱动和PyTorch官方兼容表，先在新venv安装匹配的torch/torchvision CUDA wheel，再安装 `-e '.[test]'`。本次无GPU，未验证任何CUDA wheel组合，不把CPU验证写成GPU验收。不盲目upgrade系统CUDA或用户原有torch。

Linux离屏渲染需libEGL.so.1和可用OpenGL软件/硬件实现；GenesisLab推荐libegl1、libvulkan1。本运行环境已有，未更改系统库。GUI仅在有显示服务时启用；默认评估使用离屏相机。

## 模型资产

第一次加载G1由GenesisLab下载固定mjlab commit的XML与35个mesh，缓存默认 `~/.cache/genesislab/assets/unitree_g1/`。可设 `GENESISLAB_ASSET_CACHE`。首次初始化会编译物理kernel，需要等候；失败时报告真实异常，不回退成假仿真。

configs/robots/g1_29dof.yaml包含XML SHA256和关节参数快照；每个训练run导出robot.json。12 DoF仅保留unsupported配置，不能拿29维策略加载。

## 最小真实运行

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/train.py --task H2 --num-envs 2 --device cpu \
  --ppo-config configs/algorithms/ppo_smoke.yaml --output outputs/h2_smoke
.venv/bin/python scripts/train.py --task H2 --num-envs 2 --device cpu \
  --ppo-config configs/algorithms/ppo_smoke.yaml --iterations 1 \
  --resume outputs/h2_smoke/model_final.pt --output outputs/h2_resume
.venv/bin/python scripts/evaluate.py --task H2 --device cpu --episodes 2 --num-envs 2 \
  --checkpoint outputs/h2_smoke/model_final.pt --video --output outputs/h2_eval
.venv/bin/python scripts/plot.py --evaluation outputs/h2_eval \
  --training outputs/h2_smoke --output outputs/h2_eval/diagnostics.png
```

输出目录必须不存在，避免覆盖已有实验。checkpoint需与同目录task.json/runner.json/robot.json一起移动。烟测不是行走收敛结果。

## 完整训练与五seed

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/train.py --task H2 --seed "$seed" --device cuda:0 --num-envs 4096
done
.venv/bin/python scripts/evaluate.py --task H2 --checkpoint <run>/model_final.pt \
  --episodes 64 --num-envs 16 --seed <sealed-test-seed> --device cuda:0 --output outputs/test_run
.venv/bin/python scripts/export.py --checkpoint <run>/model_final.pt --output outputs/export
```

训练默认3000 iterations只是起点，是否收敛必须看独立指标。训练和resume异常时保留日志、配置和已保存checkpoint，不能据退出码0以外的猜测标记通过。

## 当前统一入口

重新执行editable安装后提供 `ghrl`，也可使用 `python -m humanoid`。`ghrl assets --output outputs/assets.json` 可在训练前并行预取固定资产并记录哈希。所有子命令通过 `ghrl <command> --help` 查看参数。`ghrl sweep --dry-run` 先检查五seed计划；默认顺序执行，避免多个场景同时编译占满内存。当前流程见README及EXTENSIONS.md。
