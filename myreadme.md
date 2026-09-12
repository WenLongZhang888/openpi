# DIVL：LIBERO 数据采集启动命令

当前先使用 `pi05_libero` 固定策略采集，成功和失败轨迹都保存，不更新 actor 或 critic。
先完成下面的单任务 10 条试采，再扩展到 Spatial、Object、Goal；暂不采 LIBERO-10。
首批约 750 条指各任务、各策略的成功与失败轨迹合计，不是 750 条成功轨迹。

## 终端一：启动策略服务

使用 `.venvs/openpi-b300` 的 Python、本地 `pi05_libero` 权重和 GPU 4。
独立的 cuDNN 目录已准备好，下面的命令需要保留 `LD_LIBRARY_PATH` 设置。

```bash
cd /home/zwl/openpi

export LD_LIBRARY_PATH="$PWD/.cache/divl_validation/cudnn-cu13/nvidia/cudnn/lib:$PWD/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/cu13/lib"

CUDA_VISIBLE_DEVICES=4 \
OPENPI_DATA_HOME=/home/zwl/openpi/.cache \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venvs/openpi-b300/bin/python scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/zwl/openpi/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

权重路径显式指向本地目录；`OPENPI_DATA_HOME` 指向已有 actor tokenizer 的缓存根目录。
保持此终端运行，等待服务开始监听后，再启动采集。首次推理需要 JAX 编译，会比后续请求慢。

## 终端二：采集一个任务的 10 条轨迹

复用 `/home/zwl/RLinf/.venv` 中已经验证可用的 LIBERO 仿真环境。
`PYTHONPATH` 让采集端使用本项目的 `openpi_client`。

```bash
cd /home/zwl/openpi

MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=4 \
PYTHONPATH=/home/zwl/openpi/packages/openpi-client/src \
/home/zwl/RLinf/.venv/bin/python examples/libero/collect_divl.py \
  --host localhost \
  --port 8000 \
  --suite libero_spatial \
  --task-id 0 \
  --episodes 10 \
  --replan-steps 5 \
  --action-horizon 30 \
  --gamma 0.9999 \
  --seed 7 \
  --policy-id pi05_libero \
  --output data/divl_libero/episodes
```

每条完整轨迹保存为 `data/divl_libero/episodes/<uuid>.npz`。
终端打印 `success=True/False`、实际执行步数和 `train/val` 分组。
`--episodes 10` 是总尝试次数，失败也计数并保存；环境异常不作为失败训练数据保存。

## 后续调整参数

| 参数 | 含义 |
| --- | --- |
| `--suite` | 当前选择 `libero_spatial`、`libero_object` 或 `libero_goal` |
| `--task-id` | 套件内任务编号，以上三个套件均为 0～9 |
| `--episodes` | 本次进程采集的总轨迹数，包含成功与失败 |
| `--policy-id` | 写入数据的策略来源标签，不负责切换服务端权重 |
| `--replan-steps 5` | 每次预测后实际最多执行前 5 步，再重新预测 |
| `--action-horizon 30` | 保存动作数组的容量；仅实际执行的步数在 action_mask 中标为有效 |

切换权重需要重启终端一的策略服务，并同步修改终端二的 `--policy-id`。
base 的 LIBERO 推理配置和归一化仍需单独验证，不能只改来源标签就当作 base 数据。
每次重新启动采集脚本，初始状态编号会从 0 开始，新的 UUID 文件不会覆盖旧数据。
因此重复启动同一任务并不代表采到了全新的初始状态；正式扩量时需要留意重复计数。

## 三个套件批量采集（可续采）

`scripts/collect_divl_suites.py` 按 Spatial、Object、Goal 的顺序执行。
每个套件 10 个任务，每任务补齐 25 个初始状态（编号 0～24），共 750 条。
已有相同策略、套件、任务、初始状态的轨迹计入配额，成功和失败都计数。
例如已有 Spatial task 0 的初始状态 0～9 时，只需再补 740 条。

策略服务启动后，采集端使用以下命令（同一输出目录只运行一个批量采集进程）：

```bash
cd /home/zwl/openpi

MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=4 \
PYTHONPATH=/home/zwl/openpi/packages/openpi-client/src \
/home/zwl/RLinf/.venv/bin/python -u scripts/collect_divl_suites.py \
  --episodes-per-task 25 \
  --policy-id pi05_libero \
  --host localhost \
  --port 8000 \
  --output data/divl_libero/episodes \
  --run-dir data/divl_libero/collection_run
```

`data/divl_libero/collection_run/status.json` 记录每套件已完成数量、成功/失败数及当前任务，
在任务批次结束时更新；同目录的任务日志会逐条输出轨迹结果。
发生异常时批量进程停止并记录 `failed`，修复后重新执行同一命令即可补采缺失的初始状态。
现有单任务脚本也支持 `--start-episode` 指定起始初始状态编号。

2026-09-11 已启动后台采集，进程信息保存在 `data/divl_libero/collection_run/launch.json` 和 `pid` 中。
查看进度（后台启动时的总日志为 `launcher.log`）：

```bash
cd /home/zwl/openpi
cat data/divl_libero/collection_run/status.json
tail -n 20 data/divl_libero/collection_run/launcher.log
```

## 真实轨迹 critic 训练检查

2026-09-11 使用最初的 10 条真实成功轨迹（8 条训练、2 条验证），
加载完整 Gemma/SigLIP 预训练权重，batch size 2，执行 3 次全参数 Adam 更新。
损失/梯度均为有限值，各模块参数发生更新，首次 EMA 数值检查通过。
训练 loss 为 19.39 → 14.13 → 49.60，留出集 loss 为 369.01。
这说明训练连接可运行，但小样本、小 batch 下波动较大，尚未证明收敛或价值估计可靠性。
正式训练前需要进一步检查学习率、预热和 batch size；此次测试没有保存训练后的 critic checkpoint。

详细报告：`.cache/divl_validation/train_real_episodes_report.json`。
测试日志：`.cache/divl_validation/train_real_episodes.log`。
测试还使用了独立 Gemma 依赖目录，具体路径记录在报告的 `dependency_overlays` 中。

## openpi-b300 环境兼容说明

- 该 uv 环境的 Orbax 0.11.25 返回 `StepMetadata`，旧代码直接读取 `metadata["params"]` 会报错。
  已在 `src/openpi/models/model.py` 中增加 `item_metadata` 解包，同时兼容旧环境的 Orbax 0.11.13。
- 该环境同时安装了 CUDA 12/13 的 cuDNN 包，库文件混用会导致推理时报缺失符号。
  当前通过 `.cache/divl_validation/cudnn-cu13` 中独立的 cuDNN 9.24.0.43 和上述 `LD_LIBRARY_PATH` 解决。
  该目录是当前启动所需的运行库，清理缓存后需要恢复。问题来自依赖版本与库文件混用，并非 B300 硬件本身。
- 已验证实际 `pi05_libero` 权重加载、真实 LIBERO 策略 rollout 和完整轨迹保存；最初 10 条试采均成功。
