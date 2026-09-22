# AIRBOT cube：离线 LWD

从项目根目录 `/home/zwl/openpi` 运行。配置文件是
[`lwd_offline.yaml`](lwd_offline.yaml)，入口是
[`scripts/train_airbot_lwd.py`](../../scripts/train_airbot_lwd.py)。

## 数据与初始化

- BC 初始化：`checkpoints/pi05_airbot_cube/cube200_pi05base_4gpu_10k/4000`。
- 固定数据池：`data/lerobot/airbot_cube_200` 和 `data/lerobot/cube_0920_100`，
  共 300 条、53,284 帧，287 条成功、13 条失败。无在线采集或数据池刷新。
- 按数据集、结果标签分层，并整条 episode 划分。训练 269 条（258 成功、11 失败），
  验证 31 条（29 成功、2 失败）；对应 1,610 / 201 个动作段。均匀采样训练动作段。
- 验证集只保证不参与当前 RL 更新。初始 BC 已使用 200 条示教，因此其中的示教验证
  episode 不是对 BC 未见的数据；验证 TD loss 不能替代真机成功率。
- Actor、reference、critic 都接收同一状态的三路相机和任务指令 `pick and place cube`。
- AIRBOT 14 维动作顺序：左臂 6、左夹爪、右臂 6、右夹爪。关节目标转为相对于
  动作段首帧状态的增量，夹爪保持绝对值。Actor 和 critic 共用 BC checkpoint 内
  固定的分位数归一化统计；actor 网络保留 32 维输出，critic 使用前 14 维。
  不施加 LIBERO 的物理动作 `[-1, 1]` 裁剪。

用户确认每次实际执行 32 步，因此预测和 critic 动作段均为 32 步（25 Hz 下 1.28 秒）。
最后不足 32 步的动作段使用有效位掩码。成功标签位于 episode 级别，当前将成功奖励
放在最后一个记录动作；其他奖励为 0，成功和失败的真实终止折扣均为 0。
最后一帧只在终止 transition 中作为不参与 bootstrap 的 next-observation 占位。

12 条轨迹记录了接管时的时间间隔。间隔两侧分别组成动作段，丢弃跨间隔的那一步动作；
间隔前的动作段仍 bootstrap 到最后一个连续观测，不把间隔伪造成成功或失败终止。

## 训练设置及论文对应

对照 [LWD v4 IV-B、IV-C 和附录 B](https://arxiv.org/html/2605.00416v4)：

| 项目 | 本配置 |
| --- | --- |
| 更新顺序 | 每步 V → Q → target EMA → QAM actor |
| critic-only warmup | 无，从第 1 步联合训练 |
| actor 初始化 | AIRBOT BC checkpoint |
| reference | 初始 BC 的独立固定副本，不更新 |
| critic 初始化 | Gemma 3 270M、SigLIP So400M 预训练骨干，新建 V/Q 头 |
| 参数更新范围 | Actor 和 critic 均全参数训练；reference、target 不参与梯度更新 |
| actor 优化器 | AdamW，2e-5，余弦衰减，梯度范数裁剪 1 |
| critic 优化器 | V、Q 两个 Adam，5e-4，余弦衰减，共享 encoder，独立动量 |
| γ / τ / α | 0.9999 / 0.6 / 0.3 |
| EMA 新参数权重 | 0.005 |
| QAM 温度 | 2 |
| TD | 短时 cube 任务采用 1-step **chunk-level** TD |
| 训练预算 | 10,000 步，GPU 0–3，全局 batch 32（每卡 8） |
| 保存 / 验证 | 每 1,000 / 200 步，以及本次运行最后一步 |

10,000 步、全局 batch 32 和 AIRBOT 的 32 步动作长度是本数据集/硬件的工程配置，不是论文
给定的实验规模。QAM 从固定 reference 生成路径（IV-B）；保留项目已有的 QAM
数值实现：10 步、epsilon=0.1、配套 SDE/伴随系数、最后一步 reference ODE 去噪。
这不是作者官方代码的逐项复现。LWD 伪代码中 endpoint 的记号与 IV-B 正文存在歧义；
此处按正文使用生成动作的终点求 critic 梯度，不把 replay 失败动作作为 BC 监督。

## 启动

```bash
cd /home/zwl/openpi
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/airbot/train_lwd_offline.sh \
  --batch-size 32 --run-dir checkpoints/lwd_airbot_cube/offline_gpu03_b32
```

启动脚本配置本机已验证的 CUDA/cuDNN、Gemma 依赖和本地模型缓存。使用四卡同步数据并行，模型与优化器复制到各卡，梯度跨卡同步。
首次更新需要 JAX 编译。可更改 `CUDA_VISIBLE_DEVICES` 选择空闲卡。

用 tmux 启动（本命令直接进入训练会话）：

```bash
tmux new-session -s lwd_cube_gpu03 -c /home/zwl/openpi \
  'CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/airbot/train_lwd_offline.sh --batch-size 32 --run-dir checkpoints/lwd_airbot_cube/offline_gpu03_b32'
```

按 `Ctrl+b`，再按 `d` 离开会话，训练继续。重新进入：

```bash
tmux attach-session -t lwd_cube_gpu03
```

续训必须保持数据、配置、BC 归一化统计不变：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/airbot/train_lwd_offline.sh \
  --batch-size 32 --run-dir checkpoints/lwd_airbot_cube/offline_gpu03_b32 --resume
```

`config.json` 保存生效配置，`replay_manifest.json` 保存 episode 划分和元数据哈希，
`metrics.jsonl` 记录每步训练指标及固定验证样本的指标。验证仅用于观察，不自动选优。
遇到非有限 loss/梯度立即停止，并保留此前完整 checkpoint。

每个 `checkpoints/<step>/` 保存 actor、固定 reference、critic/target、三个优化器、RNG
及步数，可恢复联合训练；同时包含标准 OpenPI 的 `params/` 和
`assets/airbot_cube_200/norm_stats.json`，供 `pi05_airbot_cube` 推理配置加载。
只保留最近两个完整 checkpoint，新 checkpoint 保存成功后才清理旧 checkpoint。
不完整 checkpoint 位于 `<step>.incomplete`，不会用于恢复。

## 验证命令

仅数据预检（不加载大模型）：

```bash
JAX_PLATFORMS=cpu bash examples/airbot/train_lwd_offline.sh \
  --prepare-only --run-dir .cache/airbot_lwd/preflight
```

两步真实联合更新、保存和恢复检查（使用独立目录）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/airbot/train_lwd_offline.sh \
  --smoke-steps 2 --batch-size 32 --run-dir .cache/airbot_lwd/smoke_gpu03_b32
```

`--smoke-steps` 限制本次更新次数，不改变 10,000 步学习率调度。
测试 checkpoint 可在同目录用 `--resume` 继续，但正式实验建议使用新的 run 目录。

## 本机验证记录

数据预检与 7 项动作/边界回归检查已通过。B300 单卡、batch 4 的两步联合更新均为有限值，
训练 critic loss 为 36.798、24.212；已确认 actor/critic 参数探针变化、reference 参数探针
保持不变，并完成完整训练 checkpoint 的保存和恢复。导出 actor 可通过标准
`pi05_airbot_cube` 配置加载，全部 3,353,433,872 个参数为有限值。
这些是运行连通性检查，不是收敛或真机成功率结论。

- 数据报告：`.cache/airbot_lwd/preflight/data_audit.json`
- 联合训练报告：`.cache/airbot_lwd/smoke/smoke_report_2.json`
- 导出加载报告：`.cache/airbot_lwd/export_validation.json`

四卡配置验证（GPU 0–3，全局 batch 32、每卡 8）：两步联合更新均为有限值，
critic loss 为 37.417、14.897，模型/优化器复制状态检查通过。首次编译后第二步约 7 秒。
四设备全局梯度归约的独立回归检查也通过（总计 8 项）。
四卡测试报告目录：`.cache/airbot_lwd/smoke_gpu03_b32/`。

## 2026-09-21：统一优化入口

`examples/airbot/train_lwd_offline.sh` 现统一启用优化实现，直接运行
`scripts/train_airbot_lwd.py` 也使用同一实现。默认 GPU 仍为 0–3，
支持原有参数和 `CUDA_VISIBLE_DEVICES` 覆盖。旧的独立优化启动器仅转发至统一入口。

优化包括：单次更新内复用 reference/actor 的图像语言 KV 前缀（Actor 前缀仍参与求导）；
复用 reference 采样路径上已经计算的速度；缓存训练动作段起点/后继观测的解码图像。
仅对 AIRBOT 的固定 reference 采样启用该 QAM 实现，其他任务原有默认更新函数不变。
TD、QAM 的 10 个时间步、优化器及 checkpoint 格式不变。

两卡 GPU 6、7、每卡 batch 8 的独立诊断中，Actor 更新从 5.458 秒降至 0.597 秒；
优化完整步（尚未加解码图像缓存）为 1.257 秒。低精度图计算和梯度累加顺序有所改变：
单步损失差异约 0.77%，参数更新相对 L2 差异约 5.17%，因此不是逐位等价替换。
具体诊断位于 `.cache/airbot_lwd/perf_gpu67_20260921/`。

每次训练会保存 `implementation.json` 标记所用实现。续训仍要求配置和数据快照一致。
修改代码不会改变已经运行的旧进程；只有重新启动的进程使用优化实现。
