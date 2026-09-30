# AIRBOT BC and offline LWD

该入口在固定离线 replay 上联合训练 DIVL critic 和 QAM actor。BC checkpoint
同时用于初始化可训练 actor 和独立的冻结 reference；之后 actor 持续由 QAM 更新，
不会继续计算 BC/SFT loss。

## 配置

`lwd_offline.yaml` 是 cube 示例，默认使用 200 条遥操作数据和 100 条
rollout 数据。其他任务复制该 YAML 后修改数据与超参数。

加入其他数据集时，复制此模板并修改 `datasets`；已知坏轨迹仍须通过
`exclude_episodes` 排除。例如，若加入 `data/lerobot/cube_0921_100`，保留
该批次已确认的排除项：

```yaml
exclude_episodes:
  - dataset: cube_0921_100
    source_file: 39.mcap
```

仅当相应数据集存在于 `datasets` 中时添加该条目；适配器会拒绝无法匹配的排除项。

每个训练 YAML 都完整声明本次实验的数据集、BC checkpoint、训练超参数和
`airbot` 数据模式。`airbot` 中的动作维度、相机映射、delta-action 维度与归一化
asset ID 必须和该 checkpoint、数据集一致；新任务应新增自己的 YAML，不要在训练
入口里增加任务特例。

默认配置使用：

- 32 步动作段和 1-step chunk-level TD；
- `gamma=0.9999`；成功 terminal reward 位于最后一个有效动作，失败为 0；
- 固定 BC reference 采样，10 个 QAM 时间步；
- actor AdamW `2e-5`，critic Adam `5e-4`；
- 全局 batch 32，设备数由 `runtime.gpus` 决定。

## 环境与统一模式

统一入口为 `scripts/train_airbot_lwd.py`，`mode: bc` 只训练 actor/EMA，
`mode: offline` 执行 DIVL + QAM。
配置文件的 `runtime` 控制设备、CPU worker 和机器依赖路径，
`batch_size` 由 GPU 数 × `per_device_batch_size` 得到，可用 `--batch-size` 显式覆盖。

## BC 归一化与训练

`lwd_bc.yaml` 是正式 BC 配置，也是 AIRBOT normalization stats 的唯一配置来源。
首次训练或数据发生变化后先计算统计量：

```bash
$OPENPI_PYTHON scripts/compute_norm_stats.py \
  --config examples/airbot/lwd_bc.yaml
```

统计量写入该 YAML 的 `norm_stats_dir`。计算过程和训练共用 `AirbotReplay.transition()`，
因此 joint delta、动作 horizon 和 terminal padding 语义保持一致。然后执行预检和训练：

```bash
$OPENPI_PYTHON scripts/train_airbot_lwd.py \
  --config examples/airbot/lwd_bc.yaml \
  --check

$OPENPI_PYTHON scripts/train_airbot_lwd.py \
  --config examples/airbot/lwd_bc.yaml
```

OpenPI 其他任务仍可通过 `compute_norm_stats.py --config-name <name>` 使用原有
`TrainConfig` 统计流程；AIRBOT 不再在 `training/config.py` 注册独立配置。

`runtime.data_workers: 0` 使用主进程串行构造 batch。将其设为正数会启用
`airbot_parallel_loader.py` 的 CPU worker 和共享内存预取；此时还需配置
`runtime.data_worker_cpus`，并可用 `runtime.prefetch_batches` 控制缓冲区数量。

W&B 默认关闭。需要记录实验时可在任一训练 YAML 中加入：

```yaml
wandb:
  enabled: true
  project: airbot-lwd
  name: cube-offline
```

`airbot_wandb.py` 会为同一 `run_dir` 保存稳定的 run ID，恢复训练时继续写入同一条记录；
checkpoint 本身始终保存在本地。

## 数据预检与训练

只检查配置、标签、数据形状、terminal 语义和动作归一化：

```bash
$OPENPI_PYTHON scripts/train_airbot_lwd.py \
  --config examples/airbot/lwd_offline.yaml \
  --run-dir checkpoints/lwd_airbot_cube/preflight \
  --prepare-only
```

训练或恢复：

```bash
$OPENPI_PYTHON scripts/train_airbot_lwd.py \
  --config examples/airbot/lwd_offline.yaml \
  --run-dir checkpoints/lwd_airbot_cube/offline

$OPENPI_PYTHON scripts/train_airbot_lwd.py \
  --config examples/airbot/lwd_offline.yaml \
  --run-dir checkpoints/lwd_airbot_cube/offline \
  --resume
```

恢复训练会同时校验解析后的训练配置、replay manifest 和训练实现源码哈希。
任何一项变化都会拒绝续训，避免新代码静默接到旧优化器状态上。完整 checkpoint
保存 actor、固定 reference、critic/target、三个优化器、RNG 和归一化资产；`params/`
可直接作为部署 actor。

## terminal chunk 语义

一个 chunk 是从一条 episode 中截出的、长度不超过 horizon 的连续动作段。
非 terminal chunk 的 `next_observation` 是该动作段之后的状态，并用
`gamma ** chunk_length` bootstrap。terminal chunk 没有 episode 外的下一状态，
因此 discount 为 0；当前实现用最后一条已记录 observation 填充张量位置，但 critic
不会读取其 bootstrap value。

当前适配器将 terminal 行的 action 计为有效动作。这只有在 MCAP 的 `action[t]`
表示从 `state[t]` 开始执行的命令时才成立；若采集格式把最后一行仅作为终止观测，
应先修正转换/切块语义，不能仅靠 mask 掩盖。

## 缓存 QAM

默认 actor 更新与 `qam_training.actor_train_step` 的算法流程相同，但复用 Pi0 的
图像/语言 prefix KV，并复用固定 reference 在采样路径上已经计算过的 velocity。
缓存仅减少重复前向计算；path、terminal Q 梯度、反向 adjoint 和 actor optimizer
仍按原顺序执行。该实现只允许 `sample_from_reference: true`。

缓存实现与 Pi0 的 `embed_prefix`、`embed_suffix` 和 KV-cache 接口直接耦合；修改
Pi0 attention 实现后必须重新开始训练，不能绕过实现哈希校验恢复旧 run。
