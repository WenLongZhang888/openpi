# AIRBOT cube: fine-tuning from pi05_base

Run commands from `/home/zwl/openpi` using `.venvs/openpi-b300`.

## Dataset conversion

```bash
cd /home/zwl/openpi
.venvs/openpi-b300/bin/python examples/airbot/convert_mcap_to_lerobot.py \
  --config examples/airbot/conversion_cube_200.yaml \
  --limit 200
```

This selects `mcap001.mcap` through `mcap200.mcap` and writes LeRobot v2.1
to `data/airbot_cube_200`. Add `--resume` to continue an interrupted conversion.

## Training configuration

`pi05_airbot_cube` in `src/openpi/training/config.py` uses:

- The local `pi05_base/params` checkpoint under `.cache/openpi/openpi-assets/checkpoints`.
- Full parameter fine-tuning, batch size 32, 20,000 steps, and action horizon 32
  (1.28 seconds at the dataset's 25 Hz sampling rate).
- The instruction `pick and place cube`, injected by the training configuration
  instead of the dataset's `cube` task label.
- Three cameras: base, left wrist, and right wrist, at 224 x 224.
- State/action order: left arm (6), left gripper (1), right arm (6), right gripper (1).
  Arm targets become deltas relative to the current observation; grippers remain
  absolute. The inverse transform restores absolute arm targets for inference.
  Trossen-specific joint signs and gripper scaling are disabled (`adapt_to_pi=False`).
- The model's 32-dimensional input/output space, with the 14 robot dimensions
  padded during training and output sliced back to 14 at inference.
- Dataset-specific quantile normalization, a 1,000-step learning-rate warmup to
  2.5e-5 followed by cosine decay to 2.5e-6, and EMA decay 0.99.
- Checkpoints every 1,000 steps, retaining milestones every 5,000 steps.
  W&B logging is disabled by default; pass `--wandb-enabled` to enable it.

## Compute normalization statistics

Run this once for this dataset and action representation, or after changing either:

```bash
cd /home/zwl/openpi
HF_LEROBOT_HOME="$PWD/data" \
OPENPI_DATA_HOME="$PWD/.cache" \
JAX_PLATFORMS=cpu \
.venvs/openpi-b300/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_airbot_cube
```

Statistics are saved to `assets/pi05_airbot_cube/airbot_cube_200/norm_stats.json`.

## Start training

Use an available GPU; the example selects GPU 2. Keep the library path for the
existing B300 environment's CUDA/cuDNN setup.

```bash
cd /home/zwl/openpi
export HF_LEROBOT_HOME="$PWD/data"
export OPENPI_DATA_HOME="$PWD/.cache"
export LD_LIBRARY_PATH="$PWD/.cache/divl_validation/cudnn-cu13/nvidia/cudnn/lib:$PWD/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/cu13/lib"

CUDA_VISIBLE_DEVICES=2 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venvs/openpi-b300/bin/python scripts/train.py pi05_airbot_cube \
  --exp-name cube200_pi05base
```

Checkpoints and their normalization assets are saved under
`checkpoints/pi05_airbot_cube/cube200_pi05base/`.
To resume, use the same command and experiment name with `--resume` appended.
For a new run, use a new experiment name. This is a training configuration;
training loss alone does not establish real-robot task success.

## Local validation (2026-09-20)

The converted dataset contains 200 episodes and 34,741 frames. Normalization
statistics have been generated. Real samples, including episode boundaries,
passed camera mapping, prompt injection, and delta/absolute action round-trip checks.
Two full training updates on one real batch of 32 loaded `pi05_base` successfully
on GPU 2, with finite losses (0.14799, 0.13887) and gradient norms. No trained
checkpoint was saved by this check, and the 20,000-step run has not been started.
The validation report is `.cache/airbot_cube_validation/train_smoke.json`.
