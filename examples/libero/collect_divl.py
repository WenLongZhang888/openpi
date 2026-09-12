import argparse
from pathlib import Path
import uuid

import numpy as np
from openpi_client import image_tools


def policy_observation(obs, prompt):
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    axisangle = np.zeros(3) if den < 1e-8 else quat[:3] * (2.0 * np.arccos(w) / den)

    def image(key):
        rotated = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, 224, 224))

    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate([obs["robot0_eef_pos"], axisangle, obs["robot0_gripper_qpos"]]).astype(
            np.float32
        ),
        "prompt": str(prompt),
    }


def collect_episode(env, client, obs, prompt, max_steps, replan_steps=5, action_horizon=30, gamma=0.9999):
    """One transition per executed action prefix; observations at chunk boundaries."""
    if not 1 <= replan_steps <= action_horizon:
        raise ValueError("replan_steps must be between 1 and action_horizon")
    frames = [policy_observation(obs, prompt)]
    actions, masks, rewards, discounts, terminals, lengths = [], [], [], [], [], []
    steps = 0
    success = False
    while steps < max_steps:
        plan = np.asarray(client.infer(frames[-1])["actions"], dtype=np.float32)
        count = min(replan_steps, max_steps - steps)
        if plan.ndim != 2 or plan.shape[0] < count or plan.shape[1] != 7:
            raise ValueError(f"Expected a LIBERO action plan [H,7], got {plan.shape}")
        executed, chunk_reward = [], 0.0
        terminal = False
        for j in range(count):
            # Record exactly what is passed to the environment.
            action = np.clip(plan[j], -1.0, 1.0)
            obs, _, done, _ = env.step(action.tolist())
            steps += 1
            success = bool(env.check_success())
            chunk_reward += gamma**j * float(success)
            executed.append(action)
            # This adapter treats the benchmark time limit as terminal failure.
            terminal = success or bool(done) or steps >= max_steps
            if terminal:
                break
        k = len(executed)
        padded = np.zeros((action_horizon, 7), dtype=np.float32)
        padded[:k] = executed
        actions.append(padded)
        masks.append(np.arange(action_horizon) < k)
        rewards.append(chunk_reward)
        discounts.append(0.0 if terminal else gamma**k)
        terminals.append(terminal)
        lengths.append(k)
        frames.append(policy_observation(obs, prompt))
        if terminal:
            break

    mc_returns = np.zeros(len(rewards), dtype=np.float32)
    value = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        value = rewards[i] + discounts[i] * value
        mc_returns[i] = value
    return {
        "base": np.stack([f["observation/image"] for f in frames]),
        "wrist": np.stack([f["observation/wrist_image"] for f in frames]),
        "state": np.stack([f["observation/state"] for f in frames]),
        "actions": np.stack(actions),
        "action_mask": np.stack(masks),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "discounts": np.asarray(discounts, dtype=np.float32),
        "terminal": np.asarray(terminals, dtype=bool),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "mc_returns": mc_returns,
        "prompt": np.asarray(prompt),
        "success": np.asarray(success),
        "gamma": np.asarray(gamma),
    }


def save_episode(directory, episode):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.npz"
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **episode)
    temporary.replace(path)
    return path


def main():
    from libero.libero import benchmark
    from main import LIBERO_DUMMY_ACTION
    from main import LIBERO_ENV_RESOLUTION
    from main import _get_libero_env
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-episode", type=int, default=0, help="Starting initial-state index for continuation")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-id", required=True, help="Checkpoint/source label for this fixed policy server")
    parser.add_argument("--output", default="data/divl_libero/episodes")
    args = parser.parse_args()
    limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, prompt = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    client = WebsocketClientPolicy(args.host, args.port)
    try:
        for episode_index in range(args.start_episode, args.start_episode + args.episodes):
            init_id = episode_index % len(init_states)
            env.reset()
            obs = env.set_init_state(init_states[init_id])
            for _ in range(10):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            client.reset()
            episode = collect_episode(
                env, client, obs, prompt, limits[args.suite], args.replan_steps, args.action_horizon, args.gamma
            )
            episode.update(
                suite=np.asarray(args.suite),
                task_id=np.asarray(args.task_id),
                init_id=np.asarray(init_id),
                policy_id=np.asarray(args.policy_id),
                source=np.asarray("rollout"),
                seed=np.asarray(args.seed),
                # Same initial state stays in the same split across all checkpoints.
                split=np.asarray("val" if init_id % 5 == 0 else "train"),
            )
            path = save_episode(args.output, episode)
            print(
                f"{path}: success={bool(episode['success'])}, steps={episode['lengths'].sum()}, split={episode['split']}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
