"""
Training script for Multi-UAV DSAC-T.

Features:
  - Multi-environment training (vectorized)
  - Periodic checkpoint saving (every 10 min by default)
  - Periodic demo rendering (every 5 min) → saves MP4 to demo_clips/
  - Resume training from checkpoint

Usage:
  python train.py --num_envs 16 --total_steps 10000000
  python train.py --load_checkpoint checkpoints/model_xxx.pth
"""

import argparse
import glob
import os
import time
import numpy as np
import torch
from datetime import datetime
from typing import Optional

from config import CONFIG
from env.fixedwing_env import MultiFixedWingEnv
from dsac_t import DSACTAgent, ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-UAV DSAC-T")
    parser.add_argument("--num_envs", type=int, default=CONFIG["train"]["num_envs"],
                        help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=CONFIG["train"]["total_steps"],
                        help="Total environment steps")
    parser.add_argument("--save_interval_min", type=int, default=CONFIG["train"]["save_interval_min"],
                        help="Checkpoint save interval (minutes)")
    parser.add_argument("--demo_interval_min", type=int, default=CONFIG["train"]["demo_interval_min"],
                        help="Demo rendering interval (minutes)")
    parser.add_argument("--log_interval", type=int, default=CONFIG["train"]["log_interval"],
                        help="Logging interval (steps)")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval_episodes", type=int, default=CONFIG["train"]["eval_episodes"],
                        help="Episodes per demo render")
    return parser.parse_args()


def render_demo_clip(agent: DSACTAgent, env: MultiFixedWingEnv,
                     episode: int, save_dir: str, max_steps: int = 500,
                     step: int = 0):
    """
    Render a demo clip showing UAV navigation.
    Saves frames as a video file using matplotlib animation.

    Args:
        agent: trained DSACTAgent
        env: environment instance
        episode: episode number (for filename)
        save_dir: directory to save the clip
        max_steps: max steps per episode
        step: cumulative training step for filename labeling
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, axes = plt.subplots(1, 2, figsize=CONFIG["demo"]["fig_size"])
    ax_map, ax_pc = axes

    obs_list = env.reset()
    frames = []
    total_reward = np.zeros(env.num_uavs)

    for step in range(max_steps):
        actions = []
        for uav_obs in obs_list:
            action = agent.select_action(uav_obs, deterministic=True)
            actions.append(action)
        actions = np.stack(actions)

        obs_list, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Capture frame every 5 steps to keep file size manageable
        if step % 10 == 0:
            # Left: world view
            ax_map.clear()
            ax_map.set_xlim(0, env.world_size)
            ax_map.set_ylim(0, env.world_size)
            ax_map.set_aspect("equal")
            ax_map.set_title(f"Step {step}")

            # Draw static obstacles
            for obs in env.static_obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color="gray", alpha=0.5)
                ax_map.add_patch(circle)

            # Draw UAVs
            colors = ["red"] + [plt.cm.tab10(i / max(env.num_uavs - 1, 1))
                                for i in range(1, env.num_uavs)]
            for i, uav in enumerate(env.uavs):
                k = uav.kinematics
                color = colors[i]

                # Path history
                if len(uav.path_history) > 1:
                    path = np.array(list(uav.path_history))
                    ax_map.plot(path[:, 0], path[:, 1], color=color, alpha=0.5, linewidth=1)

                # UAV position
                ax_map.plot(k.x, k.y, "o", color=color, markersize=8, label=f"UAV {i}")

                # Heading arrow
                arrow_len = 1.5
                ax_map.arrow(k.x, k.y,
                             arrow_len * np.cos(k.psi),
                             arrow_len * np.sin(k.psi),
                             head_width=0.3, color=color, alpha=0.8)

                # Goal
                ax_map.plot(uav.goal[0], uav.goal[1], "*", color=color, markersize=12)

                # Point cloud rays
                ray_angles = np.linspace(-CONFIG["perception"]["fov"] / 2,
                                         CONFIG["perception"]["fov"] / 2,
                                         CONFIG["perception"]["num_rays"])
                pc = uav_obs["pointcloud"] * CONFIG["perception"]["max_range"]
                for j, (r, angle) in enumerate(zip(pc, ray_angles)):
                    if r < CONFIG["perception"]["max_range"] - 0.1:
                        ray_angle = k.psi + angle
                        ex = k.x + r * np.cos(ray_angle)
                        ey = k.y + r * np.sin(ray_angle)
                        ax_map.plot([k.x, ex], [k.y, ey], color=color, alpha=0.15, linewidth=0.5)

            ax_map.legend(loc="upper right", fontsize=8)

            # Right: occupancy grid (UAV 0)
            ax_pc.clear()
            grid = obs_list[0]["grid_map"]
            ax_pc.imshow(grid, cmap="hot_r", origin="lower",
                         extent=(0, CONFIG["grid"]["size"],
                                 CONFIG["grid"]["size"], 0))
            ax_pc.set_title("Occupancy Grid (UAV 0)")
            ax_pc.set_xlabel("X (m)")
            ax_pc.set_ylabel("Y (m)")

            plt.tight_layout()

            # Capture frame
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, 1:]  # ARGB → RGB
            frames.append(frame)

        if all(dones):
            break

    # Save as GIF (more portable than MP4)
    if frames:
        import imageio
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(save_dir, f"demo_ep{episode}_step{step}_{timestamp}.gif")
        imageio.mimsave(gif_path, frames, fps=CONFIG["demo"]["fps"])
        print(f"  Demo clip saved: {gif_path} ({len(frames)} frames)")

    plt.close(fig)

    mean_reward = total_reward.mean()
    print(f"  Demo episode {episode}: mean reward = {mean_reward:.2f}")
    return mean_reward


def cleanup_old_files(directory: str, pattern: str, keep: int = 10):
    """Remove old files, keeping only the `keep` most recent ones by modification time."""
    files = sorted(glob.glob(os.path.join(directory, pattern)),
                   key=os.path.getmtime)
    while len(files) > keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass


def train():
    args = parse_args()

    # Create directories
    checkpoint_dir = CONFIG["paths"]["checkpoint_dir"]
    demo_clip_dir = CONFIG["paths"]["demo_clip_dir"]
    log_dir = CONFIG["paths"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(demo_clip_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    envs = [MultiFixedWingEnv() for _ in range(args.num_envs)]

    # Create agent
    agent = DSACTAgent()

    # Load checkpoint if resuming
    start_step = 0
    if args.load_checkpoint:
        start_step = agent.load_checkpoint(args.load_checkpoint)
        print(f"Resumed from checkpoint: {args.load_checkpoint} (step {start_step})")

    # Log file
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    # Training stats
    episode_rewards = [[] for _ in range(args.num_envs)]
    episode_lens = [[] for _ in range(args.num_envs)]
    total_steps = 0
    episode_count = 0
    best_reward = float("-inf")

    # Timing
    start_time = time.time()
    last_save_time = start_time
    last_demo_time = start_time
    last_log_step = 0

    log(f"Starting training: {args.num_envs} envs, {args.total_steps} total steps")
    log(f"Checkpoint interval: {args.save_interval_min}min, Demo interval: {args.demo_interval_min}min")

    # Reset all environments
    obs_list_list = [env.reset() for env in envs]

    # Main training loop
    while total_steps < args.total_steps:
        # Collect actions for all envs
        env_actions = []
        for env_idx in range(args.num_envs):
            uav_actions = []
            for uav_idx in range(len(envs[env_idx].uavs)):
                obs = obs_list_list[env_idx][uav_idx]
                action = agent.select_action(obs)
                uav_actions.append(action)
            env_actions.append(np.stack(uav_actions))

        # Step all environments
        for env_idx in range(args.num_envs):
            actions = env_actions[env_idx]
            obs_list, rewards, dones, info = envs[env_idx].step(actions)
            obs_list_list[env_idx] = obs_list

            # Store transitions
            for uav_idx in range(len(envs[env_idx].uavs)):
                uav = envs[env_idx].uavs[uav_idx]
                obs = {
                    "grid_map": obs_list[uav_idx]["grid_map"],
                    "pointcloud": obs_list[uav_idx]["pointcloud"],
                    "self_state": obs_list[uav_idx]["self_state"],
                    "dynamic_obs": obs_list[uav_idx]["dynamic_obs"],
                }
                next_obs = obs  # For simplicity, current obs is the "next" after step
                agent.buffer.push(
                    obs,
                    actions[uav_idx],
                    rewards[uav_idx],
                    next_obs,
                    bool(dones[uav_idx]),
                )
                episode_rewards[env_idx].append(rewards[uav_idx])

            total_steps += 1
            agent.total_env_steps += 1

            # Update agent periodically
            if total_steps % agent.update_every == 0 and len(agent.buffer) >= agent.batch_size:
                for _ in range(agent.updates_per_step):
                    batch = agent.buffer.sample(agent.batch_size)
                    critic_loss, actor_loss, alpha_loss = agent.update(batch)

            # Log progress
            if total_steps - last_log_step >= args.log_interval:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
                buffer_size = len(agent.buffer)
                alpha_val = agent.temperature.get_alpha()

                log(
                    f"Step {total_steps}/{args.total_steps} | "
                    f"Buffer: {buffer_size} | "
                    f"Alpha: {alpha_val:.4f} | "
                    f"Speed: {steps_per_sec:.0f} steps/s | "
                    f"Elapsed: {elapsed:.0f}s"
                )
                last_log_step = total_steps

            # Checkpoint save
            elapsed_since_save = time.time() - last_save_time
            if elapsed_since_save >= args.save_interval_min * 60:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"model_step{agent.total_env_steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                )
                agent.save_checkpoint(ckpt_path)
                log(f"Checkpoint saved: {ckpt_path}")
                cleanup_old_files(checkpoint_dir, "model_*.pth", keep=10)
                last_save_time = time.time()

            # Demo render (every N minutes)
            elapsed_since_demo = time.time() - last_demo_time
            if elapsed_since_demo >= args.demo_interval_min * 60:
                log(f"Rendering demo clip...")
                try:
                    # Create a fresh eval environment
                    eval_env = MultiFixedWingEnv()
                    for ep in range(args.eval_episodes):
                        render_demo_clip(agent, eval_env, ep + 1,
                                         demo_clip_dir, CONFIG["train"]["eval_max_steps"],
                                         step=agent.total_env_steps)
                    log(f"Demo clip rendered ({args.eval_episodes} episodes)")
                    cleanup_old_files(demo_clip_dir, "demo_*.gif", keep=10)
                except Exception as e:
                    log(f"Demo rendering failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    last_demo_time = time.time()

    # Final save
    final_ckpt = os.path.join(
        checkpoint_dir,
        f"model_final_step{agent.total_env_steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )
    agent.save_checkpoint(final_ckpt)
    log(f"Training complete. Final checkpoint: {final_ckpt}")
    cleanup_old_files(checkpoint_dir, "model_*.pth", keep=10)

    # Final demo
    log("Rendering final demo clip...")
    try:
        eval_env = MultiFixedWingEnv()
        for ep in range(args.eval_episodes):
            render_demo_clip(agent, eval_env, ep,
                             demo_clip_dir, CONFIG["train"]["eval_max_steps"],
                             step=agent.total_env_steps)
    except Exception as e:
        log(f"Final demo failed: {e}")

    total_time = time.time() - start_time
    log(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    train()
