"""
Demo / Visualization script for Multi-UAV DSAC-T.

Features:
  - Main view: world map with UAVs, obstacles, point cloud, path history
  - Side panel: occupancy grid, velocity vectors, uncertainty ellipses
  - Future trajectory prediction (1-2s)
  - Save rendered clips as GIF/MP4

Usage:
  python demo.py --load_checkpoint checkpoints/model_xxx.pth --max_episodes 5
  python demo.py --load_checkpoint checkpoints/model_xxx.pth --headless --max_episodes 3
"""

import argparse
import os
import numpy as np
import torch
import math
from datetime import datetime
from typing import Optional, List
import glob

from config import CONFIG
from env.fixedwing_env import MultiFixedWingEnv
from dsac_t import DSACTAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Multi-UAV DSAC-T")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Checkpoint path (or wildcard)")
    parser.add_argument("--max_episodes", type=int, default=3,
                        help="Number of demo episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI (save frames to files)")
    parser.add_argument("--save_dir", type=str, default=CONFIG["paths"]["demo_clip_dir"],
                        help="Directory to save clips")
    parser.add_argument("--fps", type=int, default=CONFIG["demo"]["fps"],
                        help="Frames per second for saved clips")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip rendering (benchmark mode)")
    return parser.parse_args()


def predict_future_trajectory(agent: DSACTAgent, obs: dict,
                              num_steps: int = 20, dt: float = 0.1) -> np.ndarray:
    """
    Predict future trajectory by rolling out the policy.
    Returns: (num_steps, 2) array of predicted positions.
    """
    from env.fixedwing_env import FixedWingKinematics

    # Extract current state from observation
    # We need to reconstruct kinematics from the obs
    # self_state: [v_norm, sin(psi), cos(psi), d_goal_norm, theta_goal_norm, arrived]
    self_state = obs["self_state"]
    v_norm = (self_state[0] + 1) / 2  # back from [-1,1] to [0,1]
    v = v_norm * (CONFIG["uav"]["v_max"] - CONFIG["uav"]["v_min"]) + CONFIG["uav"]["v_min"]
    psi = math.atan2(self_state[1], self_state[2])

    # Create a temporary kinematics for prediction
    pred_kin = FixedWingKinematics(0, 0, psi, v)

    # Predict using current policy
    predicted_positions = []
    for step in range(num_steps):
        action = agent.select_action(obs, deterministic=True)
        pred_kin.step(action[0], action[1], dt)
        predicted_positions.append([pred_kin.x, pred_kin.y])

    return np.array(predicted_positions)


def run_demo_episode(agent: DSACTAgent, env: MultiFixedWingEnv,
                     episode: int, save_dir: str, headless: bool,
                     fps: int = 10, max_steps: int = 500) -> float:
    """
    Run a single demo episode with visualization.

    Args:
        agent: trained agent
        env: environment
        episode: episode number
        save_dir: directory for saved frames
        headless: if True, don't show interactive window
        fps: frames per second for saved clips
        max_steps: max steps

    Returns:
        mean episode reward
    """
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Ellipse

    obs_list = env.reset()
    total_reward = np.zeros(env.num_uavs)
    frames = []

    # Setup figure: 2x2 layout
    fig = plt.figure(figsize=CONFIG["demo"]["fig_size"])
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Main world view (spans top-left and bottom-left)
    ax_world = fig.add_subplot(gs[:, 0])
    ax_grid = fig.add_subplot(gs[0, 1])
    ax_pc = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[1, 1:])

    colors = plt.cm.tab10(np.linspace(0, 1, env.num_uavs))

    for step in range(max_steps):
        actions = []
        for uav_obs in obs_list:
            action = agent.select_action(uav_obs, deterministic=True)
            actions.append(action)
        actions = np.stack(actions)

        obs_list, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Render every 3 steps to balance speed and smoothness
        if step % 3 == 0 or step == max_steps - 1:
            # ---- World View ----
            ax_world.clear()
            ax_world.set_xlim(-2, env.world_size + 2)
            ax_world.set_ylim(-2, env.world_size + 2)
            ax_world.set_aspect("equal")
            ax_world.set_title(f"Multi-UAV Navigation — Step {step}", fontsize=12)
            ax_world.grid(True, alpha=0.3)

            # Static obstacles
            for obs in env.static_obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2],
                                    color="gray", alpha=0.6, edgecolor="black")
                ax_world.add_patch(circle)

            # Tracked dynamic objects
            for dobj in env.tracked_objects:
                pos = dobj.pos
                vel = dobj.vel
                circle = plt.Circle((pos[0], pos[1]), 0.4,
                                    color="red", alpha=0.4, edgecolor="darkred")
                ax_world.add_patch(circle)
                ax_world.arrow(pos[0], pos[1], vel[0], vel[1],
                               head_width=0.2, color="red", alpha=0.6)

            # UAVs
            for i, uav in enumerate(env.uavs):
                k = uav.kinematics
                color = colors[i]

                # Path history
                if len(uav.path_history) > 1:
                    path = np.array(list(uav.path_history))
                    ax_world.plot(path[:, 0], path[:, 1],
                                  color=color, alpha=0.4, linewidth=1.5)

                # Future trajectory prediction
                try:
                    future_path = predict_future_trajectory(
                        agent, obs_list[i],
                        num_steps=CONFIG["demo"]["predict_steps"],
                        dt=env.dt
                    )
                    # Transform predicted path relative to current position
                    if len(future_path) > 0:
                        ax_world.plot(
                            k.x + future_path[:, 0],
                            k.y + future_path[:, 1],
                            "--", color=color, alpha=0.6, linewidth=1.5,
                            label=f"UAV {i} pred" if step == 0 else ""
                        )
                except Exception:
                    pass

                # UAV position
                ax_world.plot(k.x, k.y, "o", color=color, markersize=10,
                              markeredgecolor="black", markeredgewidth=1,
                              label=f"UAV {i}" if step == 0 else "")

                # Heading indicator (velocity vector)
                arrow_len = 2.0
                ax_world.arrow(k.x, k.y,
                               arrow_len * math.cos(k.psi),
                               arrow_len * math.sin(k.psi),
                               head_width=0.4, head_length=0.5,
                               color=color, alpha=0.8)

                # Uncertainty ellipse (placeholder - would use critic std)
                ellipse = Ellipse(xy=(k.x, k.y), width=0.8, height=0.8,
                                  angle=np.degrees(k.psi),
                                  color=color, alpha=0.15)
                ax_world.add_patch(ellipse)

                # Goal marker
                ax_world.plot(uav.goal[0], uav.goal[1], "*",
                              color=color, markersize=15,
                              markeredgecolor="white", markeredgewidth=0.5)

                # Point cloud rays
                ray_angles = np.linspace(-CONFIG["perception"]["fov"] / 2,
                                         CONFIG["perception"]["fov"] / 2,
                                         CONFIG["perception"]["num_rays"])
                pc = obs_list[i]["pointcloud"] * CONFIG["perception"]["max_range"]
                for j in range(0, len(pc), 3):  # Every 3rd ray for clarity
                    if pc[j] < CONFIG["perception"]["max_range"] - 0.1:
                        ray_angle = k.psi + ray_angles[j]
                        ex = k.x + pc[j] * math.cos(ray_angle)
                        ey = k.y + pc[j] * math.sin(ray_angle)
                        ax_world.plot([k.x, ex], [k.y, ey],
                                      color=color, alpha=0.15, linewidth=0.5)

            ax_world.legend(loc="upper right", fontsize=7, ncol=2)

            # ---- Occupancy Grid (UAV 0) ----
            ax_grid.clear()
            grid = obs_list[0]["grid_map"]
            im = ax_grid.imshow(grid, cmap="hot_r", origin="lower",
                                extent=(0, CONFIG["grid"]["size"],
                                        CONFIG["grid"]["size"], 0),
                                vmin=0, vmax=1)
            ax_grid.set_title(f"Occupancy Grid (UAV 0)", fontsize=10)
            ax_grid.set_xlabel("X (m)")
            ax_grid.set_ylabel("Y (m)")

            # ---- Point Cloud (UAV 0) ----
            ax_pc.clear()
            pc = obs_list[0]["pointcloud"]
            angles = np.linspace(-CONFIG["perception"]["fov"] / 2,
                                  CONFIG["perception"]["fov"] / 2,
                                  len(pc))
            ax_pc.plot(np.degrees(angles), pc, "b.-", linewidth=1)
            ax_pc.set_title(f"Point Cloud (UAV 0)", fontsize=10)
            ax_pc.set_xlabel("Angle (deg)")
            ax_pc.set_ylabel("Distance (norm)")
            ax_pc.set_ylim(0, 1.1)
            ax_pc.grid(True, alpha=0.3)

            # ---- Stats ----
            ax_stats.clear()
            metrics = {
                "Mean Reward": total_reward.mean(),
                "Collisions": sum(1 for u in env.uavs if u.collided),
                "Arrived": sum(1 for u in env.uavs if u.arrived),
                "Tracked Objs": len(env.tracked_objects),
                "Step": step,
            }
            labels = list(metrics.keys())
            values = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in metrics.values()]
            ax_stats.axis("off")
            table = ax_stats.table(
                cellText=[values],
                colLabels=labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            ax_stats.set_title("Episode Stats", fontsize=10)

            plt.tight_layout()

            if not headless:
                plt.pause(0.01)

            # Capture frame
            if save_dir:
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                frame = frame[:, :, 1:]  # ARGB → RGB
                frames.append(frame)

        if all(dones):
            break

    # Save clip
    if frames and save_dir:
        import imageio
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(save_dir, f"demo_ep{episode}_{timestamp}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"  Saved: {gif_path} ({len(frames)} frames)")

    mean_reward = total_reward.mean()
    print(f"  Episode {episode}: mean reward = {mean_reward:.2f}, "
          f"steps = {step + 1}, "
          f"collisions = {sum(1 for u in env.uavs if u.collided)}")

    if not headless:
        plt.show()
    else:
        plt.close(fig)

    return mean_reward


def main():
    args = parse_args()

    # Resolve checkpoint path
    if args.load_checkpoint:
        # Support wildcard
        if "*" in args.load_checkpoint:
            files = sorted(glob.glob(args.load_checkpoint))
            if not files:
                print(f"No checkpoint files matching: {args.load_checkpoint}")
                return
            checkpoint_path = files[-1]
        else:
            checkpoint_path = args.load_checkpoint
    else:
        # Find latest checkpoint
        files = sorted(glob.glob(os.path.join(CONFIG["paths"]["checkpoint_dir"], "model_*.pth")))
        if not files:
            print("No checkpoint found. Use --load_checkpoint to specify.")
            return
        checkpoint_path = files[-1]

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load agent
    agent = DSACTAgent()
    agent.load_checkpoint(checkpoint_path)
    agent.actor.eval()
    agent.critic.eval()

    print(f"Agent loaded (step {agent.step}) | Device: {agent.device}")

    # Create environment
    env = MultiFixedWingEnv()

    # Run demo episodes
    total_reward = 0
    for ep in range(args.max_episodes):
        print(f"\nEpisode {ep + 1}/{args.max_episodes}")
        reward = run_demo_episode(
            agent, env, ep + 1,
            save_dir=args.save_dir if not args.no_render else None,
            headless=args.headless or args.no_render,
            fps=args.fps,
            max_steps=args.max_steps if not args.no_render else 10,
        )
        total_reward += reward

    avg_reward = total_reward / args.max_episodes
    print(f"\nAverage reward over {args.max_episodes} episodes: {avg_reward:.2f}")
    print(f"Demo clips saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
