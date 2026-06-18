"""
Demo / Visualization script for Multi-UAV DSAC-T.

Features:
  - Main view: world map with UAVs, obstacles, point cloud, path history
  - Side panel: occupancy grid with all UAV positions
  - Save rendered clips as GIF

Usage:
  python demo.py                           # 生成最新模型的 GIF（不弹窗）
  python demo.py --show                    # 生成 GIF 并弹窗显示
  python demo.py --load_checkpoint ...     # 指定模型
  python demo.py --max_episodes 1 --show   # 仅运行一个 episode 并弹窗
"""

import argparse
import os
import numpy as np
import torch
from datetime import datetime
import glob

from config import CONFIG
from env.quadrotor_env import MultiQuadrotorEnv
from dsac_t import DSACTAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Multi-UAV DSAC-T")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Checkpoint path (or wildcard)")
    parser.add_argument("--max_episodes", type=int, default=1,
                        help="Number of demo episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--show", action="store_true",
                        help="显示图形界面（默认不显示）")
    parser.add_argument("--save_dir", type=str, default=CONFIG["paths"]["demo_clip_dir"],
                        help="Directory to save clips")
    parser.add_argument("--fps", type=int, default=CONFIG["demo"]["fps"],
                        help="Frames per second for saved clips")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip rendering (benchmark mode)")
    parser.add_argument("--stage", type=int, default=None,
                        help="Curriculum stage index (0-based), auto-detected from checkpoint if not given")
    return parser.parse_args()


def run_demo_episode(agent, env, episode, save_dir, headless, fps,
                     max_steps=500, step=0, stage_name=""):
    """
    Run a single demo episode with visualization.
    Simplified 1x2 layout matching train.py render_demo_clip.

    Args:
        agent: trained agent
        env: environment
        episode: episode number
        save_dir: directory for saved frames
        headless: if True, don't show interactive window
        fps: frames per second for saved clips
        max_steps: max steps
        step: global training step (for filename)
        stage_name: curriculum stage name (for title & filename)

    Returns:
        mean episode reward
    """
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    obs_list, _ = env.reset()
    total_reward = np.zeros(env.num_uavs)
    frames = []

    fig, axes = plt.subplots(1, 2, figsize=CONFIG["demo"]["fig_size"])
    ax_map, ax_grid = axes

    for _demo_step in range(max_steps):
        actions = [agent.select_action(obs, deterministic=True) for obs in obs_list]
        actions = np.stack(actions)

        obs_list, rewards, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        total_reward += rewards

        if _demo_step % 10 == 0:
            # ---- 世界视图 ----
            ax_map.clear()
            ax_map.set_xlim(0, env.world_size)
            ax_map.set_ylim(0, env.world_size)
            ax_map.set_aspect("equal")
            ax_map.set_title(f"{stage_name} — Step {_demo_step}" if stage_name else f"Step {_demo_step}")

            # 静态障碍物
            for obs in env.static_obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color="gray", alpha=0.5)
                ax_map.add_patch(circle)

            colors = ["red"] + [plt.cm.tab10(i / max(env.num_uavs - 1, 1)) for i in range(1, env.num_uavs)]
            for i, uav in enumerate(env.uavs):
                k = uav.kinematics
                color = colors[i]

                # 轨迹 (多机仅 UAV 0)
                if env.num_uavs == 1 or i == 0:
                    if len(uav.path_history) > 1:
                        path = np.array(list(uav.path_history))
                        ax_map.plot(path[:, 0], path[:, 1], color=color, alpha=0.5, linewidth=1)

                # 位置
                ax_map.plot(k.x, k.y, "o", color=color, markersize=8, label=f"UAV {i}")
                # 航向箭头
                arrow_len = 1.5
                ax_map.arrow(k.x, k.y, arrow_len * np.cos(k.psi), arrow_len * np.sin(k.psi),
                             head_width=0.3, color=color, alpha=0.8)
                # 目标
                ax_map.plot(uav.goal[0], uav.goal[1], "*", color=color, markersize=12)

                # 通讯圈 (多机仅 UAV 0)
                if env.num_uavs == 1 or i == 0:
                    comm_range = CONFIG["comm"]["range"]
                    comm_circle = plt.Circle((k.x, k.y), comm_range, color=color, fill=False,
                                             linestyle='--', alpha=0.4, linewidth=0.8)
                    ax_map.add_patch(comm_circle)

                # 点云射线 (多机仅 UAV 0)
                if env.num_uavs == 1 or i == 0:
                    ray_angles = np.linspace(-CONFIG["perception"]["fov"] / 2,
                                             CONFIG["perception"]["fov"] / 2,
                                             CONFIG["perception"]["num_rays"])
                    pc = obs_list[i]["pointcloud"] * CONFIG["perception"]["max_range"]
                    for j, (r, angle) in enumerate(zip(pc, ray_angles)):
                        if r < CONFIG["perception"]["max_range"] - 0.1:
                            ray_angle = k.psi + angle
                            ex = k.x + r * np.cos(ray_angle)
                            ey = k.y + r * np.sin(ray_angle)
                            ax_map.plot([k.x, ex], [k.y, ey], color=color, alpha=0.15, linewidth=0.5)

            ax_map.legend(loc="upper right", fontsize=8)

            # ---- 占据栅格 ----
            ax_grid.clear()
            grid = obs_list[0]["grid_map"]
            ax_grid.imshow(grid, cmap="hot_r", origin="lower",
                           extent=(0, CONFIG["grid"]["size"],
                                   0, CONFIG["grid"]["size"]))  # Y 轴 0→40，与左图一致
            ax_grid.set_title("Occupancy Grid (UAV 0)")
            ax_grid.set_xlabel("X (m)")
            ax_grid.set_ylabel("Y (m)")

            # 只绘制 UAV 0 在栅格中的位置
            uav0 = env.uavs[0]
            k0 = uav0.kinematics
            pos_txt = f"UAV0: ({k0.x:.1f},{k0.y:.1f})m  heading:{np.degrees(k0.psi):.0f}deg  v:{k0.v:.1f}m/s"
            ax_grid.text(0.5, -0.12, pos_txt, transform=ax_grid.transAxes, fontsize=7, ha="center", va="top",
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

            plt.tight_layout()
            if save_dir:
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                frame = frame[:, :, 1:]  # ARGB → RGB
                frames.append(frame)

        if all(dones):
            break

    # 保存 GIF
    if frames and save_dir:
        import imageio
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{stage_name}_" if stage_name else ""
        gif_path = os.path.join(save_dir, f"demo_{prefix}ep{episode}_step{step}_{timestamp}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"  Saved: {gif_path} ({len(frames)} frames)")
        # 清理旧 GIF
        import glob as _glob
        gifs = sorted(_glob.glob(os.path.join(save_dir, "demo_*.gif")), key=os.path.getmtime)
        while len(gifs) > 10:
            oldest = gifs.pop(0)
            try: os.remove(oldest)
            except OSError: pass

    mean_reward = total_reward.mean()
    print(f"  Episode {episode}: mean reward = {mean_reward:.2f}, steps = {_demo_step + 1}")
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
            files = sorted(glob.glob(args.load_checkpoint),
                           key=os.path.getmtime)
            if not files:
                print(f"No checkpoint files matching: {args.load_checkpoint}")
                return
            checkpoint_path = files[-1]
        else:
            checkpoint_path = args.load_checkpoint
    else:
        # Find latest checkpoint
        files = sorted(glob.glob(os.path.join(CONFIG["paths"]["checkpoint_dir"], "model_*.pth")),
                       key=os.path.getmtime)
        if not files:
            print("No checkpoint found. Use --load_checkpoint to specify.")
            return
        checkpoint_path = files[-1]

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load agent + extract curriculum stage metadata
    agent = DSACTAgent()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.temperature.load_state_dict(ckpt["temperature"])
    agent.step = ckpt.get("step", 0)
    agent.total_env_steps = ckpt.get("total_env_steps", 0)
    agent.actor.eval()
    agent.critic.eval()

    ckpt_stage = ckpt.get("curriculum_stage", 0)
    stage_idx = args.stage if args.stage is not None else ckpt_stage
    print(f"Agent loaded (step {agent.step}) | Device: {agent.device}")

    # Create environment matching the curriculum stage
    stages = CONFIG.get("curriculum", {}).get("stages", [])
    stage_name = ""
    if stages and stage_idx < len(stages):
        stage_cfg = stages[stage_idx]
        stage_name = stage_cfg["name"]
        env = MultiQuadrotorEnv(
            num_uavs=stage_cfg["num_uavs"],
            use_dynamic_obs=stage_cfg.get("dynamic_obs", True),
            static_obstacles_enabled=stage_cfg.get("static_obstacles", True),
        )
        print(f"┌──────────────────────────────────────────────────────────┐")
        print(f"│  Stage: {stage_cfg['name']} (idx {stage_idx})                        │")
        print(f"│  num_uavs: {stage_cfg['num_uavs']}                                       │")
        print(f"│  dynamic_obs: {str(stage_cfg.get('dynamic_obs', True)):>5}  static_obs: {str(stage_cfg.get('static_obstacles', True)):>5}         │")
        print(f"│  heading_scale: {stage_cfg.get('heading_scale', 0.1):.1f}                               │")
        print(f"│  total_steps: {stage_cfg['total_steps']:>8}                           │")
        print(f"└──────────────────────────────────────────────────────────┘")
    else:
        env = MultiQuadrotorEnv()
        print(f"Env: default ({CONFIG['num_uavs']} UAVs, use_dynamic_obs=True)")

    # 独立运行时：不弹窗 (headless=True) 除非指定 --show
    headless = not args.show

    total_reward = 0
    for ep in range(args.max_episodes):
        print(f"\nEpisode {ep + 1}/{args.max_episodes}")
        reward = run_demo_episode(
            agent, env, ep + 1,
            save_dir=args.save_dir if not args.no_render else None,
            headless=headless or args.no_render,
            fps=args.fps,
            max_steps=args.max_steps if not args.no_render else 10,
            step=agent.total_env_steps,
            stage_name=stage_name,
        )
        total_reward += reward

    avg_reward = total_reward / args.max_episodes
    print(f"\nAverage reward over {args.max_episodes} episodes: {avg_reward:.2f}")
    print(f"Demo clips saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
