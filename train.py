"""
Training script for Multi-UAV DSAC-T with Curriculum Learning.

Features:
  - Staged curriculum: single-UAV → multi-UAV → dynamic obstacles
  - Early exit per stage when avg reward threshold met
  - Stage-aware checkpoint naming (model_stage1_single_step12345.pth)
  - Multi-environment training (vectorized)
  - Periodic checkpoint saving (every 10 min by default)
  - Periodic demo rendering (every 5 min) → saves GIF to demo_clips/
  - Resume training from stage checkpoint

Usage:
  python train.py --num_env s 16
  python train.py --load_checkpoint checkpoints/model_stage1_single_step12345.pth
  python train.py --stage 0  # start from specific curriculum stage
  python train.py --curriculum no  # disable curriculum, use flat training
"""

import argparse
import glob
import os
import time
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from config import CONFIG
from env.quadrotor_env import MultiQuadrotorEnv
from dsac_t import DSACTAgent, ReplayBuffer
from demo import run_demo_episode


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-UAV DSAC-T")
    parser.add_argument("--num_envs", type=int, default=CONFIG["train"]["num_envs"],
                        help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=CONFIG["train"]["total_steps"],
                        help="Total environment steps (flat mode)")
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
    # Curriculum controls
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Enable/disable curriculum: yes / no")
    parser.add_argument("--stage", type=int, default=None,
                        help="Start from a specific curriculum stage (0-based)")
    return parser.parse_args()


# =============================================================================
#  Environment factory
# =============================================================================

def make_env(stage: Dict) -> MultiQuadrotorEnv:
    """Create an environment instance matching a curriculum stage."""
    return MultiQuadrotorEnv(
        num_uavs=stage["num_uavs"],
        use_dynamic_obs=stage.get("dynamic_obs", True),
        static_obstacles_enabled=stage.get("static_obstacles", True),
    )


# =============================================================================
#  Stage-aware checkpoint I/O
# =============================================================================

def save_stage_checkpoint(agent: DSACTAgent, path: str,
                          stage_idx: int, stage_step: int,
                          extra_info: Dict = None):
    """Save checkpoint with curriculum metadata."""
    ckpt = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "temperature": agent.temperature.state_dict(),
        "optimizer_actor": agent.actor_opt.state_dict(),
        "optimizer_critic": agent.critic_opt.state_dict(),
        "optimizer_alpha": agent.temp_opt.state_dict(),
        "step": agent.step,
        "total_env_steps": agent.total_env_steps,
        "curriculum_stage": stage_idx,
        "curriculum_stage_step": stage_step,
    }
    if extra_info:
        ckpt.update(extra_info)
    torch.save(ckpt, path)


def load_checkpoint_with_curriculum(agent: DSACTAgent, path: str) -> Dict:
    """Load checkpoint and return curriculum resume info."""
    ckpt = torch.load(path, map_location=agent.device, weights_only=True)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.temperature.load_state_dict(ckpt["temperature"])
    agent.actor_opt.load_state_dict(ckpt["optimizer_actor"])
    agent.critic_opt.load_state_dict(ckpt["optimizer_critic"])
    agent.temp_opt.load_state_dict(ckpt["optimizer_alpha"])
    agent.step = ckpt.get("step", 0)
    agent.total_env_steps = ckpt.get("total_env_steps", 0)
    return {
        "curriculum_stage": ckpt.get("curriculum_stage", 0),
        "curriculum_stage_step": ckpt.get("curriculum_stage_step", 0),
    }


# =============================================================================
#  Demo rendering
# =============================================================================

# =============================================================================
#  File management
# =============================================================================

def cleanup_old_files(directory: str, pattern: str, keep: int = 10):
    """Remove old files, keeping only the `keep` most recent ones."""
    files = sorted(glob.glob(os.path.join(directory, pattern)),
                   key=os.path.getmtime)
    while len(files) > keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    """Compress occupancy grid to uint8 for replay buffer storage."""
    return (grid * 255).astype(np.uint8)


# =============================================================================
#  Stage-level training loop
# =============================================================================

def run_stage_training(agent: DSACTAgent, stage: Dict, stage_idx: int,
                       args, envs: List[MultiQuadrotorEnv],
                       obs_list_list: List, eval_env: MultiQuadrotorEnv,
                       checkpoint_dir: str, demo_clip_dir: str,
                       log_fn, start_step: int = 0) -> Tuple[int, float]:
    """
    Train within a single curriculum stage.

    Args:
        start_step: step offset for resume (default 0 for fresh starts)

    Returns:
        (stage_step, avg_reward_at_exit) — number of steps trained in this stage
    """
    stage_name = stage["name"]
    stage_total_steps = stage["total_steps"]
    early_exit_reward = stage.get("early_exit_avg_reward", None)
    early_exit_min_steps = CONFIG["curriculum"].get("early_exit_min_steps", 50000)
    early_exit_window = CONFIG["curriculum"].get("early_exit_window", 5)
    heading_scale = stage.get("heading_scale", 0.1)
    dynamic_obs = stage.get("dynamic_obs", True)
    num_uavs_stage = stage["num_uavs"]

    # Override heading_scale for this stage
    CONFIG["reward"]["heading_scale"] = heading_scale

    # Per-stage stats
    episode_rewards = [[] for _ in range(args.num_envs)]
    stage_step = start_step   # resume from saved step count
    log_collisions = 0
    log_goals = 0
    best_avg_reward = float("-inf")

    # Timing
    stage_start_time = time.time()
    last_save_time = stage_start_time
    last_demo_time = stage_start_time
    last_log_step = 0

    log_fn(f"=== Stage {stage_idx}: {stage_name} ===")
    log_fn(f"  num_uavs={num_uavs_stage} dynamic_obs={dynamic_obs} "
           f"heading_scale={heading_scale} total_steps={stage_total_steps}")
    if early_exit_reward is not None:
        log_fn(f"  early_exit @ avg_reward >= {early_exit_reward} "
               f"(window={early_exit_window}, min {early_exit_min_steps} steps)")

    early_exit = False
    early_exit_count = 0

    # ---------- 预热阶段：用随机动作填充缓冲区 ----------
    warmup_target = stage.get("warmup_samples", 0)
    min_warmup = agent.batch_size * 20
    warmup_target = max(warmup_target, min_warmup)

    if len(agent.buffer) < warmup_target:
        log_fn(f"  Warming up buffer: {len(agent.buffer)} → {warmup_target} samples")

    while len(agent.buffer) < warmup_target and stage_step < stage_total_steps:
        for env_idx in range(args.num_envs):
            num_uavs = len(envs[env_idx].uavs)
            random_actions = np.random.uniform(-1.0, 1.0, size=(num_uavs, 2))
            prev_obs_list = obs_list_list[env_idx]
            obs_list, rewards, terminated, truncated, info = envs[env_idx].step(random_actions)
            dones = terminated | truncated
            obs_list_list[env_idx] = obs_list

            for uav_idx in range(num_uavs):
                obs_packed = {
                    "grid_map": _compress_grid(prev_obs_list[uav_idx]["grid_map"]),
                    "pointcloud": prev_obs_list[uav_idx]["pointcloud"],
                    "self_state": prev_obs_list[uav_idx]["self_state"],
                    "dynamic_obs": prev_obs_list[uav_idx]["dynamic_obs"],
                }
                next_obs_packed = {
                    "grid_map": _compress_grid(obs_list[uav_idx]["grid_map"]),
                    "pointcloud": obs_list[uav_idx]["pointcloud"],
                    "self_state": obs_list[uav_idx]["self_state"],
                    "dynamic_obs": obs_list[uav_idx]["dynamic_obs"],
                }
                agent.buffer.push(obs_packed, random_actions[uav_idx],
                                  rewards[uav_idx], next_obs_packed, bool(dones[uav_idx]))

            if dones.any():
                obs_list, _ = envs[env_idx].reset()
                obs_list_list[env_idx] = obs_list

        stage_step += 1
        agent.total_env_steps += 1

        if stage_step % args.log_interval == 0:
            log_fn(f"  [Warmup] Buffer: {len(agent.buffer)}/{warmup_target} | Step: {stage_step}")

    log_fn(f"  Warmup complete. Buffer size: {len(agent.buffer)}")

    while stage_step < stage_total_steps and not early_exit:
        # Collect actions for all envs
        env_actions = []
        for env_idx in range(args.num_envs):
            uav_actions = []
            for obs in obs_list_list[env_idx]:
                action = agent.select_action(obs)
                uav_actions.append(action)
            env_actions.append(np.stack(uav_actions))

        # Step all environments
        for env_idx in range(args.num_envs):
            actions = env_actions[env_idx]
            prev_obs_list = obs_list_list[env_idx]
            obs_list, rewards, terminated, truncated, info = envs[env_idx].step(actions)
            dones = terminated | truncated
            obs_list_list[env_idx] = obs_list

            # Store transitions
            for uav_idx in range(len(envs[env_idx].uavs)):
                obs_packed = {
                    "grid_map": _compress_grid(prev_obs_list[uav_idx]["grid_map"]),
                    "pointcloud": prev_obs_list[uav_idx]["pointcloud"],
                    "self_state": prev_obs_list[uav_idx]["self_state"],
                    "dynamic_obs": prev_obs_list[uav_idx]["dynamic_obs"],
                }
                next_obs_packed = {
                    "grid_map": _compress_grid(obs_list[uav_idx]["grid_map"]),
                    "pointcloud": obs_list[uav_idx]["pointcloud"],
                    "self_state": obs_list[uav_idx]["self_state"],
                    "dynamic_obs": obs_list[uav_idx]["dynamic_obs"],
                }
                agent.buffer.push(
                    obs_packed,
                    actions[uav_idx],
                    rewards[uav_idx],
                    next_obs_packed,
                    bool(dones[uav_idx]),
                )
                episode_rewards[env_idx].append(rewards[uav_idx])

            # Accumulate stats
            if info:
                log_collisions += info.get("n_collisions", 0)
                log_goals += info.get("n_goals", 0)

            stage_step += 1
            agent.total_env_steps += 1

            # Reset env when episode ends
            if dones.any():
                obs_list, _ = envs[env_idx].reset()
                obs_list_list[env_idx] = obs_list

            # Network update
            if stage_step % agent.update_every == 0 and len(agent.buffer) >= agent.batch_size:
                for _ in range(agent.updates_per_step):
                    batch = agent.buffer.sample(agent.batch_size)
                    agent.update(batch)

            # Logging
            if stage_step - last_log_step >= args.log_interval:
                elapsed = time.time() - stage_start_time
                steps_per_sec = stage_step / elapsed if elapsed > 0 else 0
                buffer_size = len(agent.buffer)
                alpha_val = agent.temperature.get_alpha()

                recent_rewards = [r for ep in episode_rewards for r in ep[-100:]]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

                n_collisions = log_collisions
                n_goals = log_goals
                log_collisions = 0
                log_goals = 0

                log_fn(
                    f"[{stage_name}] Step {stage_step}/{stage_total_steps} | "
                    f"Buffer: {buffer_size} | Alpha: {alpha_val:.4f} | "
                    f"AvgReward: {avg_reward:.3f} | "
                    f"Collisions: {n_collisions} | Goals: {n_goals} | "
                    f"Speed: {steps_per_sec:.0f} steps/s | "
                    f"Elapsed: {elapsed:.0f}s"
                )
                last_log_step = stage_step

                # Early exit check — sliding window: require N consecutive passes
                if early_exit_reward is not None and stage_step > early_exit_min_steps:
                    if avg_reward >= early_exit_reward:
                        early_exit_count += 1
                    else:
                        early_exit_count = 0

                    if early_exit_count >= early_exit_window:
                        log_fn(f"  >>> Early exit triggered: avg_reward >= {early_exit_reward} "
                               f"for {early_exit_window} consecutive checks")
                        early_exit = True
                        break

                best_avg_reward = max(best_avg_reward, avg_reward)

            # Checkpoint save
            elapsed_since_save = time.time() - last_save_time
            if elapsed_since_save >= args.save_interval_min * 60:
                ckpt_name = f"model_{stage_name}_step{agent.total_env_steps}_" \
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                save_stage_checkpoint(agent, ckpt_path, stage_idx, stage_step)
                log_fn(f"  Checkpoint saved: {ckpt_path}")
                cleanup_old_files(checkpoint_dir, f"model_{stage_name}_*.pth", keep=10)
                cleanup_old_files(checkpoint_dir, "model_*.pth", keep=10)
                last_save_time = time.time()

            # Demo render
            elapsed_since_demo = time.time() - last_demo_time
            if elapsed_since_demo >= args.demo_interval_min * 60:
                log_fn("  Rendering demo clip...")
                try:
                    demo_env = make_env(stage)
                    for ep in range(1):
                        run_demo_episode(
                            agent, demo_env, ep + 1,
                            demo_clip_dir,
                            headless=True,
                            fps=CONFIG["demo"]["fps"],
                            max_steps=CONFIG["train"]["eval_max_steps"],
                            step=agent.total_env_steps,
                            stage_name=stage_name,
                        )
                    demo_env.close()
                    log_fn("  Demo clip rendered (1 episode)")
                    cleanup_old_files(demo_clip_dir, "demo_*.gif", keep=10)
                except Exception as e:
                    log_fn(f"  Demo rendering failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    last_demo_time = time.time()

    # Stage-level final save
    ckpt_name = f"model_{stage_name}_final_step{agent.total_env_steps}_" \
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    save_stage_checkpoint(agent, ckpt_path, stage_idx, stage_step)
    log_fn(f"  Stage final checkpoint: {ckpt_path}")
    cleanup_old_files(checkpoint_dir, f"model_{stage_name}_*.pth", keep=10)
    cleanup_old_files(checkpoint_dir, "model_*.pth", keep=10)

    elapsed = time.time() - stage_start_time
    log_fn(f"=== Stage {stage_name} finished: {stage_step} steps, "
           f"best_avg_reward={best_avg_reward:.2f}, elapsed={elapsed:.0f}s ===")

    return stage_step, best_avg_reward


# =============================================================================
#  Main entry point
# =============================================================================

def train():
    args = parse_args()
    torch.manual_seed(42)

    # Resolve curriculum config
    cur_cfg = CONFIG.get("curriculum", {})
    enable_curriculum = cur_cfg.get("enabled", False)
    if args.curriculum is not None:
        enable_curriculum = args.curriculum.lower() == "yes"

    # Build stage list
    if enable_curriculum:
        stages = cur_cfg.get("stages", [])
        if not stages:
            print("Curriculum enabled but no stages defined. Using flat training.")
            enable_curriculum = False

    if not enable_curriculum:
        # Flat training: single synthetic stage
        stages = [{
            "name": "flat",
            "num_uavs": CONFIG["num_uavs"],
            "dynamic_obs": True,
            "total_steps": args.total_steps,
            "early_exit_avg_reward": None,
            "heading_scale": CONFIG["reward"].get("heading_scale", 0.1),
        }]

    # Create directories
    checkpoint_dir = CONFIG["paths"]["checkpoint_dir"]
    demo_clip_dir = CONFIG["paths"]["demo_clip_dir"]
    log_dir = CONFIG["paths"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(demo_clip_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    # Create agent
    agent = DSACTAgent()

    # Determine starting stage and per-stage step offset
    start_stage_idx = cur_cfg.get("current_stage", 0)
    resume_info = {}
    if args.load_checkpoint:
        resume_info = load_checkpoint_with_curriculum(agent, args.load_checkpoint)
        start_stage_idx = resume_info["curriculum_stage"]
        log(f"Resumed from checkpoint: {args.load_checkpoint}")
        log(f"  curriculum_stage={start_stage_idx} "
            f"stage_step={resume_info['curriculum_stage_step']} "
            f"total_env_steps={agent.total_env_steps}")
    elif args.stage is not None:
        start_stage_idx = args.stage
        log(f"Starting from stage {start_stage_idx} (--stage override)")
    else:
        # Auto-detect latest checkpoint
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pth")),
                            key=os.path.getmtime)
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            resume_info = load_checkpoint_with_curriculum(agent, latest_ckpt)
            start_stage_idx = resume_info["curriculum_stage"]
            log(f"Auto-resumed from: {latest_ckpt}")
            log(f"  curriculum_stage={start_stage_idx} "
                f"stage_step={resume_info['curriculum_stage_step']}")
        else:
            log("No checkpoint found, starting fresh training.")

    log(f"Curriculum: {'enabled' if enable_curriculum else 'disabled'}, "
        f"{len(stages)} stage(s), starting at stage {start_stage_idx}")
    log(f"Config: {args.num_envs} envs, save every {args.save_interval_min}min, "
        f"demo every {args.demo_interval_min}min")

    # Iterate through stages
    for stage_idx in range(start_stage_idx, len(stages)):
        stage = stages[stage_idx]
        stage_name = stage["name"]

        # Clear replay buffer from previous stage — old experience is invalid for new env config
        agent.buffer = ReplayBuffer()

        # Build environments for this stage
        envs = [make_env(stage) for _ in range(args.num_envs)]
        eval_env = make_env(stage)

        # Reset all
        obs_list_list = [env.reset()[0] for env in envs]

        # Compute start_step: only the resumed stage inherits saved step count
        stage_start_step = 0
        if stage_idx == start_stage_idx and resume_info:
            stage_start_step = resume_info.get("curriculum_stage_step", 0)

        # Run stage training
        stage_steps, best_reward = run_stage_training(
            agent=agent,
            stage=stage,
            stage_idx=stage_idx,
            args=args,
            envs=envs,
            obs_list_list=obs_list_list,
            eval_env=eval_env,
            checkpoint_dir=checkpoint_dir,
            demo_clip_dir=demo_clip_dir,
            log_fn=log,
            start_step=stage_start_step,
        )

        # Stage final demo
        log(f"Rendering final demo for stage {stage_name}...")
        try:
            demo_env = make_env(stage)
            for ep in range(1):
                run_demo_episode(
                    agent, demo_env, ep + 1,
                    demo_clip_dir,
                    headless=True,
                    fps=CONFIG["demo"]["fps"],
                    max_steps=CONFIG["train"]["eval_max_steps"],
                    step=agent.total_env_steps,
                    stage_name=stage_name,
                )
            demo_env.close()
        except Exception as e:
            log(f"Final demo failed: {e}")

        # Clean up envs for this stage
        for env in envs:
            env.close()
        eval_env.close()

    # All stages complete
    log("All training stages completed.")


if __name__ == "__main__":
    train()
