"""
Evaluation script for Multi-UAV DSAC-T.

Runs the trained model on each curriculum stage for a full episode (2000 steps),
and reports: goal arrivals, collisions, avg reward, avg speed, max speed,
avg time to goal, and more.

Usage:
  python evaluate.py                           # latest checkpoint, all stages
  python evaluate.py --load_checkpoint ...     # specific model
  python evaluate.py --stage 2                 # only a specific stage
  python evaluate.py --episodes 20             # number of evaluation episodes
"""

import argparse
import os
import numpy as np
import torch
from typing import Dict, List, Tuple
import glob
from datetime import datetime

from config import CONFIG
from env.quadrotor_env import MultiQuadrotorEnv
from dsac_t import DSACTAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-UAV DSAC-T")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: latest)")
    parser.add_argument("--stage", type=int, default=None,
                        help="Evaluate only a specific stage (0-3)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per stage")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def run_episode(agent, env, max_steps=2000):
    """
    Run one episode and collect detailed statistics.
    Returns dict with metrics.
    """
    obs_list, _ = env.reset()
    total_reward = np.zeros(env.num_uavs)
    total_goals = 0
    total_collisions = 0
    step_count = 0
    speed_sum = 0.0
    max_speed = 0.0
    goal_times = []          # list of step numbers when a UAV arrived

    # 记录每个 UAV 初始的 success_count
    prev_success = [uav.success_count for uav in env.uavs]

    for step in range(max_steps):
        actions = [agent.select_action(obs, deterministic=True) for obs in obs_list]
        actions = np.stack(actions)
        obs_list, rewards, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        total_reward += rewards
        step_count += 1

        total_goals += info.get("n_goals", 0)
        total_collisions += info.get("n_collisions", 0)

        # 通过 success_count 变化精准检测每次到达
        for i, uav in enumerate(env.uavs):
            if uav.success_count > prev_success[i]:
                goal_times.append(step)
            prev_success[i] = uav.success_count

        # 速度统计
        for uav in env.uavs:
            v = uav.kinematics.v
            speed_sum += v
            if v > max_speed:
                max_speed = v

        if all(dones):
            break

    avg_reward = total_reward.mean() / step_count if step_count > 0 else 0.0
    avg_speed = speed_sum / (step_count * env.num_uavs) if step_count > 0 else 0.0
    avg_goal_time = np.mean(goal_times) if goal_times else 0.0

    return {
        "steps": step_count,
        "avg_reward": avg_reward,
        "total_goals": total_goals,
        "total_collisions": total_collisions,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "avg_goal_time": avg_goal_time,
        "goal_times": goal_times,
    }


def evaluate_stage(agent, stage_cfg, episodes, max_steps, seed):
    """Evaluate on a specific curriculum stage."""
    env = MultiQuadrotorEnv(
        num_uavs=stage_cfg["num_uavs"],
        use_dynamic_obs=stage_cfg.get("dynamic_obs", True),
        static_obstacles_enabled=stage_cfg.get("static_obstacles", True),
    )
    env.reset(seed=seed)

    all_metrics = []
    for ep in range(episodes):
        metrics = run_episode(agent, env, max_steps)
        all_metrics.append(metrics)
        print(f"  Ep {ep+1}: goals={metrics['total_goals']}, "
              f"collisions={metrics['total_collisions']}, "
              f"avg_reward={metrics['avg_reward']:.3f}, "
              f"avg_speed={metrics['avg_speed']:.2f}, "
              f"max_speed={metrics['max_speed']:.2f}, "
              f"avg_goal_time={metrics['avg_goal_time']:.1f}")

    env.close()
    return all_metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load checkpoint
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
    else:
        ckpt_dir = CONFIG["paths"]["checkpoint_dir"]
        files = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.pth")),
                       key=os.path.getmtime)
        if not files:
            print("No checkpoint found!")
            return
        checkpoint_path = files[-1]

    print(f"Loading checkpoint: {checkpoint_path}")
    agent = DSACTAgent()
    ckpt = torch.load(checkpoint_path, map_location=agent.device, weights_only=True)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.temperature.load_state_dict(ckpt["temperature"])
    agent.step = ckpt.get("step", 0)
    agent.actor.eval()
    agent.critic.eval()
    print(f"Model loaded (step {agent.step})")

    # Determine stages to evaluate
    stages = CONFIG["curriculum"]["stages"]
    if args.stage is not None:
        stages = [stages[args.stage]]
        stage_names = [stages[0]["name"]]
    else:
        stage_names = [s["name"] for s in stages]

    print(f"Evaluating {len(stages)} stage(s): {stage_names}")
    print(f"Episodes per stage: {args.episodes}, Max steps: {args.max_steps}\n")

    # Run evaluation
    for i, stage_cfg in enumerate(stages):
        print(f"=== {stage_cfg['name']} ===")
        metrics_list = evaluate_stage(agent, stage_cfg, args.episodes,
                                      args.max_steps, args.seed + i)
        # Summary
        avg_goals = np.mean([m["total_goals"] for m in metrics_list])
        avg_coll = np.mean([m["total_collisions"] for m in metrics_list])
        avg_r = np.mean([m["avg_reward"] for m in metrics_list])
        avg_v = np.mean([m["avg_speed"] for m in metrics_list])
        max_v = np.max([m["max_speed"] for m in metrics_list])
        goal_times = [m["avg_goal_time"] for m in metrics_list if m["avg_goal_time"] > 0]
        avg_t = np.mean(goal_times) if goal_times else 0.0
        print(f"Summary: goals={avg_goals:.1f}, collisions={avg_coll:.1f}, "
              f"avg_reward={avg_r:.3f}, avg_speed={avg_v:.2f}, "
              f"max_speed={max_v:.2f}, avg_goal_time={avg_t:.1f}\n")


if __name__ == "__main__":
    main()
