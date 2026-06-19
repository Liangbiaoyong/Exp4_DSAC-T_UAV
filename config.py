"""Unified configuration for Multi-UAV DSAC-T system."""

CONFIG = {
    # =========================================================================
    # Environment
    # =========================================================================
    "world_size": 50.0,            # 50×50 m world
    "num_uavs": 5,
    "max_steps": 2000,
    "dt": 0.1,                     # physics timestep (s)

    # Quadrotor 2D point-mass model
    "uav": {
        "max_speed": 8.0,          # maximum speed (m/s)
        "max_acc": 6.0,            # maximum acceleration (m/s²)
        "drag_coef": 0.1,          # drag coefficient F_drag = drag_coef * v²
        "tau_acc": 0.15,           # acceleration 1st-order lag time constant (s)
        "init_v": 0.0,             # initial speed (start from rest)
    },

    # Point-cloud perception
    "perception": {
        "num_rays": 80,            # increased from 60 — denser scan lines
        "fov": 2.094,              # 120° in radians
        "max_range": 15.0,         # max detection range (m)
        "D0": 10.0,                # characteristic distance for detection prob (was 5.0)
        "noise_coef": 0.05,        # halved from 0.1 — more accurate readings
        "ghost_prob": 0.002,       # reduced from 0.005 — fewer ghost points
    },

    # Occupancy grid mapping
    "grid": {
        "size": 40.0,              # grid world size (m)
        "resolution": 0.25,        # m/cell
        "width": 160,              # cells (40 / 0.25)
        "height": 160,
        "free_inc": -0.08,         # Bresenham free-space increment (was -0.1)
        "occ_inc": 0.5,            # occupancy increment (was 0.3)
        "clip_min": 0.0,
        "clip_max": 1.0,
        "occ_threshold": 0.6,      # considered occupied
    },

    # Dynamic object tracking
    "tracking": {
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 3,
        "kalman_R": 0.5,           # measurement noise
        "kalman_Q": 0.1,           # process noise
        "lost_threshold": 2,       # frames before deletion
        "max_tracked": 10,
    },

    # Observation dimensions
    "obs": {
        "pointcloud_dim": 80,
        "grid_h": 160,
        "grid_w": 160,
        "state_dim": 6,            # [v_norm, sin(psi), cos(psi), d_goal_norm, theta_goal_norm, success_history]
        "dyn_obs_dim": 25,         # K=5 objects × 5 features
        "k_objects": 5,
        "object_feats": 5,         # [dx, dy, dvx, dvy, size]
    },

    # Reward
    "reward": {
        "goal_reward": 10.0,
        "step_penalty": -0.01,
        "collision_penalty_base": 1000.0,   # C_coll
        "beta_init": 1.0,
        "beta_increment": 0,           # per success
        "goal_radius": 1.5,              # distance to consider goal reached
        "guidance_scale": 0.3,           # guidance reward scale (dense shaping, was 0.1)
        "heading_scale": 0.2,            # heading alignment reward weight (cos(Δθ)-1)
        "beta_cap": 5.0,                 # collision penalty beta upper limit
    },

    # Collision
    "collision": {
        "uav_radius": 0.5,         # UAV physical radius (m)
        "obstacle_radius": 0.5,    # obstacle radius
    },

    # Communication-limited perception of other UAVs
    "comm": {
        "range": 20.0,               # communication sensing radius (m)
        "noise_pos_std": 0.15,       # received position Gaussian noise std (m)
        "noise_vel_std": 0.1,        # received velocity Gaussian noise std (m/s)
    },

    # Safe goal generation
    "safe_goal": {
        "min_dist_from_obs": 3.0,  # min distance from obstacle center when placing goal
        "margin": 3.0,            # min distance from world boundary
    },

    # =========================================================================
    # Networks
    # =========================================================================
    "network": {
        "cnn_channels": [1, 32, 64, 64],
        "cnn_kernel": 3,
        "cnn_stride": 2,
        "cnn_padding": 1,
        "cnn_pool_size": [8, 8],       # spatial grid preserved at 8×8
        "cnn_out_dim": 256,             # projected grid feature dimension
        "mlp_hidden": [128, 128],
        "fc_hidden": [256, 256],
        "action_dim": 2,
        "num_quantiles": 32,
        "log_alpha_init": 0.0,
    },

    # =========================================================================
    # DSAC-T Algorithm
    # =========================================================================
    "dsac_t": {
        "gamma": 0.99,
        "tau": 0.005,               # soft target update
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "buffer_capacity": 50_000,
        "update_every": 50,         # env steps between updates
        "updates_per_step": 1,
        "c_val": 50.0,              # distribution truncation bound
        "critic_reg": 1e-5,         # critic L2 regularization
        "target_entropy": -2.0,     # -dim_action (for auto alpha)
    },

    # =========================================================================
    # Training
    # =========================================================================
    "train": {
        "num_envs": 16,
        "total_steps": 5_000_000,
        "save_interval_min": 10,
        "demo_interval_min": 10,     # render a demo clip every 5 minutes
        "log_interval": 100,        # steps between console logs
        "eval_episodes": 1,         # episodes per demo render
        "eval_max_steps": 500,
    },

    # =========================================================================
    # Demo / Visualization
    # =========================================================================
    "demo": {
        "render_mode": "matplotlib",  # or "pygame"
        "fig_size": (8, 6),
        "trail_length": 200,          # path history points
        "predict_steps": 20,          # future prediction (2s at 0.1s dt)
        "save_frames": True,
        "fps": 10,
        "render_interval": 5,        # 每隔 N 个环境步采集一帧
    },

    # =========================================================================
    # Paths
    # =========================================================================
    "paths": {
        "checkpoint_dir": "checkpoints",
        "demo_clip_dir": "demo_clips",
        "log_dir": "logs",
    },

    # =========================================================================
    # Curriculum Learning — staged training from easy to hard
    # =========================================================================
    "curriculum": {
        "enabled": True,
        "early_exit_min_steps": 100000,   # minimum steps before early exit can trigger
        "early_exit_window": 5,          # consecutive checks before early exit triggers
        "stages": [
            {
                "name": "stage0_empty",
                "num_uavs": 1,
                "dynamic_obs": False,
                "static_obstacles": False,      # no obstacles — pure goal-reaching
                "total_steps": 300000,
                "early_exit_avg_reward": 0.15,   # per-step avg; single UAV in empty world reaches ~0.12-0.20
                "heading_scale": 0.3,            # strong heading reward in empty world
                "warmup_samples": 2000,          # 单机空旷，少量预热
            },
            {
                "name": "stage1_obstacles",
                "num_uavs": 1,
                "dynamic_obs": False,
                "static_obstacles": True,
                "total_steps": 500000,
                "early_exit_avg_reward": 0.15,   # obstacles slow down goal arrival
                "heading_scale": 0.0,
                "warmup_samples": 5000,          # 静障碍，需要更多样本
            },
            {
                "name": "stage2_multi",
                "num_uavs": 5,
                "dynamic_obs": False,
                "static_obstacles": True,
                "total_steps": 1000000,
                "early_exit_avg_reward": 0.15,   # collisions dilute reward
                "heading_scale": 0.1,
                "warmup_samples": 5000,          # 多机，更多样本
            },
            {
                "name": "stage3_dynamic",
                "num_uavs": 5,
                "dynamic_obs": True,
                "static_obstacles": True,
                "total_steps": 2000000,
                "early_exit_avg_reward": 0.15,  # hardest — low exit bar
                "heading_scale": 0.2,
                "warmup_samples": 5000,         # 最复杂阶段
            },
        ],
        "current_stage": 0,               # default starting stage index
    },
}
