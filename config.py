"""Unified configuration for Multi-UAV DSAC-T system."""

CONFIG = {
    # =========================================================================
    # Environment
    # =========================================================================
    "world_size": 50.0,            # 50×50 m world
    "num_uavs": 5,
    "max_steps": 2000,
    "dt": 0.1,                     # physics timestep (s)

    # UAV kinematics (bicycle model)
    "uav": {
        "v_min": 1.0,              # minimum speed (m/s)
        "v_max": 5.0,              # maximum speed (m/s)
        "v_cruise": 3.0,           # cruise speed
        "delta_max": 0.5,          # max steering angle (rad)
        "a_th_max": 1.0,           # max throttle
        "drag_coef": 0.1,          # drag coefficient
        "L": 0.5,                  # wheelbase (m)
    },

    # Point-cloud perception
    "perception": {
        "num_rays": 60,
        "fov": 2.094,              # 120° in radians
        "max_range": 15.0,         # max detection range (m)
        "D0": 5.0,                 # characteristic distance for detection prob
        "noise_coef": 0.1,         # noise std = k * d
        "ghost_prob": 0.005,       # ghost point probability
    },

    # Occupancy grid mapping
    "grid": {
        "size": 40.0,              # grid world size (m)
        "resolution": 0.25,        # m/cell
        "width": 160,              # cells (40 / 0.25)
        "height": 160,
        "free_inc": -0.1,          # Bresenham free-space increment
        "occ_inc": 0.3,            # occupancy increment
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
        "pointcloud_dim": 60,
        "grid_h": 160,
        "grid_w": 160,
        "state_dim": 6,            # [v, psi, d_goal, theta_goal, delta, arrived_flag]
        "dyn_obs_dim": 25,         # K=5 objects × 5 features
        "k_objects": 5,
        "object_feats": 5,         # [dx, dy, dvx, dvy, size]
    },

    # Reward
    "reward": {
        "goal_reward": 10.0,
        "step_penalty": -0.01,
        "collision_penalty_base": 5.0,   # C_coll
        "beta_init": 1.0,
        "beta_increment": 0.1,           # per success
        "goal_radius": 1.5,              # distance to consider goal reached
    },

    # Collision
    "collision": {
        "uav_radius": 0.5,         # UAV physical radius (m)
        "obstacle_radius": 0.5,    # obstacle radius
    },

    # =========================================================================
    # Networks
    # =========================================================================
    "network": {
        "cnn_channels": [1, 32, 64, 64],
        "cnn_kernel": 3,
        "cnn_stride": 2,
        "cnn_padding": 1,
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
        "buffer_capacity": 1_000_000,
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
        "total_steps": 10_000_000,
        "save_interval_min": 10,
        "demo_interval_min": 5,     # render a demo clip every 5 minutes
        "log_interval": 100,        # steps between console logs
        "eval_episodes": 3,         # episodes per demo render
        "eval_max_steps": 500,
    },

    # =========================================================================
    # Demo / Visualization
    # =========================================================================
    "demo": {
        "render_mode": "matplotlib",  # or "pygame"
        "fig_size": (12, 10),
        "trail_length": 200,          # path history points
        "predict_steps": 20,          # future prediction (2s at 0.1s dt)
        "save_frames": True,
        "fps": 10,
    },

    # =========================================================================
    # Paths
    # =========================================================================
    "paths": {
        "checkpoint_dir": "checkpoints",
        "demo_clip_dir": "demo_clips",
        "log_dir": "logs",
    },
}
