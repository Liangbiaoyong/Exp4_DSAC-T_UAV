# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

### 多固定翼无人机点云避障与最速寻路（DSAC-T + Box2D）

A multi-UAV obstacle avoidance and shortest-path navigation system in a 50×50m 2D continuous space. Five fixed-wing UAVs with forward-facing fan-shaped point-cloud perception must navigate to random goals without global maps, using onboard point-cloud mapping and dynamic object tracking. The core algorithm is **DSAC-T** (Distributed Soft Actor-Critic with Three Refinements).

## Project Structure

```text
.
├── env/
│   └── fixedwing_env.py      # Gym environment: kinematics, point cloud, mapping, tracking
├── docs/                      # Documentation
│   ├── architecture.md        # System architecture
│   └── training-guide.md      # Training & tuning guide
├── references/                # Reference materials (PDFs)
├── config.py                  # All hyperparameters unified
├── networks.py                # Actor network, Distributed Critic network definitions
├── dsac_t.py                  # DSAC-T algorithm implementation (loss, update, 3 refinements)
├── train.py                   # Training script (multi-process, checkpointing, logging, auto demo)
├── demo.py                    # Demo/visualization script (Matplotlib rendering)
├── workflow.js                # Claude Code workflow orchestration
├── requirements.txt           # Dependencies
├── CLAUDE.md                  # This file
├── README.md                  # Project readme
└── .gitignore
```

## Key Design Decisions

**Do not change these without discussing with the user first and updating this file:**

- **Physics engine**: Box2D via a custom Gym environment wrapper.
- **UAV kinematics**: Bicycle model with inertial effects (no reverse, speed-dependent turn radius).
- **Perception**: 120° forward fan, 60 rays with exponential detection probability and Gaussian noise, plus ghost points (0.5% probability).
- **Local mapping**: 40×40m occupancy grid at 0.25m resolution (160×160), updated via Bresenham line-of-sight.
- **Dynamic tracking**: DBSCAN clustering → Kalman filter (constant velocity) → Hungarian matching.
- **Algorithm**: DSAC-T (distributed critic with quantile distribution, 32 quantiles). Actor outputs Gaussian policy squashed through Tanh.
- **Reward**: +10 for goal, -0.01 per step, -C_coll * beta for collision. Beta increases with success count.
- **Collision handling**: Collided UAV resets in-place (goal unchanged); episode continues for other UAVs. Max 2000 steps.
- **Training**: AdamW (lr=3e-4, weight decay 1e-4), batch 256, replay buffer 1e6, update every 50 env steps, gamma=0.99.
- **Checkpoint**: Every 10 min auto-save to `./checkpoints/model_YYYYMMDD_HHMMSS.pth`. Resume via `--resume`.
- **Demo rendering**: Every 5 min during training, auto-renders a demo clip to `demo_clips/`.

## Architecture Details

### Environment (fixedwing_env.py)

- **Fixed-wing kinematics**: Bicycle model with drag. State: (x, y, psi, v). Control: throttle a_th ∈ [-1,1], steering delta ∈ [-delta_max, delta_max].
- **Point cloud**: 60 rays in 120° FOV. Each ray: detection prob = exp(-d/D0), noise N(0, k*d), ghost points at P=0.005.
- **Local occupancy grid**: 160×160 grid (40m × 40m, 0.25m/cell). Bresenham update: -0.1 free / +0.3 occupied, clamped [0,1]. Shifts with UAV.
- **Dynamic tracking**: DBSCAN (eps=0.5, min_samples=3) → Kalman filter (constant velocity, state [x,y,vx,vy]) → Hungarian matching (Mahalanobis distance). Lost >2 frames → delete.

### Networks (networks.py)

- **Actor**: CNN (3 layers: 32→64→64, 3×3, stride 2, ReLU) for 160×160×1 grid map + MLP (128→128) for concatenated [point cloud(60), state(6), obstacles(25)]. Combined → FC 256→256 → mu/sigma heads → Tanh squashed Gaussian.
- **Distributed Critic**: Same CNN for grid map, action concatenated before final FC layers. Outputs 32 quantile values for return distribution Z(s,a).
- **Temperature alpha**: Learned log_alpha parameter.

### DSAC-T Algorithm (dsac_t.py)

Three Refinements:

1. **Distribution truncation**: Quantile range clipped to [-C_val, C_val] (default ±50).
2. **Critic regularization**: L2 penalty or gradient clipping on critic weights.
3. **Soft target update**: tau=0.005.

### Observation Space (per UAV)

| Component         | Dim     | Description                                                       |
|-------------------|---------|-------------------------------------------------------------------|
| Point cloud       | 60      | Normalized distance values (0~1)                                  |
| Grid map          | 160×160 | Occupancy probabilities                                           |
| Self state        | 6       | [v, psi, d_goal, theta_goal, delta, arrived_flag]                 |
| Dynamic obstacles | 25      | Nearest K=5 objects, each [dx, dy, dvx, dvy, size], zero-padded   |

### Action Space

2D continuous: [a_th, delta] ∈ [-1, 1].

## Commands

### Training

```bash
python train.py --num_envs 16 --total_steps 10000000 --save_interval_min 10
# Resume from checkpoint:
python train.py --load_checkpoint checkpoints/model_YYYYMMDD_HHMMSS.pth
```

### Demo / Visualization

```bash
python demo.py --load_checkpoint checkpoints/model_xxx.pth --max_episodes 5
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Config Management

All hyperparameters live in `config.py` as a single `CONFIG` dictionary. Default checkpoint path for double-click demo execution is also defined there. Training overrides via CLI args take precedence.

## File & Document Management Rules

**These rules must be followed when creating, modifying, or organizing files in this project:**

### Directory Structure

```
.
├── env/                  # Environment module (fixedwing_env.py only)
├── docs/                 # Documentation markdown files
├── references/           # Reference PDFs and external materials
├── checkpoints/          # Model checkpoints (gitignored)
├── demo_clips/           # Demo video output (gitignored)
├── logs/                 # Training logs (gitignored)
├── (root *.py)           # Core source files (config, networks, dsac_t, train, demo)
├── workflow.js           # Claude Code workflow scripts
└── CLAUDE.md             # Project guidance (this file)
```

### Source Code Rules

- **Core Python modules** go in the project root: `config.py`, `networks.py`, `dsac_t.py`, `train.py`, `demo.py`
- **Environment code** goes in `env/fixedwing_env.py` — do not create additional files in `env/`
- **No notebook files** — this project uses standalone Python scripts
- **No `src/` or `lib/` directories** — keep root clean

### Documentation Rules

- All documentation goes in `docs/` as Markdown files
- Keep docs synced with code changes — when adding a feature, update the relevant doc
- Major architectural changes must be reflected in `docs/architecture.md`
- New commands or config options go in `docs/training-guide.md`

### Reference Material Rules

- PDFs, datasheets, and external references go in `references/`
- Do not put reference PDFs in the project root
- Name reference files descriptively

### Output & Artifact Rules

- All generated outputs go to their respective directories:
  - `checkpoints/` — model files
  - `demo_clips/` — rendered videos
  - `logs/` — text logs
- These directories are gitignored — do not commit their contents
- Do not store generated files in the project root or source directories

## Git & GitHub Management

**This project uses Git for version control and GitHub for remote backups.**

### Local Git Rules

- The repository is initialized with a `.gitignore` that excludes `checkpoints/`, `demo_clips/`, `logs/`, `__pycache__/`, `.vscode/`, and other generated files.
- **Commit frequently** with descriptive messages following conventional commits format:
  - `feat:` — new feature
  - `fix:` — bug fix
  - `docs:` — documentation changes
  - `refactor:` — code restructuring
  - `chore:` — config, dependencies, tooling
- Each commit should be atomic (one logical change per commit).
- Do not commit large files (>50MB) — checkpoints are already gitignored.

### GitHub Remote Rules

- The remote repository is: **`https://github.com/26634/Exp4_DSAC-T_UAV.git`**
- Always keep local and remote in sync:
  - Before making changes: `git pull`
  - After committing: `git push`
- The `main` branch should always be in a working state.
- Phase branches (`phase/1-environment`, etc.) are used for development and merged via PR.

### Setup Commands

```bash
# First-time setup
git init
git add .
git commit -m "chore: initial project setup"
git remote add origin https://github.com/26634/Exp4_DSAC-T_UAV.git
git push -u origin main

# Daily workflow
git add <files>
git commit -m "feat: description of change"
git push
```
