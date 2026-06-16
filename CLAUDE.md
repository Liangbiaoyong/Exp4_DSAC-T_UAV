# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

### 多四旋翼无人机点云避障与最速寻路（DSAC-T）

A multi-UAV obstacle avoidance and shortest-path navigation system in a 50×50m 2D continuous space. Five quadrotor UAVs with forward-facing fan-shaped point-cloud perception must navigate to random goals without global maps, using onboard point-cloud mapping and dynamic object tracking. The core algorithm is **DSAC-T** (Distributed Soft Actor-Critic with Three Refinements).

## Project Structure

```text
.
├── env/
│   └── quadrotor_env.py      # Gymnasium env: quadrotor kinematics, perception, mapping, tracking
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

- **Physics engine**: Custom Gym environment (Gymnasium).
- **UAV kinematics**: Quadrotor 2D point-mass model. Control: desired body acceleration [ax, ay] ∈ [-1,1]. Physics: 1st-order inertial lag (tau=0.15s) simulating motor response, quadratic air drag, speed clamped at 8 m/s, max acceleration 6 m/s². UAVs start from rest.
- **Perception**: 120° forward fan, 80 rays with exponential detection probability and Gaussian noise, plus ghost points (0.2% probability). World boundaries (walls) are detected as obstacles.
- **Local mapping**: 40×40m occupancy grid at 0.25m resolution (160×160), updated via Bresenham line-of-sight. Shifts with UAV (rolled edges cleared to 0.5).
- **Dynamic tracking**: DBSCAN clustering → Kalman filter (constant velocity) → Hungarian matching.
- **Algorithm**: DSAC-T (distributed critic with quantile distribution, 32 quantiles). Actor outputs Gaussian policy squashed through Tanh.
- **Reward**: `r = step_penalty + guidance_reward + goal_reward + collision_penalty`. Guidance reward (0.3 × distance delta) provides dense shaping signal. +10 for goal, -0.01 per step, -C_coll * beta for collision. Beta capped at 5.0.
- **Temperature alpha**: Learned via gradient descent. Controls explore/exploit trade-off. High (~1.0) = exploratory, low (~0.1) = deterministic.
- **Collision handling**: Collided UAV resets to random safe position (goal unchanged), occupancy grid cleared. Boundary contact also counts as collision. Max 2000 steps.
- **Safe goal placement**: New goals placed ≥3m from obstacles and ≥3m from world boundaries.
- **Training**: AdamW (lr=3e-4, weight decay 1e-4), batch 256, replay buffer 1e5 (uint8 compressed), update every 50 env steps, gamma=0.99.
- **Checkpoint**: Every 10 min auto-save to `./checkpoints/model_step{step}_{YYYYMMDD_HHMMSS}.pth` (keeps latest 10). Auto-resume from latest checkpoint on start.
- **Demo rendering**: Every 5 min during training, auto-renders a demo clip (1 episode) to `demo_clips/`.

## Architecture Details

### Environment (quadrotor_env.py)

- **Quadrotor kinematics**: 2D point-mass model with realistic physics. State: (x, y, vx, vy, ax_actual, ay_actual). Control: desired body acceleration [ax, ay] ∈ [-1,1] → scaled to max_acc (6 m/s²). 1st-order inertial lag (tau=0.15s) on acceleration. Quadratic air drag. Speed clamped at max_speed (8 m/s). Heading `psi` derived from velocity direction.
- **Point cloud**: 80 rays in 120° FOV. Each ray: detection prob = exp(-d/D0) with D0=10 (was 5), noise N(0, k*d) with k=0.05 (halved), ghost points at P=0.002 (reduced). World boundaries (walls) detected as obstacles.
- **Local occupancy grid**: 160×160 grid (40m × 40m, 0.25m/cell). Bresenham update: -0.08 free / +0.5 occupied, clamped [0,1]. Shifts with UAV (rolled edges cleared to 0.5).
- **Dynamic tracking**: DBSCAN (eps=0.5, min_samples=3) → Kalman filter (constant velocity, state [x,y,vx,vy]) → Hungarian matching (Mahalanobis distance). Lost >2 frames → delete.
- **Collision detection**: UAV-UAV, UAV-obstacle, and **UAV-boundary** collisions are all detected. Collided UAV reset to random safe position (goal unchanged), occupancy grid cleared.

### Networks (networks.py)

- **Actor**: CNN (3 layers: 32→64→64, 3×3, stride 2, ReLU) for 160×160×1 grid map + MLP (128→128) for concatenated [point cloud(80), state(6), obstacles(25)]. Combined → FC 256→256 → mu/sigma heads → Tanh squashed Gaussian.
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
| Point cloud       | 80      | Normalized distance values (0~1)                                  |
| Grid map          | 160×160 | Occupancy probabilities                                           |
| Self state        | 6       | [v_norm, sin(psi), cos(psi), d_goal_norm, theta_goal_norm, success_history]                 |
| Dynamic obstacles | 25      | Nearest K=5 objects, each [dx, dy, dvx, dvy, size], zero-padded   |

### Action Space

2D continuous: [ax, ay] ∈ [-1, 1].

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
├── env/                  # Environment module (quadrotor_env.py only)
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
- **Environment code** goes in `env/quadrotor_env.py` — do not create additional files in `env/`
- **No notebook files** — this project uses standalone Python scripts
- **No `src/` or `lib/` directories** — keep root clean

### Documentation Rules

- All documentation goes in `docs/` as Markdown files
- **Documentation must be updated in the same commit as the code change** — no "update docs later"
- Major architectural changes must be reflected in `docs/architecture.md`
- New commands or config options go in `docs/training-guide.md`
- Config changes (add/modify keys in `config.py`) must update the relevant config table in docs
- Parameter value changes (e.g. num_rays, buffer_capacity, lr) must sync `CLAUDE.md` and `docs/architecture.md`
- When fixing a bug, check if the fix changes any documented behavior and update accordingly

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

- The remote repository is: **`https://github.com/Liangbiaoyong/Exp4_DSAC-T_UAV.git`**
- Detailed technical documentation is in `docs/architecture.md` (reward structure, alpha temperature, physics model, safe goal generation).
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
