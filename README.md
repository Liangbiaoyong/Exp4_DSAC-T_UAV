# 多四旋翼无人机点云避障与最速寻路

**DSAC‑T + 课程学习** | 深度强化学习 | 多智能体协同

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-orange)](https://gymnasium.farama.org)

---

## 📋 项目概述

在 **50 m × 50 m** 的二维连续空间中，**5 架四旋翼无人机**仅依靠 **前向 120° 扇形点云**（80 条射线）和局部占据栅格地图，在**无全局地图**的条件下进行自主避障与最速寻路。  
系统采用 **DSAC‑T**（Distributed Soft Actor‑Critic with Three Refinements）算法进行端到端训练，并通过**四阶段课程学习**逐步提升任务难度。

### 核心特性

| 特性 | 说明 |
|------|------|
| 🚁 运动学 | 四旋翼质点模型：一阶惯性延迟、二次空气阻力、速度硬约束 |
| 📡 感知 | 前向 120° 扇形点云，80 条射线，指数检测概率、高斯噪声、鬼点 |
| 🗺️ 建图 | 40×40 m 局部占据栅格，0.25 m 分辨率，Bresenham 射线追踪更新 |
| 🤝 通讯 | 受限通讯模型（20 m 范围），带噪声的位置/速度共享 |
| 🧠 算法 | DSAC‑T：分位数分布 Critic + 分布截断 + Critic 正则化 + 软目标更新 |
| 📈 训练 | 四阶段课程学习：空旷世界 → 静态障碍 → 多机无通讯 → 全动态+通讯 |
| 🎬 可视化 | Matplotlib 实时渲染，自动生成 GIF 演示动画 |

---

## 🧠 网络结构

系统采用**双通道感知‑决策网络**，将空间特征与向量特征融合后分别输出策略与价值估计。

### 输入观测（每架无人机）

| 分量 | 维度 | 描述 |
|------|------|------|
| 占据栅格 | 1×160×160 | 0~1 归一化占据概率图，40 m 范围，0.25 m 分辨率 |
| 点云 | 80 | 前向 120° 扇形射线归一化距离值 |
| 自身状态 | 6 | [v_norm, sinψ, cosψ, d_goal_norm, θ_goal_norm, success_history] |
| 动态障碍物 | 25 | 最近 5 个目标 × [dx, dy, dvx, dvy, size]（通讯阶段有值） |

### 编码器

- **CNN 编码器**：三层卷积 (1→32→64→64) + **FiLM 条件调制**（注入目标方向与速度） → 自适应平均池化 8×8 → 线性投影至 **256 维**空间特征。
- **MLP 编码器**：点云+自身状态+动态障碍物共 **111 维** 拼接 → 两层全连接 (128→128) → 输出 **128 维**向量特征。

### 决策头

- **Actor**：融合特征 (384 维) → 两层全连接 (256→256) → 均值头 (2 维) + 标准差头 (2 维, Softplus) → 高斯分布采样 → Tanh 压缩至 [-1,1]。
- **Distributed Critic**：孪生网络，每路融合特征后输出 **32 个分位数**，通过分位数 Huber 回归学习价值分布。

---

## 🧮 算法简介：DSAC‑T

DSAC‑T 在 SAC 的基础上引入了三项精炼（Three Refinements）：

1. **分布截断 (Distribution Truncation)**：将 Critic 输出的分位数及目标值限制在 `[-C_val, C_val]`（C_val=50），防止碰撞惩罚等极端信号导致价值估计发散。
2. **Critic 正则化 (Critic Regularization)**：在 Critic 损失中添加 L2 权重衰减项，抑制过拟合，提升课程学习阶段间的迁移能力。
3. **软目标更新 (Soft Target Update)**：目标网络参数通过 Polyak 平均 (`τ=0.005`) 缓慢跟踪在线网络，避免目标值突变。

训练采用 **AdamW** 优化器（学习率 3e-4，权重衰减 1e-4），经验回放缓冲区容量 5 万，批量大小 256。

### 课程学习策略

| 阶段 | 无人机数 | 静态障碍物 | 动态感知/通讯 | 训练步数（配置上限） |
|------|----------|------------|--------------|---------------------|
| stage0_empty | 1 | 无 | 无 | 300k |
| stage1_obstacles | 1 | 有 | 无 | 500k |
| stage2_multi | 5 | 有 | 无（仅点云感知多机） | 1M |
| stage3_dynamic | 5 | 有 | 开启通讯与动态观测 | 2M |

每个阶段均可**早期退出**（连续 5 次平均奖励达标），实际总训练步数约 **60 万步**，最终模型在四阶段评估中均取得高目标到达率。

---

## 📁 项目结构

```text
.
├── env/
│   └── quadrotor_env.py      # Gym 环境：运动学、点云、栅格建图、DBSCAN 跟踪
├── checkpoints/              # 模型检查点（自动保存）
├── demo_clips/               # 演示 GIF 输出
├── logs/                     # 训练日志
├── config.py                 # 统一超参数配置
├── networks.py               # Actor + Distributed Critic 网络定义
├── dsac_t.py                 # DSAC‑T 算法核心（Agent + Replay Buffer）
├── train.py                  # 训练脚本（含课程学习、并行环境、自动 Demo）
├── demo.py                   # 可视化演示脚本
├── evaluate.py               # 模型评估脚本
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
└── .gitignore
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练

```bash
# 完整课程学习训练（16 个并行环境）
python train.py

# 从特定阶段开始训练
python train.py --stage 2

# 恢复训练（从 checkpoint）
python train.py --load_checkpoint checkpoints/model_stage2_multi_xxx.pth

# 自定义超参数
python train.py --num_envs 8 --save_interval_min 5 --demo_interval_min 10
```

训练过程中系统会：

- 每 **10 分钟**自动保存模型 checkpoint
- 每 **10 分钟**渲染一个演示 GIF 到 `demo_clips/`
- 日志实时输出到控制台和 `logs/` 目录

### 评估

```bash
# 评估最新模型在所有四个阶段的表现
python evaluate.py

# 评估指定 checkpoint
python evaluate.py --load_checkpoint checkpoints/model_stage3_dynamic_final_xxx.pth

# 只评估第 3 阶段
python evaluate.py --stage 3 --episodes 5
```

### 演示

```bash
# 使用最新模型生成 GIF（不弹窗）
python demo.py

# 显示图形界面
python demo.py --show

# 指定 checkpoint 和阶段
python demo.py --load_checkpoint checkpoints/model_xxx.pth --stage 3

# 自定义步数和帧率
python demo.py --max_steps 800 --fps 8 --render_interval 4
```

---

## ⚙️ 配置管理

所有超参数集中在 `config.py` 的 `CONFIG` 字典中，训练和评估时可通过 CLI 参数覆盖主要配置。  
关键配置项包括：环境尺寸、无人机数量、点云参数、网络结构、学习率、折扣因子、课程学习各阶段设置等。

---

## 📊 训练监控

训练日志 (`logs/`) 包含：

- 当前阶段与步数
- 平均奖励、碰撞次数、目标到达次数
- 温度系数 α 的实时值
- 训练速度 (steps/s) 与已用时间

演示片段 (`demo_clips/`) 为 GIF 动画，展示左图（世界视图 + 点云/通讯圈）与右图（无人机 0 的局部占据栅格），方便定性分析策略行为。
