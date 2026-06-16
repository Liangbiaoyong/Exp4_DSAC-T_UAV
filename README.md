# 多固定翼无人机点云避障与最速寻路

**DSAC-T + Box2D** | 深度强化学习 | 多智能体协同

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Box2D](https://img.shields.io/badge/Box2D-2.3.8-green)](https://github.com/pybox2d)

---

## 📋 项目概述

在 50×50m 二维连续空间中，5 架固定翼无人机依靠**前向扇形点云感知**（120° FOV，60 条射线），在**无全局地图**的条件下进行**避障导航**。系统使用 **DSAC-T**（Distributed Soft Actor-Critic with Three Refinements）算法进行端到端训练。

### 核心特性

| 特性 | 说明 |
|------|------|
| 🚁 运动学 | Bicycle model，惯性效应，速度相关转弯半径 |
| 📡 感知 | 120° 扇形点云，指数衰减探测概率，高斯噪声，鬼点 |
| 🗺️ 建图 | 40×40m 局部占用栅格，0.25m 分辨率，Bresenham 更新 |
| 🎯 跟踪 | DBSCAN 聚类 → 卡尔曼滤波 → Hungarian 匹配 |
| 🧠 算法 | DSAC-T（分位数分布 Critic，3 项优化） |
| 🎬 可视化 | Matplotlib 实时渲染，自动生成演示视频 |

---

## 📁 项目结构

```
.
├── env/
│   └── quadrotor_env.py      # Gym 环境：Box2D 物理 + 感知 + 建图 + 跟踪
├── docs/                      # 文档
│   ├── architecture.md        # 系统架构说明
│   └── training-guide.md      # 训练与调参指南
├── references/                # 参考文献
│   ├── 实验四指导书.pdf
│   └── 带有协方差矩阵的卷积神经网络在人体运动识别中的应用.pdf
├── config.py                  # 统一超参数配置
├── networks.py                # Actor + Distributed Critic 网络
├── dsac_t.py                  # DSAC-T 算法核心
├── train.py                   # 训练脚本（含自动 Demo 渲染）
├── demo.py                    # 可视化演示脚本
├── workflow.js                # Claude Code Workflow 编排
├── requirements.txt           # Python 依赖
├── CLAUDE.md                  # Claude Code 项目指引
├── README.md                  # 本文件
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
# 完整训练（16 个并行环境，1000 万步）
python train.py --num_envs 16 --total_steps 10000000

# 恢复训练
python train.py --load_checkpoint checkpoints/model_20260612_120000.pth

# 短训测试（5000 步）
python train.py --num_envs 4 --total_steps 5000
```

训练过程中，系统会：
- 每 **10 分钟**自动保存模型 checkpoint
- 每 **5 分钟**渲染一个演示视频到 `demo_clips/`

### 演示

```bash
# 使用最新 checkpoint 交互式演示
python demo.py

# 指定 checkpoint + 批量演示
python demo.py --load_checkpoint checkpoints/model_xxx.pth --max_episodes 5

# 无界面模式（保存视频文件）
python demo.py --load_checkpoint checkpoints/model_xxx.pth --headless --max_episodes 3
```

### 使用 Claude Code Workflow

```bash
# 在 Claude Code 中执行验证 workflow
# 脚本位于 workflow.js
```

---

## 🧠 算法架构

### 观测空间（每架无人机）

| 分量 | 维度 | 说明 |
|------|------|------|
| 点云 | 60 | 归一化距离值 (0~1) |
| 占用栅格 | 160×160 | 占据概率 |
| 自身状态 | 6 | [速度, 航向, 到目标距离, 目标方位角, 舵偏角, 到达标志] |
| 动态障碍物 | 25 | 最近 5 个目标 × [dx, dy, dvx, dvy, size] |

### 动作空间

2D 连续：`[a_th, delta] ∈ [-1, 1]`

- `a_th`: 油门（控制加速度）
- `delta`: 舵偏角（控制转向）

### DSAC-T 三项优化

1. **分布截断**：分位数范围裁剪到 [-C_val, C_val]
2. **Critic 正则化**：L2 权重正则化
3. **软目标更新**：Polyak 平均 (tau=0.005)

---

## ⚙️ 配置管理

所有超参数集中在 `config.py` 的 `CONFIG` 字典中，训练时可通过 CLI 参数覆盖：

```bash
python train.py --num_envs 32 --total_steps 5000000 --save_interval_min 5 --demo_interval_min 3
```

---

## 📊 训练监控

训练日志保存在 `logs/` 目录，包含：
- 步数 / 总步数
- 经验回放缓冲区大小
- Alpha 温度系数
- 训练速度（steps/s）
- 运行时间

演示片段保存到 `demo_clips/`，格式为 `.gif` 和 `.mp4`。

---

## 📚 参考文献

- [实验四指导书](references/实验四指导书.pdf)
- 带有协方差矩阵的卷积神经网络在人体运动识别中的应用

---

## 📝 License

This project is developed for educational purposes.
