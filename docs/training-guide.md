# 训练与调参指南

## 训练命令

### 基础训练

```bash
# 标准训练（推荐）
python train.py --num_envs 16 --total_steps 10000000

# 快速测试（验证训练循环）
python train.py --num_envs 2 --total_steps 5000 --save_interval_min 1 --demo_interval_min 1
```

### 恢复训练

```bash
python train.py --load_checkpoint checkpoints/model_20260612_120000.pth
```

### 自定义参数

```bash
python train.py \
    --num_envs 32 \              # 并行环境数
    --total_steps 5000000 \       # 总步数
    --save_interval_min 5 \       # 保存间隔
    --demo_interval_min 3 \       # 演示渲染间隔
    --log_interval 200 \          # 日志间隔
    --eval_episodes 3             # 演示 episode 数
```

## 输出产物

| 目录 | 内容 | 生成频率 |
|------|------|----------|
| `checkpoints/` | 模型文件 model_step{step}_{YYYYMMDD_HHMMSS}.pth | 每 10 分钟（保留最近 10 个） |
| `demo_clips/` | 演示视频 (.gif) | 每 5 分钟（保留最近 10 个） |
| `logs/` | 训练日志 (.log) | 实时 |

## 训练日志格式

训练过程实时输出以下指标：

```
[timestamp] Step 100/500000 | Buffer: 500 | Alpha: 0.9997 | AvgReward: -0.040 | Collisions: 0 | Goals: 0 | Speed: 4 steps/s | Elapsed: 25s
```

| 字段 | 说明 |
|------|------|
| `Step` | 当前步数 / 总步数 |
| `Buffer` | 经验回放缓冲区大小（uint8 压缩） |
| `Alpha` | 温度系数（探索/利用平衡） |
| `AvgReward` | 最近 100 步平均奖励 |
| `Collisions` | 当前日志窗口内碰撞次数 |
| `Goals` | 当前日志窗口内到达目标次数 |
| `Speed` | 训练速度（steps/s） |
| `Elapsed` | 已运行时间 |

## 演示渲染说明

训练过程中每 5 分钟自动渲染一次演示视频，包含：

- **左上**：世界地图（UAV 位置、路径历史、点云射线、障碍物）
- **右列**：占用栅格 + 点云折线图
- **右下**：训练统计表

演示输出到 `demo_clips/` 目录，同时保存 GIF 和 MP4 格式。

## 演示命令

```bash
# 交互式演示
python demo.py

# 指定 checkpoint
python demo.py --load_checkpoint checkpoints/model_xxx.pth --max_episodes 5

# 无界面批量演示
python demo.py --load_checkpoint checkpoints/model_xxx.pth --headless --max_episodes 10

# 仅基准测试（不渲染）
python demo.py --no_render
```

## 配置修改

所有超参数集中在 `config.py` 的 `CONFIG` 字典中，按模块分类：

| 配置节 | 主要内容 |
|--------|----------|
| `CONFIG["uav"]` | 无人机物理参数 |
| `CONFIG["perception"]` | 点云感知参数 |
| `CONFIG["grid"]` | 占用栅格参数 |
| `CONFIG["tracking"]` | 目标跟踪参数 |
| `CONFIG["reward"]` | 奖励函数参数 |
| `CONFIG["network"]` | 网络结构参数 |
| `CONFIG["dsac_t"]` | 算法超参数 |
| `CONFIG["train"]` | 训练参数 |
| `CONFIG["demo"]` | 可视化参数 |

## 随机种子

训练默认使用 seed=42（固定 `random`、`numpy`、`torch`），保证实验结果可复现。如需不同随机序列，可在 `train.py` 中修改 `seed` 变量。
