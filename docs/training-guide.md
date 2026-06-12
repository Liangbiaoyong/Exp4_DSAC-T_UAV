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
| `checkpoints/` | 模型文件 model_YYYYMMDD_HHMMSS.pth | 每 10 分钟 |
| `demo_clips/` | 演示视频 (.gif + .mp4) | 每 5 分钟 |
| `logs/` | 训练日志 (.log) | 实时 |

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
