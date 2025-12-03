# TensorBoard 使用说明

## 日志来源与结构
- 日志目录：`logs/`，每次 `model.learn()` 会创建/复用一个 run 目录（默认名如 `PPO_1`），其中有 `events.out.tfevents...`。
- 数据来源：
  - SB3 自带：`rollout/ep_rew_mean`、`rollout/ep_len_mean` 等来自 `Monitor`。
  - 自定义回调 `custom_callbacks/tensor_board_info.py`：`rollout/custom/ep_wave_number_mean`（平均到达波数）、`rollout/custom/ep_<tower>_count_mean`（各塔平均建造数）。
  - 算法内部：PPO 的 `train/entropy_loss`、`train/value_loss`、`train/policy_gradient_loss`、`train/approx_kl`、`train/clip_fraction`、`train/explained_variance`；DQN 的 `train/loss`、`train/learning_rate`。
- 录像/监控：`models/<run>/videos/`（RecordVideo），`models/<run>/monitor.csv`（原始 episode 统计）。

## 启动方式
1) 训练过程中或结束后均可查看：
```bash
cd TowerDefenseRL
tensorboard --logdir ./logs/  # 默认端口 6006
```
2) 浏览器打开 `http://localhost:6006`。如需远程访问，添加 `--bind_all` 并在安全前提下做端口转发。

## 核心曲线解读（策略好坏优先）
- `rollout/ep_rew_mean`：平均总奖励，越高越好，直观绩效曲线。
- `rollout/custom/ep_wave_number_mean`：平均到达波数，直接刻画存活/推进能力。
- `rollout/ep_len_mean`：平均步数/时长，配合奖励判断是“苟”还是“真提升”。
- `rollout/custom/ep_<tower>_count_mean`：各塔平均数量，观察策略偏好/经济使用。
- 训练健康度（PPO）：
  - `train/approx_kl`、`train/clip_fraction`：过大可能梯度过猛；过小可能学习停滞。
  - `train/value_loss`、`train/explained_variance`：价值估计质量，长期下降/负值提示需要调参或奖励归一。
  - `train/entropy_loss`：探索度；过低可能过早收敛，过高可能策略发散。
- 训练健康度（DQN）：`train/loss` 是否稳定收敛，`rollout/ep_rew_mean` 是否随之上升。
- 性能/进度：`time/total_timesteps`、`time/fps` 仅作监控。

## 续训/复用同一 TB run
- 想把新训练写入同一个 run，可在 `model.learn(...)` 里传 `tb_log_name="PPO_1"`（或目标 run 名），避免生成新编号。
- 加载旧模型继续训：`python train.py --load-model <path>`，如需沿用旧日志，配合上面的 `tb_log_name`。

## 快速排查思路
- 奖励/波数停滞且 `entropy_loss` 很低：提高探索（学习率/entropy_coef）。
- 奖励波动大且 `approx_kl`/`clip_fraction` 飙升：降低学习率或调低 `clip_range`。
- `value_loss` 长期高企：检查奖励尺度或价值网络结构；适当调低学习率。
