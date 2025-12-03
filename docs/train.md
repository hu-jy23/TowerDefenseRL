# 训练脚本说明（train.py）

## 核心入口
- `make_env(random_maps_path, seed_value, episode_gap, run_prefix)`：创建环境，可选随机地图包装（RandomMapWrapper）。录像/Monitor 包装目前注释掉，需录像时解除 `wrap_env` 行。
- `make_model(algo, env, load_model_path, tensorboard_log)`：
  - `ppo` → MaskablePPO（带 action mask）。
  - `dqn`/`dqn_sb3` → SB3 DQN，自动用 `FlattenMultiDiscreteAction` 压平成 Discrete。
- `main(load_model_path, random_maps_path, algo)`：组装环境、回调、训练并保存模型与最佳动作序列。

## 关键参数（文件顶部）
- `hours_to_train`：训练小时数，用于估算总步数。
- `mean_time_fps`、`mean_episode_steps`：估计值，用于计算 `training_steps` 和录像间隔 `episode_recording_gap`。
- `env_name`、`seed`：环境 ID 与随机种子。

## CLI 用法
```bash
# 默认 PPO
python train.py
# 使用随机地图
python train.py --random-maps custom-maps.json
# 加载旧模型继续训
python train.py --load-model ./models/<run>/<algo>_tower_defense.zip
# 切换算法（当前支持 ppo, dqn_sb3/dqn）
python train.py --algo dqn_sb3
```

## 回调
- `CheckpointCallback`：每三分之一进度存一份 `models/<run>/checkpoints/<algo>_tower_defense_*.zip`。
- `TensorboardInfoCallback`：写入波数与塔类型均值（TensorBoard 下 `rollout/custom/*`）。
- `SaveAgentActionsCallback`：保存到达最高波数的 episode 动作序列至 `models/<run>/best_episode_actions.json`。

## 录像/Monitor（可选）
- `wrap_env` 组合 `Monitor` + `RecordVideo` + `Autoreset`，视频存 `models/<prefix>/videos/`。
- 默认在 `make_env` 里被注释；需要录像时取消注释，或改成按需触发。

## 续训与 TensorBoard 复用
- 想把日志写入已有 run，可在 `model.learn` 传 `tb_log_name="PPO_1"`。
- 负载旧模型继续训练时，可同时指定 `--load-model` 与 `tb_log_name`，保持时间步连续 (`reset_num_timesteps` 基于是否加载)。
