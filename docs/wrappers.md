# 封装器说明（gymnasium_env/wrappers）

## wrap_env
- 作用：`Monitor`(记录 episode 统计) + `RecordVideo`(按间隔录制) + `Autoreset`。
- 路径：`gymnasium_env/wrappers/wrap.py`。
- 默认在 `make_env` 中被注释，如需录像/monitor，解除注释并设定 `episode_recording_gap`、`prefix`。

## RandomMapWrapper
- 作用：在 `reset` 前随机选择给定 JSON 中的地图 waypoint，POST 到游戏服务器 `/set-map`。
- 参数：`map_list`（list[dict]，结构同 `custom-maps.json`）。
- 开启方式：`python train.py --random-maps custom-maps.json`；内部会先 `env.reset(seed=seed_value)` 保持随机序列可复现。
- 路径：`gymnasium_env/wrappers/random_map_wrapper.py`。

## FlattenMultiDiscreteAction
- 作用：把 `MultiDiscrete([A,T,X,Y])` 动作编码成单一 `Discrete(N)`，便于 DQN 这类只支持离散动作的算法。
- 编码：`id = sum(a[i] * radices[i])`，`radices[i] = prod(nvec[i+1:])`；提供 `encode`/`decode` 方法。
- 在 `make_model(algo="dqn"|"dqn_sb3")` 中自动应用；PPO 不需要。
- 路径：`gymnasium_env/wrappers/flatten_multidiscrete.py`。
