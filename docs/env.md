# 环境接口（TowerDefenseWorldEnv）
路径：`gymnasium_env/envs/tower_defense_world.py`

## Action
- 空间：`MultiDiscrete([A, T, X, Y])`
  - `A` 动作类型（取自游戏 `actions`，含 BUILD_TOWER 等）
  - `T` 塔类型索引（取自游戏 `towers`）
  - `X, Y` 网格坐标：`map.width / cell_size` × `map.height / cell_size`
- DQN 需离散化时，用 `FlattenMultiDiscreteAction` 包装，将动作编码为单一 `Discrete(∏nvec)`。

## Observation（扁平特征向量 Box[0,1]）
- 结构：`[全局段][塔槽位段][敌槽位段]`，未占用槽位为 0。
- 全局段（5 + 2×|path_cells|）：游戏时间、波数、金钱、生命、gameOver，再加路径格坐标归一化串联。
- 塔槽位段：`max_towers × (5 + 塔类型数)`；每塔包含 active 标记、xy、攻速冷却、DPS 归一化、塔类型 one-hot。
- 敌槽位段：`max_enemies × (5 + 敌类型数)`；每敌包含 active、xy、血量比例、pathProgress、敌类型 one-hot。

## Action Mask
- 仅屏蔽：
  - 当金钱低于最便宜塔时，禁止 BUILD_TOWER 动作类型。
  - 对各塔类型：若金钱不足或未解锁则置 False。
- 坐标维度未屏蔽，非法坐标会在 `step` 返回奖励 -1。

## Reward 关键项
- + 击杀数（上一状态敌人数量减当前数量）。
- + 通过波数奖励：波号 × 2。
- + 建塔：覆盖路径越多、DPS/成本越高奖励越大；无覆盖则罚 -30。
- - 囤积过多金钱（> 最贵塔价）。
- - 失去生命：每条 -20。
- - GameOver：-1000。

## 终止条件
- `terminated`：gameOver 或 达到最大波数/金钱（游戏配置上限）。
- `truncated`：游戏时间超上限。
