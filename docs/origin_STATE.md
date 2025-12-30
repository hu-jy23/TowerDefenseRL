# 塔防游戏强化学习状态特征向量 (Observation) 详解

本文档详细阐述塔防游戏中状态 s（observation）的特征向量组成。该向量是一个一维 NumPy 数组（`np.float32`），所有值都被归一化到 [0, 1] 区间，以确保模型输入稳定。

向量被分成 **三个主要模块**：全局特征、塔特征、敌人特征。每个模块的特征按顺序拼接，形成完整的 observation。总维度 = 全局特征数 + 塔特征数 + 敌人特征数。

## 1. 全局特征模块 (Global Features)
- **位置**：向量开头（索引 0 到 `global_feature_count - 1`）。
- **数量**：5 + `len(path_cells_coordinates_normalized)`（路径坐标点数 * 2，因为每个点有 x 和 y）。
- **目的**：描述游戏整体状态，不依赖具体塔或敌人。
- **具体特征**：
  - **gameTime (归一化游戏时间)**：`self.game_state["gameTime"] / self.game_info["max_global_info"]["gameTime"]`
    - 含义：当前游戏经过的时间（秒），用于判断游戏进度和时间压力。
    - 范围：[0, 1]，1 表示达到最大游戏时长（默认 1300 秒）。
  - **waveNumber (归一化波数)**：`self.game_state["waveNumber"] / self.game_info["max_global_info"]["waveNumber"]`
    - 含义：当前敌人波次，用于评估游戏难度和进度。
    - 范围：[0, 1]，1 表示达到最大波数（默认 50）。
  - **money (归一化金钱)**：`self.game_state["money"] / self.game_info["max_global_info"]["money"]`
    - 含义：当前玩家持有的金钱，用于判断是否能建塔。
    - 范围：[0, 1]，1 表示达到金钱上限（默认 999）。
  - **lives (归一化生命值)**：`self.game_state["lives"] / self.game_info["max_global_info"]["lives"]`
    - 含义：当前玩家剩余生命值，用于评估生存风险。
    - 范围：[0, 1]，1 表示满生命（默认 3）。
  - **gameOver (游戏结束标志)**：`self.game_state["gameOver"]`
    - 含义：游戏是否结束（胜利/失败）。0 = 进行中，1 = 已结束。
    - 范围：[0, 1]，二值特征。
  - **path_cells_coordinates_normalized (路径坐标序列)**：`self.path_cells_coordinates_normalized`
    - 含义：敌人前进路径的所有坐标点（x, y 对），归一化后。每个点：`x / map_width`, `y / map_height`。
    - 数量：路径点数 * 2（例如，50 个点 = 100 个值）。
    - 范围：[0, 1]，帮助模型理解地图布局，即使地图随机变化。

## 2. 塔特征模块 (Tower Features)
- **位置**：紧接全局特征后（索引 `global_feature_count` 到 `global_feature_count + tower_feature_count - 1`）。
- **数量**：`max_towers * features_per_tower`（每个塔槽位 7 + `len(tower_types)` 个特征）。
  - `max_towers`：地图上最大可能塔数（网格总数 - 路径格数）。
  - `features_per_tower`：7 + 塔类型数（默认 3: archer, cannon, sniper）。
- **目的**：描述场上所有防御塔的状态。使用固定长度槽位，即使实际塔少，也填充 0。
- **具体特征**（每个塔槽位重复）：
  - **active (塔是否存在)**：1 如果该槽位有塔，否则 0。
    - 含义：标记槽位是否被占用。
    - 范围：[0, 1]，二值。
  - **x (归一化塔 x 坐标)**：`tower["position"]["x"] / map_width`
    - 含义：塔在地图上的水平位置。
    - 范围：[0, 1]。
  - **y (归一化塔 y 坐标)**：`tower["position"]["y"] / map_height`
    - 含义：塔在地图上的垂直位置。
    - 范围：[0, 1]。
  - **attackCooldown (归一化攻击冷却)**：`tower["attackCooldown"] / max_cooldown`
    - 含义：塔下次攻击前的剩余冷却时间。0 = 可立即攻击。
    - 范围：[0, 1]，max_cooldown 来自最慢塔（默认 3.0 秒）。
  - **dps (归一化每秒伤害)**：`tower["dps"] / max_tower_dps`
    - 含义：塔的伤害输出能力。
    - 范围：[0, 1]，max_tower_dps 是所有塔中最高 DPS。
  - **blast_radius (归一化爆炸半径)**：`tower["blast_radius"] / max_blast_radius`
    - 含义：塔的 AOE 伤害范围。0 表示无范围伤害。
    - 范围：[0, 1]，max_blast_radius 是所有塔中最大爆炸半径。
  - **range (归一化攻击范围)**：`tower["range"] / max_tower_range`
    - 含义：塔的攻击距离。
    - 范围：[0, 1]，max_tower_range 是所有塔中最大攻击范围。
  - **type_one_hot (塔类型独热编码)**：`len(tower_types)` 位向量，例如 [1,0,0] = archer。
    - 含义：塔的类型（archer, cannon, sniper）。
    - 范围：[0, 1]，每位 0 或 1。

## 3. 敌人特征模块 (Enemy Features)
- **位置**：紧接塔特征后（索引 `global_feature_count + tower_feature_count` 到末尾）。
- **数量**：`max_enemies * features_per_enemy`（每个敌人槽位 6 + `len(enemy_types)` 个特征）。
  - `max_enemies`：估算的最大同屏敌人数（基于波次和速度）。
  - `features_per_enemy`：6 + 敌人类型数（默认 3: tank, basic, fast）。
- **目的**：描述场上所有敌人的状态。使用固定长度槽位，即使实际敌人少，也填充 0。
- **具体特征**（每个敌人槽位重复）：
  - **active (敌人是否存在)**：1 如果该槽位有敌人，否则 0。
    - 含义：标记槽位是否被占用。
    - 范围：[0, 1]，二值。
  - **x (归一化敌人 x 坐标)**：`enemy["position"]["x"] / map_width`
    - 含义：敌人在地图上的水平位置。
    - 范围：[0, 1]。
  - **y (归一化敌人 y 坐标)**：`enemy["position"]["y"] / map_height`
    - 含义：敌人在地图上的垂直位置。
    - 范围：[0, 1]。
  - **health (归一化血量)**：`enemy["currentHealth"] / enemy["fullHealth"]`
    - 含义：敌人的当前血量比例。
    - 范围：[0, 1]，1 = 满血。
  - **pathProgress (路径进度)**：`enemy["pathProgress"]`
    - 含义：敌人走完路径的百分比。
    - 范围：[0, 1]，0 = 起点，1 = 终点。
  - **currentSpeed (归一化移动速度)**：`enemy["currentSpeed"] / max_enemy_speed`
    - 含义：敌人的当前移动速度。
    - 范围：[0, 1]，max_enemy_speed 在游戏服务器里规定 250 像素/秒。
  - **type_one_hot (敌人类型独热编码)**：`len(enemy_types)` 位向量，例如 [1,0,0] = tank。
    - 含义：敌人的类型（tank, basic, fast）。
    - 范围：[0, 1]，每位 0 或 1。

## 总结
- **总维度示例**：假设 `max_towers=100`, `max_enemies=50`, 路径点数=50，塔类型=3，敌人类型=3，则总维度 ≈ 5+100 + 100*(7+3) + 50*(6+3) = 1555。
- **设计优势**：固定长度便于神经网络处理；归一化避免数值问题；模块化让模型能分别学习全局策略、塔管理和敌人应对。
- **潜在优化**：如果某些特征冗余（如路径坐标在固定地图下），可以移除；或添加新特征（如塔的攻击范围）。

此文档基于 `tower_defense_world.py` 中的 `__get_observation()` 方法生成。如需修改，请编辑该方法。