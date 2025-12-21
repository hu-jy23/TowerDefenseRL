# Tower Defense Game API 服务接口文档

## 概述

Tower Defense Game API 提供了一系列 RESTful 接口，用于获取游戏信息、控制游戏状态和渲染游戏画面。以下是主要接口及其返回参数的详细说明。

## 接口列表

### 1. GET /info
**描述**: 获取游戏的全局固定参数，用于构建动作空间和观察空间。

**返回类型**: JSON 对象 (GameInfo)

**字段说明**:

- `max_global_info`: 全局最大信息
  - `gameTime`: number - 游戏时间的最大值（秒），例如 1300 秒足够到达第 50 波
  - `waveNumber`: number - 波数的最大值，例如 50（实际可能无法达到，因为敌人生命值会过高）
  - `money`: number - 金币的最大值，例如 999
  - `lives`: number - 生命值的最大值，等于初始生命值
  - `gameOver`: boolean - 游戏结束标志，始终为 false（表示最大情况下的游戏未结束）

- `actions`: Action[] - 可用的动作示例
  - 动作类型包括：
    - `{ type: 'NONE' }` - 无动作
    - `{ type: 'BUILD_TOWER', towerType: TowerType, position: { x: number, y: number } }` - 建造塔动作
      - `type`: string - 动作类型，值为 'BUILD_TOWER'
      - `towerType`: TowerType - 要建造的塔类型（如 'archer', 'cannon', 'sniper'）
      - `position`: Position - 建造位置
        - `x`: number - X坐标
        - `y`: number - Y坐标

- `map`: 地图信息
  - `width`: number - 地图宽度（像素）
  - `height`: number - 地图高度（像素）
  - `cell_size`: number - 网格单元大小（像素）
  - `path_length`: number - 路径长度（单元格数）
  - `path_cells`: Position[] - 路径上的所有单元格位置
  - 每个元素为 Position 对象：
    - `x`: number - 单元格的 X 坐标
    - `y`: number - 单元格的 Y 坐标

- `towers`: Tower[] - 所有塔类型的信息
  - `type`: TowerType - 塔的类型（如 'archer', 'cannon', 'sniper'）
  - `range`: number - 塔的攻击范围（像素）
  - `dps`: number - 每秒伤害值（damage per second）
  - `cost`: number - 建造成本
  - `unlock_wave`: number - 解锁波数
  - `blast_radius`: number - 爆炸半径（像素），0表示单体攻击

- `slower_tower_sample`: Tower - 最慢塔类型的样本（根据攻击冷却时间 attackCooldown 选择，冷却时间越长攻击越慢）
  - 根据当前 GameConfig 配置，最慢的塔类型是 'cannon'（攻击冷却时间 2 秒，与 'sniper' 相同，但代码中选择第一个遇到的）
  - `type`: TowerType - 塔类型，值为 'cannon'
  - `position`: Position - 位置（示例值 {x: 0, y: 0}）
    - `x`: number - X坐标，值为 0
    - `y`: number - Y坐标，值为 0
  - `attackCooldown`: number - 攻击冷却时间（秒），值为 2

- `waves`: 波次信息
  - `wave_delay`: number - 波次之间的延迟（秒）
  - `spawn_delay`: number - 敌人生成间隔（秒）
  - `max_enemies`: number - 单波最大敌人数量
  - `enemy_types`: EnemyType[] - 所有敌人类型
  - `slower_enemy_sample`: Enemy - 最慢敌人类型的样本
    - `type`: EnemyType - 敌人类型
    - `fullHealth`: number - 满血生命值
    - `currentHealth`: number - 当前生命值
    - `currentSpeed`: number - 当前速度
    - `position`: Position - 位置
    - `direction`: Direction - 方向
    - `currentWaypointIndex`: number - 当前路径点索引
    - `pathProgress`: number - 路径进度（0-1）

### 2. POST /reset
**描述**: 重置游戏状态到初始状态。

**返回类型**: JSON 对象 (GameState)

**字段说明**:
- `gameTime`: number - 游戏时间（秒），重置后为 0
- `waveNumber`: number - 当前波数，重置后为 0
- `money`: number - 当前金币数量，重置后为初始值
- `lives`: number - 当前生命值，重置后为初始值
- `gameOver`: boolean - 游戏是否结束，重置后为 false
- `enemies`: Enemy[] - 当前地图上的敌人列表
  - `type`: EnemyType - 敌人类型（如 'basic', 'fast', 'tank'）
  - `fullHealth`: number - 敌人满血生命值
  - `currentHealth`: number - 敌人当前生命值
  - `currentSpeed`: number - 敌人当前移动速度（像素/秒）
  - `position`: Position - 敌人当前位置
    - `x`: number - X坐标
    - `y`: number - Y坐标
  - `direction`: Direction - 敌人移动方向
    - `dx`: number - X方向分量（-1, 0, 1）
    - `dy`: number - Y方向分量（-1, 0, 1）
  - `currentWaypointIndex`: number - 当前目标路径点的索引
  - `pathProgress`: number - 在当前路径段上的进度（0-1）
- `towers`: Tower[] - 当前地图上的塔列表
  - `type`: TowerType - 塔类型（如 'archer', 'cannon', 'sniper'）
  - `position`: Position - 塔的位置
    - `x`: number - X坐标
    - `y`: number - Y坐标
  - `attackCooldown`: number - 当前攻击冷却时间（秒）

### 3. POST /step
**描述**: 执行一个游戏步骤（动作），更新游戏状态。

**请求体**: Action 对象（例如 {type: 'BUILD_TOWER', towerType: 'archer', position: {x: 10, y: 20}} 或 {type: 'NONE'}）

**返回类型**: JSON 对象 (GameState)

**字段说明**: 同 POST /reset，返回当前游戏状态的快照。

### 4. 其他接口

- **GET /**: 返回字符串 "Tower Defense Game API"
- **POST /set-map**: 设置新的地图路径点，返回路径点数组
- **GET /render**: 返回游戏画面的 PNG 图像缓冲区（仅在游戏未结束时可用）

## 数据类型说明

- `Position`: {x: number, y: number} - 二维坐标
- `Direction`: {dx: number, dy: number} - 方向向量
- `Enemy`: 包含类型、健康、速度、位置等信息的敌人对象
- `Tower`: 包含类型、位置、攻击冷却时间的塔对象
- `Action`: 游戏动作，如建造塔或无动作