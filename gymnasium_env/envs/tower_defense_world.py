from copy import deepcopy
import gymnasium as gym
import requests
import math
import io
from PIL import Image
import numpy as np
from gymnasium import spaces

url = "http://localhost:3000/"

class TowerDefenseWorldEnv(gym.Env):
    """
    塔防游戏环境类，继承自 Gymnasium Env。
    通过 HTTP 请求与运行在 localhost:3000 的游戏服务器进行交互。

    Attributes:
        render_mode (str): 渲染模式，支持 "human" 或 "rgb_array"。
        game_info (dict): 从服务器获取的游戏初始配置信息（包含地图、塔类型、波次设置等）。
        action_types (list): 游戏支持的动作类型列表 (如 BUILD_TOWER, NONE)。
        tower_types (list): 游戏支持的防御塔类型列表。
        cell_size (int): 地图网格单元的像素大小。
        map_horizontal_cells (int): 地图水平方向的网格数。
        map_vertical_cells (int): 地图垂直方向的网格数。
        action_space (spaces.MultiDiscrete): 动作空间，维度为 [动作类型数, 塔类型数, x坐标数, y坐标数]。
        observation_space (spaces.Box): 观测空间，包含全局信息、塔信息和敌人信息的扁平化向量。
        max_towers (int): 地图上允许建造的最大防御塔数量 (网格总数 - 路径格数)。
        max_enemies (int): 预估的同时存在的最大敌人数量 (用于固定观测空间大小)。
        path_cells_coordinates_normalized (list[float]): 归一化后的路径单元坐标列表 [x1, y1, x2, y2, ...]。
        global_feature_count (int): 全局特征的数量 (时间, 波数, 金钱, 生命, 游戏结束, 路径坐标序列)。
        features_per_tower (int): 每个防御塔的特征数量。
        tower_feature_count (int): 所有塔特征的总数量。
        features_per_enemy (int): 每个敌人的特征数量。
        enemy_feature_count (int): 所有敌人特征的总数量。
        tower_type_to_index (dict): 塔类型名称到索引的映射 {'type_name': index}。
        enemy_type_to_index (dict): 敌人类型名称到索引的映射 {'type_name': index}。
        most_expensive_tower_cost (int): 最昂贵的塔的造价。
        max_tower_dps (float): 所有塔中最高的 DPS (每秒伤害)。
        game_state (dict): 当前的游戏状态数据。
        current_episode_actions (list): 当前 episode 执行过的动作列表，用于日志或回放。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    # define action_space and observation_space
    def __init__(self, render_mode="rgb_array"):
        """
        初始化环境，连接游戏服务器并建立动作/观测空间。

        Args:
            render_mode (str, optional): 渲染模式。默认为 "rgb_array"。
        
        Raises:
            ConnectionError: 如果无法连接到游戏服务器 (localhost:3000) 或获取游戏信息失败。
        """
        self.render_mode = render_mode
        response = requests.get(url + "info")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to get game info: {response.text}")

        self.game_info = response.json()
        """
        game_info 获取游戏的配置信息，有下列字段: 
        1. max_global_info (dict): 全局最大信息
            - gameTime (int): 游戏最大时长（秒）限制。比如 1300。
            - waveNumber (int): 最大波数限制。比如 50。
            - money (int): 归一化用的金钱上限。比如 999。
            - lives (int): 玩家生命值上限。比如 3。
            - gameOver (boolean): 游戏是否结束。比如 false。
        2. actions (list): 支持的动作类型列表
            actions = [{ "type": "NONE" }, { "type": "BUILD_TOWER", ... }]
            - 第一个元素 {"type": "NONE"} 表示无动作。
            - 第二个元素 {"type": "BUILD_TOWER", "towerType": "archer", "position": { "x": 0, "y": 0 } } 表示建塔动作。
                towerType (str): 塔类型名称，比如 "archer"。
                position (dict): 建塔位置 (position["x"], position["y"])。
        3. map (dict): 地图信息
            - width (int): 地图宽度（像素）。比如 900。
            - height (int): 地图高度（像素）。比如 600。
            - cell_size (int): 网格单元大小（像素）。比如 50。
            - path_length (float): 敌人前进路径总长度（像素）。比如 2500.0。
            - path_cells (list): 路径经过的所有坐标点列表。比如 [{"x": 75, "y": 25}, {"x": 75, "y": 75 }, ...]
        4. towers (list): 防御塔类型列表
            towers = [
                {
                    "type": "archer",       # type (str): 塔的名称，箭塔
                    "range": 125,           # range (int): 攻击半径
                    "dps": 10.0,            # dps (float): 每秒伤害 (10 伤害 / 1.0 秒间隔)
                    "cost": 20,             # cost (int): 造价
                    "unlock_wave": 0        # unlock_wave (int): 解锁波次，0 表示一开始就能造
                },
                {
                    "type": "cannon",       # type (str): 塔的名称，炮塔
                    "range": 75,
                    "dps": 37.5,            # dps (float): 每秒伤害 (75 伤害 / 2.0 秒间隔)
                    "cost": 35,
                    "unlock_wave": 4        # unlock_wave (int): 第 4 波才解锁
                },
                {
                    "type": "sniper",       # type (str): 塔的名称，狙击塔
                    "range": 175,
                    "dps": 25.0,            # dps (float): 每秒伤害 (75 伤害 / 3.0 秒间隔)
                    "cost": 50,
                    "unlock_wave": 7        # unlock_wave (int): 第 7 波才解锁
                }
            ]
        5. slower_tower_sample (dict): 最慢塔样本，用于归一化。它提供了游戏中攻击速度最慢的塔数据，作为归一化的分母。
            slower_tower_sample = {
                "type": "sniper",
                "position": {"x": 0, "y": 0},  # 这里的坐标没有意义，占位
                "attackCooldown": 3.0          # attackCooldown (float): 游戏中最大的攻击冷却时间，秒
            }
        6. waves (dict): 敌人波次信息
            - wave_delay (int): 波次间隔时间（秒）。比如 10。
            - spawn_delay (float): 敌人生成间隔时间（秒）。比如 1.2。
            - max_enemies (int): 每波敌人最大数量。比如 20。
            - enemy_types (list): 敌人类型名称列表，固定为 ["tank", "basic", "fast"]。
            - slower_enemy_sample (dict): 跑得最慢的敌人样本，用于估算最大同屏敌人数。
                "slower_enemy_sample": {
                    "type": "tank",                     # 敌人类型名称
                    "fullHealth": 150,                  # 血量上限。不同波次可能会有倍率加成。
                    "currentHealth": 150,               # 当前血量
                    "currentSpeed": 40,                 # 当前移动速度（像素/秒）
                    "position": {"x": 175, "y": 225},   # 敌人在地图上的绝对坐标（像素）。原点 (0,0) 在地图左上角。
                    "direction": {"dx": 1, "dy": 0},    # 移动方向向量，只有 4 个方向: 右 (1,0)，下 (0,1)，左 (-1,0)，上 (0,-1)
                    "currentWaypointIndex": 1,          # 敌人正在前往（或刚经过）的路径点的索引
                    "pathProgress": 0                   # 敌人走完整个路径的百分比进度，范围 0.0 - 1.0
                }
        """
        
        self.action_types = self.game_info["actions"]
        self.tower_types = self.game_info["towers"]
        self.cell_size = self.game_info["map"]["cell_size"]
        self.map_horizontal_cells = self.game_info["map"]["width"] // self.cell_size
        self.map_vertical_cells = self.game_info["map"]["height"] // self.cell_size

        self.action_space = spaces.MultiDiscrete([len(self.action_types), len(self.tower_types), self.map_horizontal_cells, self.map_vertical_cells]) # action, tower type, x, y

        self.path_cells_coordinates_normalized = self.__normalize_path_cells()
        self.max_towers = self.map_horizontal_cells * self.map_vertical_cells - self.game_info["map"]["path_length"] // self.cell_size
        self.max_enemies = self.__calculate_total_enemies()

        """
        全局特征 (5 + len(path_cells_coordinates_normalized)): 
            - game time: 游戏时间（归一化）
            - wave number: 当前波数（归一化）
            - money: 当前金钱（归一化）
            - lives: 当前玩家生命值（归一化）
            - game over: 游戏是否结束（0 或 1）
            - path cells coordinates: 路径单元坐标序列 (x1, y1, x2, y2, ...)，通过把路径坐标“喂”给神经网络，Agent 才知道每局的路径，适应不同的地形。
        """
        #self.global_feature_count = 4+self.map_horizontal_cells*self.map_vertical_cells # game time, wave number, money, game over, grid map
        self.global_feature_count = 5+len(self.path_cells_coordinates_normalized) # game time, wave number, money, lives, game over, path cells coordinates
        
        """
        每个塔的特征 (5 + 3 = 8): 
            - active: 表示这个“塔槽位”当前是有塔的（因为 RL 的输入是固定长度的数组，如果当前场上塔少于最大值，多余槽位的 active 就是 0）。
            - x, y: 网格坐标索引（归一化）
            - attack cooldown: 攻击冷却时间（归一化）
            - dps: 每秒伤害（归一化）
            - one-hot encoding type: 塔类型的独热编码，比如 [1,0,0] = archer, [0,1,0] = cannon, [0,0,1] = sniper
        """
        self.features_per_tower = 5+len(self.tower_types) # active, x, y, attack cooldown, dps, one-hot encoding type
        self.tower_feature_count = self.max_towers * self.features_per_tower
        
        """
        每个敌人的特征 (5 + 3 = 8): 
            - active: 表示这个“敌人槽位”当前是有敌人的（因为 RL 的输入是固定长度的数组，如果当前场上敌人少于最大值，多余槽位的 active 就是 0）。
            - x, y: 网格坐标索引（归一化）
            - health: 当前血量（归一化）
            - path progress: 路径进度（归一化）
            - one-hot encoding type: 敌人类型的独热编码，比如 [1,0,0] = tank, [0,1,0] = basic, [0,0,1] = fast
        """
        self.features_per_enemy = 5+len(self.game_info["waves"]["enemy_types"]) # active, x, y, health, path progress, one-hot encoding type
        self.enemy_feature_count = self.max_enemies * self.features_per_enemy

        total_features_count = self.global_feature_count + self.tower_feature_count + self.enemy_feature_count
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_features_count,),
            dtype=np.float32
        )

        self.tower_type_to_index = {tower["type"]: idx for idx, tower in enumerate(self.tower_types)}  # tower_type_to_index: {'archer': 0, 'cannon': 1, 'sniper': 2}
        self.enemy_type_to_index = {enemy_type: idx for idx, enemy_type in enumerate(self.game_info["waves"]["enemy_types"])}  # enemy_type_to_index: {'tank': 0, 'basic': 1, 'fast': 2}
        self.most_expensive_tower_cost = max(tower["cost"] for tower in self.tower_types)  # 从所有塔的配置中提取 cost 字段，找出最贵塔的造价。
        self.max_tower_dps = max(tower["dps"] for tower in self.tower_types)  # 从所有塔的配置中提取 dps 字段，找出最高 DPS。

    # reset the environment and return the initial observation and info
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        重置环境状态，并向服务器发送 /reset 请求以重置游戏逻辑。

        Args:
            seed (int | None, optional): 随机种子，用于复现。
            options (dict | None, optional): 额外的重置选项。

        Returns:
            tuple[np.ndarray, dict]: 
                - observation (np.ndarray): 初始观测向量。维度为 (total_features_count,)。
                - info (dict): 辅助信息字典，包含游戏时间、波数、塔数量等。
        
        Raises:
            ConnectionError: 如果向服务器发送重置请求失败。
        """
        super().reset(seed=seed)
        response = requests.post(url + "reset")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to reset game: {response.text}")

        """
        game_state 获取游戏的当前状态，有下列字段:
            - gameTime (float): 当前游戏时间（秒）。比如 12.300001。训练模式下，每一步 step() 固定增加 0.1 秒。
            - waveNumber (int): 当前波数。比如 1。
            - money (int): 当前玩家持有的金钱。比如 80。
            - lives (int): 当前玩家剩余生命值。比如 2。
            - gameOver (boolean): 游戏是否结束。比如 false。
            - enemies (list): 当前场上所有存活的敌人对象列表。
                每个敌人对象是一个字典，包含:
                    - type (str): 敌人类型名称，比如 "tank"。
                    - fullHealth (float): 血量上限。不同波次可能会有倍率加成。
                    - currentHealth (float): 当前血量。
                    - currentSpeed (float): 当前移动速度（像素/秒）。
                    - position (dict): 敌人在地图上的绝对坐标（像素），原点 (0,0) 在地图左上角。比如 position = {"x": 75, "y": 25}。
                    - direction (dict): 移动方向向量，只有 4 个方向: 右 (1,0)，下 (0,1)，左 (-1,0)，上 (0,-1)。
                    - pathProgress (float): 敌人走完整个路径的百分比进度，范围 0.0 - 1.0。
            - towers (list): 当前场上所有建造的防御塔对象列表。
                每个塔对象是一个字典，包含:
                    - type (str): 塔的类型名称，比如 "archer"。
                    - position (dict): 塔在地图上的绝对坐标（像素），原点 (0,0) 在地图左上角。比如 position = {"x": 150, "y": 50}。
                    - attackCooldown (float): 当前攻击剩余冷却时间（秒）。当 cooldown <= 0 时，塔可以攻击。
        """
        self.game_state = response.json()
        observation = self.__get_observation()
        self.current_episode_actions = [] # to log actions taken in the current episode
        info = self.__get_info()
        
        return observation, info

    # perform the action and return the new observation, reward, terminated, truncated, info
    def step(self, action: np.ndarray) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        执行动作，推进环境一步。

        Args:
            action (np.ndarray): 动作向量。维度为 (4,)。格式为 [action_index, tower_index, x, y]。
                - action_index: 动作类型索引 (例如 0=NONE, 1=BUILD_TOWER)。
                - tower_index: 塔类型索引。
                - x: 网格 X 坐标。
                - y: 网格 Y 坐标。

        Returns:
            tuple(np.ndarray, int, bool, bool, dict):
                - observation (np.ndarray): 执行动作后的新观测向量。
                - reward (int): 这一步获得的奖励值。
                - terminated (bool): 是否因为游戏结束（胜利/失败/条件达成）而终止。
                - truncated (bool): 是否因为时间限制（游戏时间超限）而截断。
                - info (dict): 调试信息字典。
        """
        action_index, tower_index, x, y = action  # 解包动作输入
        game_action = self.action_types[action_index]
        if game_action["type"] == "BUILD_TOWER":
            # game_action = {"type": "BUILD_TOWER", "towerType": "archer", "position": {"x": 0, "y": 0}}，后面两个是占位符，会被输入填充
            game_action["towerType"] = self.tower_types[tower_index]["type"]
            game_action["position"]["x"] = self.cell_size/2 + self.cell_size*x
            game_action["position"]["y"] = self.cell_size/2 + self.cell_size*y

        # log the action taken
        self.current_episode_actions.append(deepcopy(game_action))

        response = requests.post(url + "step", json = game_action)
        
        # 非法错误情况: 建塔位置在路径上或已被占用，或玩家资金不足
        if response.status_code != 200:
            last_observation = self.__get_observation()
            info = self.__get_info()
            return last_observation, -1, False, False, info # small penalty for illegal action (building tower on path or in occupied cell or having insufficient money)

        new_game_state = response.json()
        reward = self.__calculate_reward(new_game_state)
        self.game_state = new_game_state
        observation = self.__get_observation()
        # 游戏终止条件: 游戏失败，或达到最大波数，或金钱达到上限（应该惩罚无意义囤钱）
        terminated = new_game_state["gameOver"] or new_game_state["waveNumber"] >= self.game_info["max_global_info"]["waveNumber"] or new_game_state["money"] >= self.game_info["max_global_info"]["money"]
        truncated = new_game_state["gameTime"] >= self.game_info["max_global_info"]["gameTime"]
        info = self.__get_info(terminated or truncated)

        return observation, reward, terminated, truncated, info

    # returns the game state as an rgb array
    def render(self) -> np.ndarray:
        """
        渲染当前环境状态。
        通过请求服务器的 /render 接口获取图像。

        Returns:
            np.ndarray: 当前帧的 RGB 图像数组。维度为 (Height, Width, 3)。
                        如果渲染失败或模式不匹配，返回全黑图像。
        """
        black_frame = np.zeros((self.game_info["map"]["height"], self.game_info["map"]["width"], 3), dtype=np.uint8)
        if self.render_mode == "rgb_array":
            response = requests.get(url + "render")
            if response.status_code != 200:
                print(f"Error during render: {response.text}")
                return black_frame
            image_bytes = io.BytesIO(response.content)
            image = Image.open(image_bytes)
            rgb_array = np.array(image)
            return rgb_array
        return black_frame
    
    # just to comply with the interface
    def close(self):
        """
        关闭环境，释放资源。
        目前仅作为接口占位符。
        """
        pass

    # create an action mask to disable illegal actions
    def action_masks(self) -> np.ndarray:
        """
        生成动作掩码，用于屏蔽非法动作（如资金不足、位置非法、未解锁等）。
        主要用于 Maskable PPO 等支持动作掩码的算法。

        Returns:
            np.ndarray: 连接后的布尔掩码数组。
                结构为 [action_type_mask, tower_type_mask, x_mask, y_mask]。
                True 表示对应动作合法，False 表示非法。
        """
        action_type_mask = np.ones(len(self.action_types), dtype=bool)
        tower_type_mask = np.ones(len(self.tower_types), dtype=bool)
        x_coordinate_mask = np.ones(self.map_horizontal_cells, dtype=bool)
        y_coordinate_mask = np.ones(self.map_vertical_cells, dtype=bool)

        # disable building towers if not enough money
        cheapest_tower_type = min(self.tower_types, key=lambda t: t["cost"])
        if self.game_state["money"] < cheapest_tower_type["cost"]:
            action_type_mask[1] = False # 1 = BUILD_TOWER

        # disable building towers if they cost too much or are locked
        for idx, tower in enumerate(self.tower_types):
            if self.game_state["money"] < tower["cost"] or self.game_state["waveNumber"] < tower["unlock_wave"]:
                tower_type_mask[idx] = False

        # for illegal coordinates I can't disable the action directly because the mask is applied per-dimension so I would disable all horizontal or vertical cells
        return np.concatenate([action_type_mask, tower_type_mask, x_coordinate_mask, y_coordinate_mask])
    
    # encodes the self game state into a tensor of shape self.observation_space.shape
    def __get_observation(self) -> np.ndarray:
        """
        将当前游戏状态编码并归一化为观测张量。

        Returns:
            np.ndarray: 归一化的观测向量。维度为 (total_features_count,)。
                包含：
                - 当前全局特征 (时间, 波数, 金钱, 当前玩家剩余生命, 游戏结束标志, 路径坐标序列)
                - 当前塔特征列表 (每个塔: active (如果建造), x, y, cooldown, dps, type_one_hot)
                - 当前敌人特征列表 (每个敌人 (如果存活): active, x, y, health, path_progress, type_one_hot)
        
        Raises:
            ValueError: 如果观测空间形状未定义。
        """
        shape = self.observation_space.shape
        if shape is None:
            raise ValueError("Observation space shape is not defined")
        observation = np.zeros(shape, dtype=np.float32)

        # global features normalized
        observation[0] = self.game_state["gameTime"] / self.game_info["max_global_info"]["gameTime"]
        observation[1] = self.game_state["waveNumber"] / self.game_info["max_global_info"]["waveNumber"]
        observation[2] = self.game_state["money"] / self.game_info["max_global_info"]["money"]
        observation[3] = self.game_state["lives"] / self.game_info["max_global_info"]["lives"]
        observation[4] = self.game_state["gameOver"]
        observation[5:5+len(self.path_cells_coordinates_normalized)] = self.path_cells_coordinates_normalized
        #observation[4:4+self.map_horizontal_cells*self.map_vertical_cells] = self.__calculate_grid_map()

        # tower features normalized
        for idx, tower in enumerate(self.game_state["towers"]):
            offset = self.global_feature_count + idx * self.features_per_tower
            observation[offset] = 1 # active
            observation[offset+1] = tower["position"]["x"] / self.game_info["map"]["width"] # normalized x
            observation[offset+2] = tower["position"]["y"] / self.game_info["map"]["height"] # normalized y
            observation[offset+3] = tower["attackCooldown"] / self.game_info["slower_tower_sample"]["attackCooldown"] # normalized attack cooldown
            observation[offset+4] = self.tower_types[self.tower_type_to_index[tower["type"]]]["dps"] / self.max_tower_dps # normalized dps
            observation[offset+5+self.tower_type_to_index[tower["type"]]] = 1 # one-hot encoding type

        # enemy features normalized
        for idx, enemy in enumerate(self.game_state["enemies"]):
            offset = self.global_feature_count + self.tower_feature_count + idx * self.features_per_enemy
            observation[offset] = 1 # active
            observation[offset+1] = enemy["position"]["x"] / self.game_info["map"]["width"] # normalized x
            observation[offset+2] = enemy["position"]["y"] / self.game_info["map"]["height"] # normalized y
            observation[offset+3] = enemy["currentHealth"] / enemy["fullHealth"] # normalized health
            observation[offset+4] = enemy["pathProgress"]
            observation[offset+5+self.enemy_type_to_index[enemy["type"]]] = 1 # one-hot encoding type

        return observation

    # additional info for debugging or logging
    def __get_info(self, is_episode_over: bool = False) -> dict:
        """
        获取用于调试或日志记录的辅助信息。info 这一部分是给人看的，agent 看不到。

        Args:
            is_episode_over (bool, optional): 当前 episode 是否结束。默认为 False。如果结束，会额外包含本局的所有动作记录以便回放。

        Returns:
            dict: 包含游戏时间、波数、各类型塔的数量、以及可能的动作记录。
        """
        info = {}
        info["game_time"] = round(self.game_state["gameTime"])
        info["wave_number"] = self.game_state["waveNumber"]
        info["tower_counts"] = {tower_type["type"]: 0 for tower_type in self.tower_types}
        for tower in self.game_state["towers"]:
            info["tower_counts"][tower["type"]] += 1
        if is_episode_over:
            info["episode_actions"] = deepcopy(self.current_episode_actions)

        return info

    def __normalize_path_cells(self) -> list[float]:
        """
        归一化地图路径单元的坐标。

        Returns:
            list[float]: 归一化后的敌人路径坐标列表 [x1, y1, x2, y2, ...]。
                每个坐标值都被除以地图的宽度或高度，值域在 [0, 1] 之间。
        """
        normalized_coordinates = []
        for cell in self.game_info["map"]["path_cells"]:
            normalized_coordinates.append(cell["x"] / self.game_info["map"]["width"])
            normalized_coordinates.append(cell["y"] / self.game_info["map"]["height"])

        return normalized_coordinates
    
    def __calculate_grid_map(self) -> list[float]:
        """
        计算网格地图的简单的占位表示 (未使用)。

        Returns:
            list[float]: 扁平化的网格地图列表。
                0.0 = 空地, 0.5 = 路径, 1.0 = 塔。
        """
        grid_map = [0.0] * (self.map_horizontal_cells * self.map_vertical_cells) # 0 = empty, 0.5 = path, 1 = tower
        for cell in self.game_info["map"]["path_cells"]:
            x_index = cell["x"] // self.cell_size
            y_index = cell["y"] // self.cell_size
            grid_map[y_index * self.map_horizontal_cells + x_index] = 0.5 # mark path cells (never changes)

        for tower in self.game_state["towers"]:
            x_index = tower["position"]["x"] // self.cell_size
            y_index = tower["position"]["y"] // self.cell_size
            grid_map[y_index * self.map_horizontal_cells + x_index] = 1.0 # mark tower cells

        return grid_map

    # worst case (assuming enemies remain alive, max number of enemies per wave and the slower spawns last):
    # - Time between waves: T = wave delay + max enemies per wave * spawn delay
    # - Number of actual waves: N = slower enemy time to complete path / T
    # - Number of total enemies: = N * max enemies per wave
    def __calculate_total_enemies(self) -> int:
        """
        估算游戏中最坏情况下可能同时存在的最大同屏敌人数。

        Returns:
            int: 估算的最大同屏敌人数量。
        """
        wave_delay = self.game_info["waves"]["wave_delay"]
        wave_max_enemies = self.game_info["waves"]["max_enemies"]
        spawn_delay = self.game_info["waves"]["spawn_delay"]
        
        # 计算最慢敌人走完全程所需时间
        slower_enemy_time = self.game_info["map"]["path_length"] / self.game_info["waves"]["slower_enemy_sample"]["currentSpeed"]
        
        # “第 N 波开始”到“第 N+1 波开始”的时间间隔 = wave_delay + spawn_delay * max_enemies
        # 同时存在的最大波数 = slower_enemy_time / (wave_delay + spawn_delay * max_enemies)
        # 总敌人数量 = 同时存在的最大波数 * max_enemies
        total_enemies = int(slower_enemy_time*wave_max_enemies/(wave_delay+spawn_delay*wave_max_enemies))
        
        # 如果敌人走得很快，在下一波开始前（wave_delay 期间）就已经跑出地图了，则最大同屏数量就是单波的最大敌人数量
        if slower_enemy_time < wave_delay:
            total_enemies = wave_max_enemies
            
        return total_enemies
    
    # calculate the rewards based on the new game state
    def __calculate_reward(self, new_game_state: dict) -> int:
        """
        根据新旧游戏状态计算奖励值。

        Args:
            new_game_state (dict): 执行动作后的新游戏状态。

        Returns:
            int: 计算得出的奖励值 (整数)。
                奖励机制包括：
                + 击杀敌人
                + 完成波次
                + 有效建造防御塔 (基于覆盖路径格子数量和DPS)
                - 无效建造 (未覆盖任何路径)
                - 囤积过多资金 (鼓励消费)
                - 损失生命值
                - 游戏失败
        """
        reward = 0
        old_state = self.game_state
        # + killing enemies
        reward += max(0, len(old_state["enemies"]) - len(new_game_state["enemies"]))

        # + completing waves
        if new_game_state["waveNumber"] > old_state["waveNumber"]:
            reward += new_game_state["waveNumber"]*2

        # building towers + based on coverage and type, - penalized if no coverage
        new_towers_count = len(new_game_state["towers"]) - len(old_state["towers"])
        if new_towers_count > 0:
            for i in range(new_towers_count):
                tower = new_game_state["towers"][-i-1] # new towers are at the end of the list，比如 tower = {"type": "archer", "position": {"x": 150, "y": 50}, "attackCooldown": 5}
                tower_info = self.tower_types[self.tower_type_to_index[tower["type"]]]
                path_coverage = self.__count_path_cells_in_range(tower)  # path_coverage: 在防御塔攻击范围内覆盖的路径单元数量
                if path_coverage == 0:
                    reward -= 30
                else:
                    reward += tower_info["cost"] * tower_info["dps"] * path_coverage / 100

        # - hoarding money uselessly
        if new_game_state["money"] > self.most_expensive_tower_cost:
            reward -= (new_game_state["money"] - self.most_expensive_tower_cost)

        # - lives lost
        reward -= (old_state["lives"] - new_game_state["lives"]) * 20

        # - game over, for the illegal actions the penalty is given in step()
        if new_game_state["gameOver"]:
            reward -= 1000

        return round(reward)

    # counts how many path cells are in range of the tower
    def __count_path_cells_in_range(self, tower: dict) -> int:
        """
        计算防御塔攻击范围内覆盖的路径单元数量。
        用于计算建塔奖励，鼓励将塔建在能覆盖更多路径的位置。

        Args:
            tower (dict): 防御塔的状态字典，包含位置和类型信息。比如 tower = {"type": "archer", "position": {"x": 150, "y": 50}, "attackCooldown": 5}。

        Returns:
            int: 覆盖的路径单元数量。
        """
        count = 0
        tower_index = self.tower_type_to_index[tower["type"]]
        tower_range = self.game_info["towers"][tower_index]["range"]

        # create a bounding box to quickly discard most path cells
        min_x = tower["position"]["x"] - tower_range
        max_x = tower["position"]["x"] + tower_range
        min_y = tower["position"]["y"] - tower_range
        max_y = tower["position"]["y"] + tower_range

        for path_cell in self.game_info["map"]["path_cells"]:
            # check if the cell is within the bounding box
            if min_x < path_cell["x"] < max_x and min_y < path_cell["y"] < max_y:
                distance = (tower["position"]["x"] - path_cell["x"])**2 + (tower["position"]["y"] - path_cell["y"])**2
                if distance < tower_range**2:
                    count += 1

        return count