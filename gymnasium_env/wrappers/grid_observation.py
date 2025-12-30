import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GridObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 获取地图尺寸
        self.h = 12 # 600 // 50
        self.w = 18 # 900 // 50
        self.c = 5  # 5个通道
        
        # 定义新的观测空间 (0.0 到 1.0 的浮点数张量)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.c, self.h, self.w), dtype=np.float32
        )

    def observation(self, obs):
        # 这里 obs 是原环境的一维向量，很难还原。
        # 建议直接访问 self.env.unwrapped.game_state 来获取原始 JSON 数据构建矩阵
        game_state = self.env.unwrapped.game_state
        game_info = self.env.unwrapped.game_info
        
        grid = np.zeros((self.c, self.h, self.w), dtype=np.float32)
        
        # Channel 0: Path
        # Channel 1: Towers
        # ... (填充逻辑)
        
        return grid