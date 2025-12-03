import gymnasium as gym
import numpy as np
import requests

url = "http://localhost:3000/"

class RandomMapWrapper(gym.Wrapper):
    def __init__(self, env, map_list: list[dict]):
        super().__init__(env)
        self.map_list = map_list  # 存下所有可选的地图数据

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # 每次重置环境时，找到最底层的游戏环境 TowerDefenseWorldEnv，使用它内部的随机数生成器，生成一个随机整数。
        selected_map_index = self.env.unwrapped.np_random.integers(0, len(self.map_list))
        
        # 发送 HTTP POST 请求将选中的地图数据传给游戏服务器。
        response = requests.post(url + "set-map", json=self.map_list[selected_map_index]["waypoints"])
        if response.status_code != 200:
            print(f"Error setting map: {response.text}")

        # 调用原环境的 reset() 方法，完成环境重置。
        return self.env.reset(seed=seed, options=options)