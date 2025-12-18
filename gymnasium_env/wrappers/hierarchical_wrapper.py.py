import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HierarchicalActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 你的新动作空间：只有 4 个离散动作
        # 0: Wait, 1: Archer, 2: Cannon, 3: Sniper
        self.action_space = spaces.Discrete(4)
        
        # 获取底层环境的静态信息（塔的属性、地图路径）
        self.game_info = self.env.unwrapped.game_info
        self.map_width = self.game_info["map"]["width"]
        self.map_height = self.game_info["map"]["height"]
        self.cell_size = self.game_info["map"]["cell_size"]
        
        # 预计算：拿到所有塔的配置数据
        self.tower_types = self.env.unwrapped.tower_types
        # 建立 索引 -> 塔名字 的映射 (例如 1 -> "archer")
        self.idx_to_tower = {
            1: self.tower_types[0], # archer
            2: self.tower_types[1], # cannon
            3: self.tower_types[2]  # sniper
        }

    def action(self, action_idx):
        """
        这里是核心：把 RL 输出的 0-3 转换成游戏需要的复杂字典
        """
        # 动作 0: 挂机/攒钱
        if action_idx == 0:
            return self._get_wait_action()

        # 动作 1-3: 造塔
        tower_config = self.idx_to_tower.get(int(action_idx))
        if not tower_config:
            return self._get_wait_action() # 异常保护

        # --- 工兵逻辑：寻找最佳建造位置 ---
        best_x, best_y = self._find_best_position(tower_config)

        # 如果找不到位置（地图满了）或者钱不够，强制转为挂机
        current_money = self.env.unwrapped.game_state["money"]
        if best_x is None or current_money < tower_config["cost"]:
            return self._get_wait_action()

        # 构造真实的动作字典发送给游戏
        real_action = {
            "action_type": "build",
            "tower_type": tower_config["type"],
            "position": {
                "x": int(best_x * self.cell_size + self.cell_size // 2),
                "y": int(best_y * self.cell_size + self.cell_size // 2)
            }
        }
        return real_action

    def _find_best_position(self, tower_config):
        """
        启发式逻辑：遍历所有格子，找到一个能覆盖最多路径点的位置
        """
        best_pos = (None, None)
        max_coverage = -1
        tower_range = tower_config["range"]
        
        # 获取当前已经被占用的格子（避免重复造）
        # 注意：这里需要实时读取 state，稍微有点耗时，但 18x12 的地图完全没问题
        occupied_positions = set()
        for t in self.env.unwrapped.game_state["towers"]:
            # 把像素坐标转回网格坐标
            gx = int(t["position"]["x"] // self.cell_size)
            gy = int(t["position"]["y"] // self.cell_size)
            occupied_positions.add((gx, gy))

        # 遍历地图所有可能的格子
        # self.game_info["map"]["buildable_cells"] 应该包含所有可建造的网格坐标
        # 这里为了演示，我们假设直接遍历网格 (0..17, 0..11)
        # 实际上你应该去读 self.game_info["map"]["buildable_cells"]
        
        # 优化：直接从 buildable_cells 里读，不要暴力遍历全图
        buildable_cells = self.env.unwrapped.game_info["map"]["buildable_cells"] 
        # buildable_cells 格式通常是 [{"x": 0, "y": 0}, ...] (网格坐标)

        for cell in buildable_cells:
            cx, cy = cell['x'], cell['y']
            
            # 1. 检查是否已被占用
            if (cx, cy) in occupied_positions:
                continue

            # 2. 计算覆盖率 (核心 Heuristic)
            coverage = self._calculate_coverage(cx, cy, tower_range)
            
            # 3. 更新最大值
            if coverage > max_coverage:
                max_coverage = coverage
                best_pos = (cx, cy)
        
        return best_pos

    def _calculate_coverage(self, gx, gy, range_val):
        """计算 (gx, gy) 位置能覆盖多少个路径点"""
        count = 0
        # 塔的像素中心
        tx = gx * self.cell_size + self.cell_size / 2
        ty = gy * self.cell_size + self.cell_size / 2
        
        path_cells = self.env.unwrapped.game_info["map"]["path_cells"]
        range_sq = range_val ** 2
        
        for p in path_cells:
            # 路径点的像素中心
            px = p['x'] * self.cell_size + self.cell_size / 2
            py = p['y'] * self.cell_size + self.cell_size / 2
            
            dist_sq = (tx - px)**2 + (ty - py)**2
            if dist_sq <= range_sq:
                count += 1
        return count

    def _get_wait_action(self):
        return {"action_type": "wait"}