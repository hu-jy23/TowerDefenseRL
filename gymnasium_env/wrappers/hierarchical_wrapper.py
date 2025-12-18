import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HierarchicalActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 你的新动作空间：只有 4 个离散动作
        # 0: Wait, 1: Archer, 2: Cannon, 3: Sniper
        self.action_space = spaces.Discrete(4)
        
        # 获取底层环境的静态信息
        self.game_info = self.env.unwrapped.game_info
        self.map_width = self.game_info["map"]["width"]
        self.map_height = self.game_info["map"]["height"]
        self.cell_size = self.game_info["map"]["cell_size"]
        
        # 预计算：拿到所有塔的配置数据
        self.tower_types = self.env.unwrapped.tower_types
        # 建立 索引 -> 塔配置 的映射
        self.idx_to_tower = {}
        for idx, tower in enumerate(self.tower_types):
             # 动作 1 对应 tower_types[0]...
             self.idx_to_tower[idx + 1] = tower

    def action(self, action_idx):
        """
        核心：把 RL 输出的 0-3 转换成游戏需要的复杂字典
        """
        # 动作 0: 挂机
        if action_idx == 0:
            return self._get_wait_action_array()

        # 动作 1-3: 造塔
        tower_config = self.idx_to_tower.get(int(action_idx))
        if not tower_config:
            return self._get_wait_action_array()

        # --- 工兵逻辑：寻找最佳建造位置 ---
        best_x, best_y = self._find_best_position(tower_config)
        
        # [DEBUG] 打印指挥官的意图和工兵的结果
        # 如果你看到 best_x, best_y 始终是 0, 0 或 None，说明逻辑还有问题
        # print(f"[HRL-Debug] Cmd: {tower_config['type']}, Found: ({best_x}, {best_y})")

        # 如果找不到位置或者钱不够
        current_money = self.env.unwrapped.game_state["money"]
        if best_x is None or current_money < tower_config["cost"]:
            return self._get_wait_action_array()

        # --- 构造原始环境需要的动作格式 ---
        # 1. 找到 tower_index
        tower_index = -1
        for idx, t in enumerate(self.tower_types):
            if t["type"] == tower_config["type"]:
                tower_index = idx
                break
        
        # 2. 找到 BUILD_TOWER 的动作索引
        action_types = self.env.unwrapped.action_types
        build_action_idx = -1
        for idx, at in enumerate(action_types):
            if at["type"] == "BUILD_TOWER":
                build_action_idx = idx
                break
        
        if build_action_idx == -1:
             return self._get_wait_action_array()

        return np.array([build_action_idx, tower_index, best_x, best_y], dtype=np.int64)

    def _find_best_position(self, tower_config):
        best_pos = (None, None)
        max_coverage = -1  # 初始值设为 -1
        tower_range = tower_config["range"]
        
        # 获取已占用格子
        occupied_positions = set()
        for t in self.env.unwrapped.game_state["towers"]:
            gx = int(t["position"]["x"] // self.cell_size)
            gy = int(t["position"]["y"] // self.cell_size)
            occupied_positions.add((gx, gy))

        # 获取候选格子
        map_info = self.env.unwrapped.game_info["map"]
        if "buildable_cells" in map_info:
            candidates = map_info["buildable_cells"]
        else:
            candidates = []
            w = map_info["width"] // self.cell_size
            h = map_info["height"] // self.cell_size
            for x in range(w):
                for y in range(h):
                    candidates.append({"x": x, "y": y})

        # [修复] 自动判断 path_cells 是像素坐标还是网格坐标
        path_cells = self.env.unwrapped.game_info["map"]["path_cells"]
        is_pixel_coords = False
        if len(path_cells) > 0:
            # 如果坐标值很大（比如大于地图宽度的网格数），说明是像素坐标
            if path_cells[0]['x'] > (self.map_width // self.cell_size) + 5:
                is_pixel_coords = True

        for cell in candidates:
            cx, cy = cell['x'], cell['y']
            
            if (cx, cy) in occupied_positions:
                continue

            # 如果没有 buildable_cells 列表，可能需要手动检查是否在路径上
            if "buildable_cells" not in map_info:
                 if self._is_on_path(cx, cy):
                     continue

            # 计算覆盖率
            coverage = self._calculate_coverage(cx, cy, tower_range, path_cells, is_pixel_coords)
            
            # 更新最大值
            if coverage > max_coverage:
                max_coverage = coverage
                best_pos = (cx, cy)
        
        return best_pos

    def _is_on_path(self, x, y):
        path_cells = self.env.unwrapped.game_info["map"]["path_cells"]
        for p in path_cells:
            # 这里要注意，如果 path_cells 是像素坐标，这里的比较逻辑也要改
            # 简单起见，假设用 buildable_cells 就不用走这里
            if p['x'] == x and p['y'] == y:
                return True
        return False

    def _calculate_coverage(self, gx, gy, range_val, path_cells, is_pixel_coords):
        """计算 (gx, gy) 位置能覆盖多少个路径点"""
        count = 0
        # 塔中心的像素坐标 (这是对的，因为 grid -> pixel)
        tx = gx * self.cell_size + self.cell_size / 2
        ty = gy * self.cell_size + self.cell_size / 2
        
        range_sq = range_val ** 2
        
        for p in path_cells:
            if is_pixel_coords:
                # [修复] 如果已经是像素坐标，直接用！不要再乘 cell_size
                px = p['x']
                py = p['y']
            else:
                # 否则才乘
                px = p['x'] * self.cell_size + self.cell_size / 2
                py = p['y'] * self.cell_size + self.cell_size / 2
            
            dist_sq = (tx - px)**2 + (ty - py)**2
            if dist_sq <= range_sq:
                count += 1
        return count

    def _get_wait_action_array(self):
        action_types = self.env.unwrapped.action_types
        wait_action_idx = 0
        for idx, at in enumerate(action_types):
            if at["type"] == "WAIT": 
                wait_action_idx = idx
                break
        return np.array([wait_action_idx, 0, 0, 0], dtype=np.int64)