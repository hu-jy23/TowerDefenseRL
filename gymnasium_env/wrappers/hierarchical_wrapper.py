import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HierarchicalActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 初始动作空间：1 (wait) + 塔的数量
        self.action_space = spaces.Discrete(1 + len(env.unwrapped.tower_types))
        
        # 1. 基础信息获取
        self.game_info = self.env.unwrapped.game_info
        self.map_width = self.game_info["map"]["width"]
        self.cell_size = self.game_info["map"]["cell_size"]
        self.half_cell = self.cell_size // 2  # [优化] 预先算好半个格子偏移量
        
        # 2. 塔配置缓存 (按照 unlock_wave 排序)
        self.tower_types = sorted(self.env.unwrapped.tower_types, key=lambda t: t["unlock_wave"])
        self.idx_to_tower = {i + 1: t for i, t in enumerate(self.tower_types)}
        
        # 3. 缓存每个塔的解锁波数，用于快速过滤
        self.tower_unlock_waves = {i + 1: t["unlock_wave"] for i, t in enumerate(self.tower_types)}

    def action(self, action_idx):
        # 动作 0: 挂机
        if action_idx == 0: return self._get_wait_action()

        # 动作 1-N: 造塔
        tower_config = self.idx_to_tower.get(int(action_idx))
        if not tower_config: return self._get_wait_action()
        
        # 检查塔是否已解锁（波数限制）
        current_wave = self.env.unwrapped.game_state["waveNumber"]
        if current_wave < tower_config["unlock_wave"]:
            # 未解锁的塔 -> 挂机（理论上不应该被选到，但加一层保护）
            return self._get_wait_action()

        # --- 工兵逻辑 ---
        best_x, best_y = self._find_best_position(tower_config)

        # 没钱或没地 -> 挂机
        if best_x is None or self.env.unwrapped.game_state["money"] < tower_config["cost"]:
            return self._get_wait_action()

        # 构造动作 (这里需要根据你的环境具体实现微调 action type ID)
        # 假设 BUILD_TOWER 是 1 (需根据实际环境 action_types 确认)
        return np.array([1, self._get_tower_type_id(tower_config), best_x, best_y], dtype=np.int64)

    def _find_best_position(self, tower_config):
        # [优化1] 预处理路径点：统一转为像素坐标 List
        # 这样就不用在内层循环里判断 is_pixel_coords 了
        path_pixels = self._get_path_in_pixels()
        
        # [优化2] 获取已占用格子 (Set 查找 O(1))
        occupied = {
            (int(t["position"]["x"] // self.cell_size), int(t["position"]["y"] // self.cell_size))
            for t in self.env.unwrapped.game_state["towers"]
        }

        best_pos = (None, None)
        max_hits = -1
        range_sq = tower_config["range"] ** 2
        
        # 候选格子生成 (如果有 buildable_cells 直接用，没有就生成)
        candidates = self.game_info["map"].get("buildable_cells")
        if not candidates:
            w, h = self.map_width // self.cell_size, self.game_info["map"]["height"] // self.cell_size
            candidates = [{"x": x, "y": y} for x in range(w) for y in range(h)]

        # --- 主循环 ---
        for cell in candidates:
            cx, cy = cell['x'], cell['y']
            
            # 快速过滤
            if (cx, cy) in occupied: continue
            
            # [优化3] 算出当前候选坑位的【像素中心】，直接拿去比对
            # 不在子函数里做乘法，这里算一次即可
            center_px = cx * self.cell_size + self.half_cell
            center_py = cy * self.cell_size + self.half_cell
            
            # 计算覆盖数
            hits = 0
            for px, py in path_pixels:
                if (center_px - px)**2 + (center_py - py)**2 <= range_sq:
                    hits += 1
            
            if hits > max_hits:
                max_hits = hits
                best_pos = (cx, cy)
        
        return best_pos

    def _get_path_in_pixels(self):
        """统一把路径点处理成像素坐标 [(x, y), ...]"""
        raw_path = self.env.unwrapped.game_info["map"]["path_cells"]
        if not raw_path: return []
        
        # 判断原始数据是不是像素 (看第一个点是否大得离谱)
        is_already_pixel = raw_path[0]['x'] > (self.map_width // self.cell_size) + 2
        
        if is_already_pixel:
            return [(p['x'], p['y']) for p in raw_path]
        else:
            # 如果是网格，批量转成像素中心
            return [(p['x'] * self.cell_size + self.half_cell, 
                     p['y'] * self.cell_size + self.half_cell) for p in raw_path]

    def _get_wait_action(self):
        # 简化写法，硬编码或动态查找均可，保持原有逻辑
        return np.array([0, 0, 0, 0], dtype=np.int64)

    def _get_tower_type_id(self, config):
        # 简单查找
        for i, t in enumerate(self.tower_types):
            if t["type"] == config["type"]: return i
        return 0
    
    def action_masks(self):
        """返回当前可用的动作mask，基于波数和金钱"""
        current_wave = self.env.unwrapped.game_state["waveNumber"]
        current_money = self.env.unwrapped.game_state["money"]
        
        # action_space.n = 1 (wait) + len(tower_types)
        masks = np.zeros(self.action_space.n, dtype=bool)
        
        # 动作0（挂机）总是可用
        masks[0] = True
        
        # 检查每个塔是否可用（已解锁 且 有足够钱）
        for idx, tower in self.idx_to_tower.items():
            if current_wave >= tower["unlock_wave"] and current_money >= tower["cost"]:
                masks[idx] = True
        
        return masks