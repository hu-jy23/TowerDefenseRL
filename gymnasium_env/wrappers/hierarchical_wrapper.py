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
        # 注意：这里假设 tower_types 的顺序是固定的，通常是 [Archer, Cannon, Sniper]
        # 最好根据 type 字段来建立映射更稳健，但这里先按索引
        self.idx_to_tower = {}
        for idx, tower in enumerate(self.tower_types):
             # 动作 1 对应 tower_types[0], 动作 2 对应 tower_types[1]...
             self.idx_to_tower[idx + 1] = tower

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
        # 原始环境需要的动作格式是 MultiDiscrete([action_type, tower_type, x, y])
        # 但这里我们直接返回字典是不行的，因为 gym.ActionWrapper 的 action() 方法
        # 应该返回原始环境 action_space 能接受的格式。
        # 
        # 等等，TowerDefenseWorldEnv 的 step() 方法接收的是 np.ndarray (MultiDiscrete)。
        # 但是 step() 内部第一件事就是解析这个 array。
        # 
        # 如果我们想让 Wrapper 能够工作，我们需要把这里的逻辑转换成
        # 原始环境能接受的 [action_index, tower_index, x, y] 数组。
        
        # 1. 找到 tower_index
        tower_index = -1
        for idx, t in enumerate(self.tower_types):
            if t["type"] == tower_config["type"]:
                tower_index = idx
                break
        
        # 2. 构造原始动作
        # action_types[0] 通常是 "BUILD_TOWER" (需要确认 env 的定义)
        # 假设 action_types 顺序是 ["BUILD_TOWER", "SELL_TOWER", "UPGRADE_TOWER", "WAIT"] (需要确认)
        # 让我们去看看 env 的定义。
        
        # 为了安全起见，我们先读取 env 的 action_types
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
        """
        启发式逻辑：遍历所有格子，找到一个能覆盖最多路径点的位置
        """
        best_pos = (None, None)
        max_coverage = -1
        tower_range = tower_config["range"]
        
        # 获取当前已经被占用的格子（避免重复造）
        occupied_positions = set()
        for t in self.env.unwrapped.game_state["towers"]:
            # 把像素坐标转回网格坐标
            gx = int(t["position"]["x"] // self.cell_size)
            gy = int(t["position"]["y"] // self.cell_size)
            occupied_positions.add((gx, gy))

        # 优化：直接从 buildable_cells 里读
        # 注意：game_info["map"] 可能没有 "buildable_cells" 字段，取决于后端实现。
        # 如果没有，我们需要遍历全图。为了稳健，我们先检查一下。
        map_info = self.env.unwrapped.game_info["map"]
        
        if "buildable_cells" in map_info:
            candidates = map_info["buildable_cells"]
        else:
            # 如果后端没给，就生成所有格子
            candidates = []
            w = map_info["width"] // self.cell_size
            h = map_info["height"] // self.cell_size
            for x in range(w):
                for y in range(h):
                    candidates.append({"x": x, "y": y})

        for cell in candidates:
            cx, cy = cell['x'], cell['y']
            
            # 1. 检查是否已被占用
            if (cx, cy) in occupied_positions:
                continue
            
            # 1.1 检查是否在路径上 (如果 buildable_cells 没过滤的话)
            # 简单的做法是检查是否在 path_cells 里
            # 但通常 buildable_cells 已经是过滤过的了。
            # 如果是手动生成的 candidates，需要检查。
            if "buildable_cells" not in map_info:
                 if self._is_on_path(cx, cy):
                     continue

            # 2. 计算覆盖率 (核心 Heuristic)
            coverage = self._calculate_coverage(cx, cy, tower_range)
            
            # 3. 更新最大值
            if coverage > max_coverage:
                max_coverage = coverage
                best_pos = (cx, cy)
        
        return best_pos

    def _is_on_path(self, x, y):
        path_cells = self.env.unwrapped.game_info["map"]["path_cells"]
        for p in path_cells:
            if p['x'] == x and p['y'] == y:
                return True
        return False

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
        # 这里的 wait action 也需要返回 array
        return self._get_wait_action_array()

    def _get_wait_action_array(self):
        # 找到 WAIT 动作的索引
        action_types = self.env.unwrapped.action_types
        wait_action_idx = 0
        for idx, at in enumerate(action_types):
            if at["type"] == "WAIT": # 或者是 "SKIP", 取决于定义
                wait_action_idx = idx
                break
        # 后面的参数无所谓
        return np.array([wait_action_idx, 0, 0, 0], dtype=np.int64)
