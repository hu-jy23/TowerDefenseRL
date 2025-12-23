from copy import deepcopy
import gymnasium as gym
import requests
import io
from PIL import Image
import numpy as np
from gymnasium import spaces

url = "http://localhost:3000/"

class TowerDefenseWorldEnv(gym.Env):
    """
    塔防游戏环境类 (Curriculum Learning Version)
    
    集成特性:
    1. 课程学习 (Curriculum Learning): 在 reset 时随机进入中后期局面。
    2. 动作屏蔽 (Action Masking): 在后期强制屏蔽低级塔，引导策略升级。
    3. 结果导向奖励 (Outcome-based Reward): 基于伤害量的奖励机制。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode
        try:
            # 获取游戏初始化信息
            response = requests.get(url + "info")
            if response.status_code != 200:
                raise ConnectionError(f"Failed to get game info: {response.text}")
            self.game_info = response.json()
        except Exception as e:
            print(f"Server connection error: {e}")
            raise e

        # --- 1. 基础参数 ---
        self.action_types = self.game_info["actions"]
        self.tower_types = self.game_info["towers"]
        
        # 地图尺寸处理
        self.cell_size = self.game_info["map"]["cell_size"]
        self.map_width_px = self.game_info["map"]["width"]
        self.map_height_px = self.game_info["map"]["height"]
        self.cols = self.map_width_px // self.cell_size
        self.rows = self.map_height_px // self.cell_size

        # 动作空间: [ActionType, TowerType, X, Y]
        self.action_space = spaces.MultiDiscrete([
            len(self.action_types), 
            len(self.tower_types), 
            self.cols, 
            self.rows
        ])

        # --- 2. 动态计算极值 ---
        self.max_time = self.game_info["max_global_info"]["gameTime"]
        self.max_wave = self.game_info["max_global_info"]["waveNumber"]
        self.max_money = self.game_info["max_global_info"]["money"]
        self.max_lives = self.game_info["max_global_info"]["lives"]
        
        # 遍历所有塔类型，找到最大值
        self.max_tower_dps = max(t["dps"] for t in self.tower_types)
        self.max_tower_cost = max(t["cost"] for t in self.tower_types)
        self.max_cooldown = self.game_info.get("slower_tower_sample", {}).get("attackCooldown", 3.0)
        self.max_enemy_speed = 250.0 
        self.max_enemy_health = 1.0 

        # --- 3. Observation Space ---
        self.n_channels = 9
        
        self.observation_space = spaces.Dict({
            # CNN 输入: (Channels, Height, Width)
            "map_input": spaces.Box(
                low=0.0, high=float('inf'), 
                shape=(self.n_channels, self.rows, self.cols),
                dtype=np.float32
            ),
            # MLP 输入: 全局数值
            "global_input": spaces.Box(
                low=0.0, high=1.0,
                shape=(5,), # time, wave, money, lives, game_over
                dtype=np.float32
            )
        })

        # --- 4. 静态层缓存 ---
        self.tower_specs = {t["type"]: t for t in self.tower_types}
        
        # 路径网格 (Channel 0)
        self.path_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        for cell in self.game_info["map"]["path_cells"]:
            c, r = self._to_grid(cell["x"], cell["y"])
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.path_grid[r, c] = 1.0

        # 用于计算伤害增量
        self.prev_total_health = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # === 课程学习调度器 (Curriculum Scheduler) ===
        # 30% 正常开局: 验证全流程能力
        # 40% Sniper特训 (Wave 12, Money 400): 强迫面对高甲怪，钱管够
        # 30% Cannon特训 (Wave 6, Money 200): 强迫面对密集怪群
        
        rand_val = np.random.random()
        reset_payload = {}

        if rand_val < 0: 
            # [特训模式 A: 绝境求生] -> 针对 Tank/Sniper
            # 直接跳到 Archer 流的崩溃边缘 (Wave 12)
            # 第 12 波开始前，此时手上有 252 金币
            reset_payload = {"start_wave": 12, "start_money": 252}
            # print(f"[Curriculum] Hard Mode: Wave 12, Money 252")
            
        elif rand_val < 0:
            # [特训模式 B: 密集防守] -> 针对 Fast/Cannon
            # 跳到怪群开始密集的阶段
            # 第 6 波开始前，此时手上有 98 金币
            reset_payload = {"start_wave": 6, "start_money": 98}
            # print(f"[Curriculum] Swarm Mode: Wave 6, Money 98")
            
        else:
            # [正常模式]
            reset_payload = {}
            # print(f"[Curriculum] Normal Start")

        try:
            response = requests.post(url + "reset", json=reset_payload)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to reset game: {response.text}")
            self.game_state = response.json()
            
            # 初始化血量追踪
            self.prev_total_health = self._get_total_health(self.game_state["enemies"])
            
        except Exception as e:
            print(f"Reset error: {e}")
            self.game_state = self._get_empty_state()

        self.current_episode_actions = []
        observation = self.__get_observation()
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
            center_x = (x * self.cell_size) + (self.cell_size / 2)
            center_y = (y * self.cell_size) + (self.cell_size / 2)
            
            game_action["towerType"] = self.tower_types[tower_index]["type"]
            game_action["position"]["x"] = float(center_x)
            game_action["position"]["y"] = float(center_y)

        self.current_episode_actions.append(deepcopy(game_action))

        response = requests.post(url + "step", json=game_action)
        
        # 非法错误情况: 建塔位置在路径上或已被占用，或玩家资金不足
        if response.status_code != 200:
            observation = self.__get_observation()
            info = self.__get_info()
            # 给予惩罚并保持状态
            return observation, -0.1, False, False, info

        new_game_state = response.json()
        
        # 计算奖励
        reward = self.__calculate_reward(new_game_state)
        
        # 更新状态
        self.game_state = new_game_state
        
        # 更新血量追踪，供下一帧使用
        self.prev_total_health = self._get_total_health(new_game_state["enemies"])
        
        observation = self.__get_observation()
        
        # 终止条件
        terminated = (
            new_game_state["gameOver"] or 
            new_game_state["waveNumber"] >= self.max_wave or 
            new_game_state["money"] >= self.max_money
        )
        truncated = new_game_state["gameTime"] >= self.max_time
        info = self.__get_info(terminated or truncated)

        return observation, reward, terminated, truncated, info

    def __get_observation(self) -> dict:
        """
        核心方法：生成 9 通道特征图和全局向量
        """
        # 初始化 3D 网格
        grid = np.zeros((self.n_channels, self.rows, self.cols), dtype=np.float32)
        
        # Channel 0: 路径
        grid[0] = self.path_grid

        # Mask layer (Channel 8): 初始包含路径
        build_mask_layer = self.path_grid.copy()

        # --- 处理塔 ---
        for tower in self.game_state["towers"]:
            c, r = self._to_grid(tower["position"]["x"], tower["position"]["y"])
            
            if 0 <= r < self.rows and 0 <= c < self.cols:
                # Ch 1: Presence
                grid[1, r, c] = 1.0
                
                # Mask Update
                build_mask_layer[r, c] = 1.0
                
                # 获取塔的静态属性
                t_type = tower["type"]
                t_spec = self.tower_specs.get(t_type, {})
                
                t_range = t_spec.get("range", 0)
                t_dps = t_spec.get("dps", 0)
                t_blast = t_spec.get("blast_radius", 0)
                
                # Ch 3: DPS (AOE 塔适当加倍)
                norm_dps = t_dps / self.max_tower_dps
                if t_blast > 0:
                    norm_dps *= 1.5 
                grid[3, r, c] = min(1.0, norm_dps)
                
                # Ch 4: Cooldown
                cd = max(0, tower["attackCooldown"])
                grid[4, r, c] = cd / self.max_cooldown
                
                # Ch 2: Range of Fire
                range_in_cells = t_range / self.cell_size
                
                # 计算基础权重：归一化 DPS
                weight = t_dps / self.max_tower_dps
                
                # AOE 加成：如果是有爆炸半径的塔，给予 1.5 倍权重
                if t_blast > 0:
                    weight *= 1.5
                
                # 裁剪最大值 (防止叠加后数值过大导致梯度爆炸，虽然 CNN 能抗住，但归一化更好)
                # weight = min(5.0, weight) 

                r_min = max(0, int(r - range_in_cells - 1))
                r_max = min(self.rows, int(r + range_in_cells + 2))
                c_min = max(0, int(c - range_in_cells - 1))
                c_max = min(self.cols, int(c + range_in_cells + 2))
                
                for yr in range(r_min, r_max):
                    for xc in range(c_min, c_max):
                        dist = ((yr - r)**2 + (xc - c)**2)**0.5
                        if dist <= range_in_cells:
                            grid[2, yr, xc] += weight

        # --- 处理敌人 ---
        for enemy in self.game_state["enemies"]:
            c, r = self._to_grid(enemy["position"]["x"], enemy["position"]["y"])
            
            if 0 <= r < self.rows and 0 <= c < self.cols:
                # Ch 5: Density 敌人落在这个格子的密度，每当有一个敌人落入该格子，该格子的值就增加 0.2
                grid[5, r, c] += 0.2
                
                # Ch 6: Total Health (current/full)
                full_hp = max(1, enemy.get("fullHealth", 100))
                hp_ratio = enemy["currentHealth"] / full_hp
                grid[6, r, c] += hp_ratio
                
                # Ch 7: Max Speed
                sp_ratio = enemy["currentSpeed"] / self.max_enemy_speed
                if sp_ratio > grid[7, r, c]:
                    grid[7, r, c] = sp_ratio

        # Ch 8: Build Mask
        grid[8] = build_mask_layer

        # Global Vector
        global_vec = np.array([
            self.game_state["gameTime"] / self.max_time,
            self.game_state["waveNumber"] / self.max_wave,
            min(1.0, self.game_state["money"] / self.max_money),
            self.game_state["lives"] / self.max_lives,
            1.0 if self.game_state["gameOver"] else 0.0
        ], dtype=np.float32)

        return {
            "map_input": grid,   
            "global_input": global_vec
        }

    def _to_grid(self, x_px, y_px):
        return int(x_px // self.cell_size), int(y_px // self.cell_size)
    
    def _get_total_health(self, enemies_list):
        """计算当前场上敌人总血量"""
        return sum(e["currentHealth"] for e in enemies_list)
    
    def __get_info(self, is_episode_over: bool = False) -> dict:
        """
        获取用于调试或日志记录的辅助信息。
        """
        info = {}
        info["game_time"] = round(self.game_state["gameTime"])
        info["wave_number"] = self.game_state["waveNumber"]
        
        # 统计每种塔的数量
        info["tower_counts"] = {t["type"]: 0 for t in self.tower_types}
        for tower in self.game_state["towers"]:
            if tower["type"] in info["tower_counts"]:
                info["tower_counts"][tower["type"]] += 1
                
        if is_episode_over:
            info["episode_actions"] = deepcopy(self.current_episode_actions)

        return info

    def __calculate_reward(self, new_game_state: dict) -> float:
        """
        基于伤害 (Damage-based) 的奖励函数
            + 造成伤害 (Damage Reward)
            + 击杀奖励 (Kill Reward)
            - 漏怪惩罚 (Leak Penalty)
            + 波次推进奖励 (Wave Progression Reward)
            - 游戏结束惩罚 (Game Over Penalty)
        """
        reward = 0.0
        old_state = self.game_state
        
        # 1. 计算血量差值 (Damage Reward)
        # 场上总血量减少 = 造成的伤害 + 敌人死亡 + 敌人漏掉
        current_total_hp = self._get_total_health(new_game_state["enemies"])
        hp_reduction = self.prev_total_health - current_total_hp
        
        # 如果新怪生成导致总血量增加(hp_reduction < 0)，我们不惩罚，记为0
        # 只有血量减少才算有效输出
        if hp_reduction > 0:
            # 缩放系数: 假设一波怪总血量 1000，打完给 10 分
            reward += hp_reduction * 0.01 

        # 2. 击杀奖励 (Kill Reward)
        # 鼓励完成最后一击
        kill_count = max(0, len(old_state["enemies"]) - len(new_game_state["enemies"]))
        if kill_count > 0:
            reward += kill_count * 1.0

        # 3. 漏怪惩罚 (Leak Penalty)
        # 必须重罚！
        # 注意: 当怪漏掉时，它的血量会从 current_total_hp 中消失，导致 hp_reduction 变大，
        # 从而错误的给了一个正奖励。所以这里的负惩罚必须足够大，覆盖掉那个误判的正奖励。
        lives_lost = old_state["lives"] - new_game_state["lives"]
        if lives_lost > 0:
            # 假设漏掉一个满血 Tank (150血)，Damage Reward 会误加 1.5分
            # 所以惩罚至少要是 -20，确保 Agent 知道这是亏的
            reward -= lives_lost * 20.0

        # 4. 辅助奖励: 波次推进
        if new_game_state["waveNumber"] > old_state["waveNumber"]:
            reward += 5.0

        # 5. 游戏结束惩罚
        if new_game_state["gameOver"]:
            reward -= 100.0
            
        # 6. (可选) 微小的闲置/囤钱惩罚
        # 为了防止 Agent 在开局完全不造塔导致直接漏怪，
        # 如果钱很多但场上塔很少，可以给一点微小压力，但伤害奖励通常足够驱动它造塔。
        
        return float(reward)

    def action_masks(self) -> np.ndarray:
        # [修改] 加入基于规则的 Masking，防止后期乱花钱
        
        action_mask = np.ones(len(self.action_types), dtype=bool)
        
        # 1. 钱不够基础 Mask
        min_cost = min(t["cost"] for t in self.tower_types)
        if self.game_state["money"] < min_cost:
            for idx, act in enumerate(self.action_types):
                if act["type"] == "BUILD_TOWER":
                    action_mask[idx] = False
                    break

        tower_mask = np.ones(len(self.tower_types), dtype=bool)
        
        # [新规则]：如果钱很多 (>200) 且到了后期 (>Wave 10)，强制禁止买 Archer
        # 强迫 Agent 只能买 Cannon/Sniper
        force_high_tech = (self.game_state["money"] > 200 and self.game_state["waveNumber"] > 10)
        
        for idx, t in enumerate(self.tower_types):
            # 基础规则: 钱不够或没解锁
            if (self.game_state["money"] < t["cost"] or 
                self.game_state["waveNumber"] < t.get("unlock_wave", 0)):
                tower_mask[idx] = False
            
            # 进阶规则: 强制消费升级
            if force_high_tech and t["type"] == "archer":
                tower_mask[idx] = False # 禁止买 Archer

        x_mask = np.ones(self.cols, dtype=bool)
        y_mask = np.ones(self.rows, dtype=bool)

        return np.concatenate([action_mask, tower_mask, x_mask, y_mask])

    def render(self):
        target_w = self.map_width_px
        target_h = self.map_height_px
        black_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if self.render_mode == "rgb_array":
            try:
                res = requests.get(url + "render")
                if res.status_code == 200:
                    img = Image.open(io.BytesIO(res.content))
                    if img.size != (target_w, target_h):
                        img = img.resize((target_w, target_h))
                    return np.array(img)
            except:
                pass
        return black_frame
    
    def close(self):
        pass

    def _get_empty_state(self):
        return {
            "gameTime": 0, "waveNumber": 0, "money": 0, "lives": 0, "gameOver": False,
            "towers": [], "enemies": []
        }