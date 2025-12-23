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
        
        # === [升级版] 模拟专家课程 (Simulated Expert Curriculum) ===
        rand_val = np.random.random()
        reset_payload = {}

        # 坐标配置 (与前端 game.ts 保持严格一致)
        # 注意: 如果 Python 端和 TypeScript 端地图坐标系一致，直接用像素坐标
        scenario_1_towers = [
            {"type": "cannon", "x": 275, "y": 425}, # 核心拐角
            {"type": "cannon", "x": 325, "y": 425}, # 核心拐角
            {"type": "archer", "x": 725, "y": 175}, # 终点防守
            {"type": "archer", "x": 725, "y": 225}  # 终点防守
        ]
        
        scenario_2_towers = [
            {"type": "archer", "x": 725, "y": 175},
            {"type": "archer", "x": 725, "y": 225}
        ]

        if rand_val < 0.4: 
            # [模式 A: 专家残局 - 学习造 Sniper] (40% 概率)
            # 场景: Wave 9, 62块, 已有火力基础
            # 目标: 配合 Mask, Agent 只能买 Sniper, 体验后期高回报
            reset_payload = {
                "start_wave": 9,
                "start_money": 62,
                "prebuilt_towers": scenario_1_towers
            }
            # print(f"[Curriculum] Scenario 1: Expert Late Game")
            
        elif rand_val < 0.7:
            # [模式 B: 过渡残局 - 学习造 Cannon] (30% 概率)
            # 场景: Wave 6, 40块, 只有基础弓
            # 目标: 配合 Mask, Agent 必须买 Cannon 才能守住怪群
            reset_payload = {
                "start_wave": 6,
                "start_money": 40,
                "prebuilt_towers": scenario_2_towers
            }
            # print(f"[Curriculum] Scenario 2: Mid Game Transition")
            
        else:
            # [模式 C: 正常开局 - 综合大考] (30% 概率)
            # 场景: Wave 0, 40块, 空地
            # 目标: 检验是否学会了前期的克制和后期的爆发
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
        # [升级版] 消费升级锁 (Consumption Lock)
        
        action_mask = np.ones(len(self.action_types), dtype=bool)
        
        # 1. 基础规则: 钱不够 Mask
        min_cost = min(t["cost"] for t in self.tower_types)
        if self.game_state["money"] < min_cost:
            for idx, act in enumerate(self.action_types):
                if act["type"] == "BUILD_TOWER":
                    action_mask[idx] = False
                    break

        tower_mask = np.ones(len(self.tower_types), dtype=bool)
        
        # 获取当前状态
        money = self.game_state["money"]
        wave = self.game_state["waveNumber"]
        towers_count = len(self.game_state["towers"])
        
        # 获取塔的元数据
        sniper_cost = next((t["cost"] for t in self.tower_types if t["type"] == "sniper"), 50)
        cannon_cost = next((t["cost"] for t in self.tower_types if t["type"] == "cannon"), 30)
        
        for idx, t in enumerate(self.tower_types):
            # 2. 基础规则: 钱不够或没解锁 -> False
            if (money < t["cost"] or wave < t.get("unlock_wave", 0)):
                tower_mask[idx] = False
                continue
            
            # === [核心] 消费升级逻辑 ===
            
            # 规则 A: 富人强制消费 (针对 Scenario 1)
            # 只要买得起 Sniper (45块), 严禁买 Archer (20块)
            # 逻辑: "有钱必须花在刀刃上"
            if money >= sniper_cost and t["type"] == "archer":
                tower_mask[idx] = False
            
            # 规则 B: 中产阶级陷阱 (针对 Scenario 2)
            # 如果是中期 (Wave 4-8), 且已经有两个 Archer 了
            # 禁止继续买 Archer, 逼迫买 Cannon 或存钱买 Sniper
            if 4 <= wave <= 8 and towers_count >= 2 and t["type"] == "archer":
                tower_mask[idx] = False

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