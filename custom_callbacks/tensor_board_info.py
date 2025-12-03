from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardInfoCallback(BaseCallback):
    """
    继承自回调函数类 BaseCallback。
    
    self.logger: 用于记录自定义指标到 TensorBoard。
    self.locals (dict): 包含调用回调函数那一刻，算法内部所有的变量。
    SB3 会自动把环境放到一个向量中 (即使只有一个环境)，并产生一些键值对放到 self.locals 中:
        - self.locals["dones"] (list): 环境向量是否结束的布尔数组。比如 self.locals["dones"][0] 表示第一个环境是否结束。
        - self.locals["infos"] (list): 环境向量返回的信息字典数组，是我们在 tower_defense_world.py 里 __get_info() 返回的字典。
        - self.locals["rewards"] (list): 环境向量返回的奖励数组。
        - self.locals["actions"] (list): 刚刚一步中算法选择的动作数组。
        - self.locals["obs"] (list): 前一步观测的数组。
        - self.locals["new_obs"] (list): 执行动作后当前的观测的数组。
        - self.locals["env"]: 环境向量对象。
    """
    def __init__(self, verbose=0):
        """
        初始化回调函数。
        
        Args:
            episode_wave_numbers (list): 用于存储每局游戏结束时的波数。
            episode_tower_counts (dict): 用于存储每局游戏结束时不同类型的塔的数量。
                                    比如 {"archer": [1, 2, 1], "cannon": [0, 4, 2], "sniper": [0, 3, 2]}。
        """
        super().__init__(verbose)
        self.episode_wave_numbers = []
        self.episode_tower_counts = {}

    def _on_step(self) -> bool:
        """
        每走一步调用一次该函数。
        如果当前一局游戏结束（done），则从 info 字典中提取波数和塔的数量，并保存到相应的列表中。
        """
        # 当一局游戏结束时
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            if "wave_number" in info:
                self.episode_wave_numbers.append(info["wave_number"])
            if "tower_counts" in info:
                for tower_type, count in info["tower_counts"].items():
                    if tower_type not in self.episode_tower_counts:
                        self.episode_tower_counts[tower_type] = []
                    self.episode_tower_counts[tower_type].append(count)
        return True
    
    def _on_rollout_end(self) -> None: # default rollout is 2048 steps, so ~205 seconds in game time
        """
        每次 rollout 结束时调用该函数，记录这些列表的平均值。
        一次 rollout 指的是收集一定数量的环境交互数据，PPO 默认一次 rollout 是 2048 步。
        
        SB3 的 TensorBoard 会自动记录以下核心指标到 TensorBoard 的二进制日志文件 ./logs/PPO_2/events.out.tfevents.* 中:
            - rollout/ep_len_mean: 每个 episode 的平均步数
            - rollout/ep_rew_mean: 每个 episode 的平均奖励
            - train/learning_rate: 当前学习率
            - train/loss: 策略/价值函数的损失
            - time/fps: 每秒处理的环境步数
            - time/total_timesteps: 已训练的总步数
        我们还自定义了一些额外的指标，并把它们记录到 TensorBoard 中:
            - rollout/custom/ep_wave_number_mean: 每个 episode 的平均波数
            - rollout/custom/ep_{tower_type}_count_mean: 每个 episode 中每种塔的平均数量
        """
        # log the mean wave number if we have data
        if len(self.episode_wave_numbers) > 0:
            mean_wave_number = np.mean(self.episode_wave_numbers)
            self.logger.record("rollout/custom/ep_wave_number_mean", mean_wave_number)
            self.episode_wave_numbers.clear()
        # log the mean tower counts for each type
        for tower_type, counts in self.episode_tower_counts.items():
            if len(counts) > 0:
                mean_count = np.mean(counts)
                self.logger.record(f"rollout/custom/ep_{tower_type}_count_mean", mean_count)
                counts.clear()