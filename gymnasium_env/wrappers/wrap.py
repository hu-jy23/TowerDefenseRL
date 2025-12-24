import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo, Autoreset
from stable_baselines3.common.monitor import Monitor

# === 新增：跳帧 Wrapper ===
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=60):
        """
        skip=4 表示：AI 每做 1 次动作，环境自动走 4 步。
        这能让时间流速变快，让 AI 更容易看到“未来的大奖”。
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        
        for i in range(self._skip):
            # 第 1 帧执行动作，后 3 帧挂机 (动作 0)
            if i == 0:
                step_action = action
            else:
                # 兼容 numpy array 类型的动作
                if isinstance(action, np.ndarray):
                    step_action = np.zeros_like(action)
                else:
                    step_action = 0
            
            obs, reward, term, trunc, info = self.env.step(step_action)
            total_reward += reward
            terminated = term
            truncated = trunc
            
            if terminated or truncated:
                break
                
        return obs, total_reward, terminated, truncated, info

# === 修改 wrap_env 函数 ===
def wrap_env(env, episode_recording_gap, prefix):
    # 1. 核心：加上 SkipFrame (skip=30 表示每 0.5秒 决策一次，大幅降低乱花钱的概率)
    env = SkipFrame(env, skip=120)
    
    env = Monitor(env, f"./models/{prefix}/monitor.csv")
    env = RecordVideo(env, video_folder=f"./models/{prefix}/videos/", name_prefix="training", episode_trigger=lambda e: e % episode_recording_gap == 0)
    env = Autoreset(env)
    return env