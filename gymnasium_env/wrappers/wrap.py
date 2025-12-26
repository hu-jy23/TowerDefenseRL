from gymnasium.wrappers import RecordVideo, Autoreset
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np

# === 跳帧 Wrapper ===
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        skip 表示 AI 每做 1 次动作，游戏自动走 skip 帧。
        在正常玩家游戏中，1 帧为 0.1 秒。
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        
        for i in range(self._skip):
            # 第 1 帧执行挑选的动作
            if i == 0:
                step_action = action
            else:
                # 其余几帧挂机
                if isinstance(action, np.ndarray):
                    step_action = np.zeros_like(action)
                else:
                    step_action = 0
            
            # 无论该帧是否挂机，底层仍每帧与服务器交互一次。PPO 会利用这些帧的数据进行参数更新。
            obs, reward, term, trunc, info = self.env.step(step_action)
            total_reward += reward
            terminated = term
            truncated = trunc
            
            if terminated or truncated:
                break
                
        return obs, total_reward, terminated, truncated, info

def wrap_env(env, episode_recording_gap, prefix):
    # 核心：加上 SkipFrame (skip=120 表示每 1 秒 决策一次，大幅降低乱花钱的概率)
    env = SkipFrame(env, skip=120)
    
    # 监控器 (Monitor): 记录每一局的 Reward (总奖励) 和 Episode Length (步数) 到 CSV 文件。
    # CSV 文件保存在 ./models/{prefix}/monitor.csv。TensorBoard 就是读取这个文件来画 Reward 曲线的。
    env = Monitor(env, f"./models/{prefix}/monitor.csv")

    # 录像机 (RecordVideo): 自动把 Agent 玩游戏的过程录制成 MP4 视频。
    # 每隔 episode_recording_gap 局，就录一次像。
    # 视频保存在 ./models/{prefix}/videos/ 目录下。
    env = RecordVideo(env, video_folder=f"./models/{prefix}/videos/", 
                      name_prefix="training", 
                      episode_trigger=lambda e: e % episode_recording_gap == 0)

    # 自动重置 (Autoreset): 当游戏结束时，自动调用 reset()。
    env = Autoreset(env)
    
    # 传入的 env 经过 SkipFrame -> Monitor -> RecordVideo -> Autoreset 层层包装后，输出一个 Wrapper 对象。
    return env