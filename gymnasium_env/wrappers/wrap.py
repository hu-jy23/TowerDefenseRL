from gymnasium.wrappers import RecordVideo, Autoreset
from stable_baselines3.common.monitor import Monitor

def wrap_env(env, episode_recording_gap, prefix):
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
    
    # 传入的 env 经过 Monitor -> RecordVideo -> Autoreset 层层包装后，输出一个 Wrapper 对象。
    return env