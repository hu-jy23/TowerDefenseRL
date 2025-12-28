import argparse
import json
import os
import numpy as np
import gymnasium as gym
import gymnasium_env.envs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from gymnasium_env.wrappers.random_map_wrapper import RandomMapWrapper
from gymnasium_env.wrappers.wrap import wrap_env

# 复用 train.py 中的配置加载逻辑，或者直接硬编码
# 这里为了独立运行，我们重新定义一个构建环境的函数
def make_eval_env(map_file_path, port=3001):
    env_name = "gymnasium_env/TowerDefenseWorld-v0"
    
    # 1. 创建基础环境
    # 注意：评估时通常不需要 survival_reward，因为我们只关心客观的 wave 数
    # 但为了保持 observation 的一致性，建议配置保持一致
    env = gym.make(env_name, port=port)
    
    # 2. 加载测试地图集
    if map_file_path:
        with open(map_file_path, "r") as f:
            map_data = json.load(f)
        # 使用 RandomMapWrapper 加载测试集
        env = RandomMapWrapper(env, map_list=map_data)
    
    # 3. 必要的 Wrapper (必须与训练时结构一致)
    # 评估时不需要录像 (episode_gap=0)
    env = wrap_env(env, episode_recording_gap=0, prefix=None)
    
    # 4. 向量化环境 (SB3 要求)
    env = DummyVecEnv([lambda: env])

    # 5. 归一化 (关键点)
    # 如果训练时用了 VecNormalize，评估时也必须用
    # 注意：理想情况下应该加载训练时保存的 vecnormalize.pkl，
    # 但如果没有保存，我们可以创建一个新的，但要关闭 training (不再更新均值方差)
    # norm_reward=False 表示我们想看原始奖励/波次，而不是归一化后的数值
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    return env

def evaluate(model_path, map_path, port, n_episodes=20):
    print(f"Loading model from: {model_path}")
    print(f"Using test maps from: {map_path}")
    
    # 1. 创建环境
    env = make_eval_env(map_path, port=port)
    
    # 2. 加载模型
    model = MaskablePPO.load(model_path, env=env)
    
    # 3. 开始评估循环
    episode_waves = []
    episode_rewards = []
    
    print(f"Starting evaluation over {n_episodes} episodes...")
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        final_wave = 0
        
        while not done:
            # deterministic=True 表示关闭随机探索，使用模型认为的最优动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 如果 episode 结束
            # deterministic=True 表示关闭随机探索，使用模型认为的最优动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 如果 episode 结束
            if done:
                # 获取波次信息
                if "wave_number" in info:
                    final_wave = info["wave_number"]
                
                episode_waves.append(final_wave)
                episode_rewards.append(total_reward)
                
                print(f"Episode {i+1}/{n_episodes}: Wave {final_wave}")
                break

    # 4. 统计结果
    mean_wave = float(np.mean(episode_waves))
    std_wave = float(np.std(episode_waves))
    min_wave = int(np.min(episode_waves))
    max_wave = int(np.max(episode_waves))
    
    results = {
        "model_path": model_path,
        "test_map_file": map_path,
        "n_episodes": n_episodes,
        "mean_wave": mean_wave,
        "std_wave": std_wave,
        "min_wave": min_wave,
        "max_wave": max_wave,
        "detailed_waves": [int(w) for w in episode_waves] # 转换为 int 列表以便 JSON 序列化
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .zip model file")
    parser.add_argument("--maps", type=str, default="test-maps.json", help="Path to test maps json")
    parser.add_argument("--port", type=int, default=3000, help="Game server port")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    
    # 执行评估
    stats = evaluate(args.model_path, args.maps, args.port, args.episodes)
    
    # 5. 保存结果到模型所在目录
    model_dir = os.path.dirname(args.model_path)
    output_file = os.path.join(model_dir, "eval_results.json")
    
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print("\n" + "="*30)
    print(f"Evaluation Complete!")
    print(f"Mean Wave: {stats['mean_wave']:.2f} (Max: {stats['max_wave']})")
    print(f"Results saved to: {output_file}")
    print("="*30)