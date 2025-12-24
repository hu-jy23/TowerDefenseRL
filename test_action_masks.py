"""测试 action_masks 功能是否正确工作"""
import gymnasium as gym
import gymnasium_env.envs
from gymnasium_env.wrappers.hierarchical_wrapper import HierarchicalActionWrapper

# 创建环境
env = gym.make("gymnasium_env/TowerDefenseWorld-v0", port=3000)
env = HierarchicalActionWrapper(env)

# 重置环境
obs, info = env.reset()
print(f"初始观察: shape={obs.shape}")
print(f"动作空间: {env.action_space}")
print(f"动作空间大小: {env.action_space.n}")

# 检查初始状态的 action masks
masks = env.action_masks()
print(f"\n第1波的 action masks: {masks}")
print(f"可用动作索引: {[i for i, m in enumerate(masks) if m]}")

# 打印每个塔的信息
print("\n塔信息:")
for idx, tower in env.idx_to_tower.items():
    mask_status = "✓ 可用" if masks[idx] else "✗ 不可用"
    print(f"  动作{idx}: {tower['type']:8s} | 解锁波数: {tower['unlock_wave']} | 成本: {tower['cost']:4d} | {mask_status}")

# 模拟几步，观察 mask 变化
print("\n模拟游戏进行...")
for step in range(5):
    action = 0  # 挂机
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
    
    wave = env.unwrapped.game_state["waveNumber"]
    money = env.unwrapped.game_state["money"]
    masks = env.action_masks()
    available_actions = [i for i, m in enumerate(masks) if m]
    
    print(f"步数{step+1}: 波数={wave}, 金钱={money}, 可用动作={available_actions}")

env.close()
print("\n测试完成！")
