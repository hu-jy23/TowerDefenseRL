"""测试不同波数下的 action_masks"""
import gymnasium as gym
import gymnasium_env.envs
from gymnasium_env.wrappers.hierarchical_wrapper import HierarchicalActionWrapper

# 创建环境
env = gym.make("gymnasium_env/TowerDefenseWorld-v0", port=3000)
env = HierarchicalActionWrapper(env)

# 重置环境
obs, info = env.reset()

print("=" * 60)
print("测试：随着波数增加，action masks 的变化")
print("=" * 60)

# 打印塔的解锁信息
print("\n塔的解锁波数:")
for idx, tower in env.idx_to_tower.items():
    print(f"  动作{idx}: {tower['type']:8s} | 解锁波数: {tower['unlock_wave']}")

# 模拟游戏到不同波数，观察mask变化
for target_wave in [1, 4, 7, 10]:
    # 重置环境
    obs, info = env.reset()
    
    # 挂机直到目标波数
    while env.unwrapped.game_state["waveNumber"] < target_wave:
        obs, reward, terminated, truncated, info = env.step(0)  # 挂机
        if terminated or truncated:
            print(f"游戏在波数 {env.unwrapped.game_state['waveNumber']} 结束")
            break
    
    if terminated or truncated:
        continue
        
    wave = env.unwrapped.game_state["waveNumber"]
    money = env.unwrapped.game_state["money"]
    masks = env.action_masks()
    available_actions = [i for i, m in enumerate(masks) if m]
    
    print(f"\n波数 {wave} | 金钱 {money}")
    print(f"  action masks: {masks}")
    print(f"  可用动作: {available_actions}")
    for idx, tower in env.idx_to_tower.items():
        if masks[idx]:
            print(f"    ✓ 动作{idx}: {tower['type']}")

env.close()
print("\n" + "=" * 60)
print("测试完成！action_masks 成功过滤未解锁的塔")
print("=" * 60)
