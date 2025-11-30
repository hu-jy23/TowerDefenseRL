import gymnasium as gym
import gymnasium_env.envs  # 注册自定义环境
import time

# 这里的 ID 对应 tower_defense_world.py 中注册的 ID
ENV_ID = "gymnasium_env/TowerDefenseWorld-v0"

try:
    print(f"1. 尝试创建环境: {ENV_ID}")
    env = gym.make(ENV_ID)
    
    print("2. 尝试重置环境 (发送 /reset 请求)...")
    obs, info = env.reset()
    print("   ✅ 环境重置成功！接收到初始观察数据。")
    
    print("3. 尝试执行随机动作 (发送 /step 请求)...")
    # 随机采样一个动作
    action = env.action_space.sample()
    # 执行一步
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✅ 动作执行成功！获得奖励: {reward}")
    
    print("\n🎉 恭喜！RL 环境与游戏服务器连接正常！")
    env.close()

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    print("请检查：")
    print("1. 游戏服务器是否在运行 (npm run start:api)？")
    print("2. 端口 3000 是否被占用或未开放？")