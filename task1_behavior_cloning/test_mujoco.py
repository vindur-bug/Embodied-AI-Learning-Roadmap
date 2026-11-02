import gymnasium as gym

# 1. 创建环境，render_mode="human" 表示我们要看到画面
env = gym.make("Reacher-v5", render_mode="human")

# 2. 重置环境，获取初始状态
observation, info = env.reset()

print("=== 环境接口信息 ===")
print(f"状态 (Observation) 的形状: {observation.shape}")
print(f"状态的具体数值:\n {observation}")
print(f"动作空间 (Action Space): {env.action_space}")
print("====================\n")

# 3. 让机器人在仿真里随机乱动 200 步
for step in range(200):
    # 随机采样一个动作（在动作空间允许的范围内）
    action = env.action_space.sample()
    
    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 如果任务结束（比如超时），就重置环境
    if terminated or truncated:
        observation, info = env.reset()

env.close()