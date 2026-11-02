import gymnasium as gym
import numpy as np

# 1. 创建环境（注意：这次我们不加 render_mode="human"，为了极速采集！）
env = gym.make("Reacher-v5")

# 我们要采集的轨迹数量
NUM_EPISODES = 50

# 存储所有轨迹的列表
all_states = []
all_actions = []

print(f"开始采集 {NUM_EPISODES} 条专家轨迹...")

for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    done = False
    step_count = 0
    
    while not done:
        # --- 专家策略逻辑开始 ---
        # Reacher-v4/v5 的 observation 最后 4 维是：
        # to_target_x, to_target_y (目标相对于指尖的坐标), fingertip_x, fingertip_y
        to_target_x = observation[-4]
        to_target_y = observation[-3]
        
        # 专家动作：比例控制 (Proportional Control)
        # 让关节朝目标方向发力。增益 Kp 控制力量大小。
        Kp = 20.0 
        expert_action = np.array([to_target_x * Kp, to_target_y * Kp])
        
        # 限制动作范围，不能超过环境允许的最大力矩
        expert_action = np.clip(expert_action, env.action_space.low, env.action_space.high)
        # --- 专家策略逻辑结束 ---
        
        # 记录当前状态和专家动作
        all_states.append(observation)
        all_actions.append(expert_action)
        
        # 让环境执行专家的动作
        observation, reward, terminated, truncated, info = env.step(expert_action)
        done = terminated or truncated
        step_count += 1
        
    if (episode + 1) % 10 == 0:
        print(f"已采集 {episode + 1}/{NUM_EPISODES} 条轨迹")

# 将列表转换为 NumPy 数组 (N, state_dim) 和 (N, action_dim)
states_array = np.array(all_states)
actions_array = np.array(all_actions)

print("\n=== 数据采集完成 ===")
print(f"状态数据形状 (States): {states_array.shape}")
print(f"动作数据形状 (Actions): {actions_array.shape}")

# 保存到文件，供下一步训练使用
np.savez("expert_data.npz", states=states_array, actions=actions_array)
print("数据已成功保存为 expert_data.npz")

env.close()