import gymnasium as gym
import torch
from torch import nn
import numpy as np
import time

# 1. 必须定义一模一样的网络结构，否则无法加载权重！
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

# 2. 加载刚刚训练好的“大脑”
model.load_state_dict(torch.load("bc_policy.pth"))
model.eval() # 极其重要！告诉网络进入“评估/测试”模式

# 3. 创建带画面的环境
env = gym.make("Reacher-v5", render_mode="human")

NUM_TEST_EPISODES = 10 # 测试 10 个回合
success_count = 0

print("开始测试模型自主执行能力...")

for episode in range(NUM_TEST_EPISODES):
    observation, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # --- 核心变化：自己预测动作 ---
        # 把 numpy 的 observation 转换成 PyTorch 的 Tensor
        state_tensor = torch.tensor(observation, dtype=torch.float32)
        
        # 使用 torch.no_grad() 表示我们只是在做推断，不需要计算梯度
        with torch.no_grad():
            action_tensor = model(state_tensor)
            
        # 把输出转换回 numpy 数组
        action = action_tensor.numpy()
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 记录累计奖励（在 Reacher 任务里，越靠近目标，reward 越大，越远离越小）
        total_reward += reward
        done = terminated or truncated
        
        # 稍微慢一点，让你看清它的动作
        time.sleep(0.03) 
        
    print(f"回合 {episode+1}/{NUM_TEST_EPISODES}, 累计奖励 (Reward): {total_reward:.2f}")

env.close()