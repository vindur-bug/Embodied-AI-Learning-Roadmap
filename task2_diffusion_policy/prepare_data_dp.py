import numpy as np

# 加载 Task 1 采集的原始数据
import os
# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼凑出 Task 1 数据的绝对路径
data_path = os.path.join(current_dir, "../task1_behavior_cloning/expert_data.npz")

data = np.load(data_path)
states = data["states"]    # (2500, 10)
actions = data["actions"]  # (2500, 2)

# 定义超参数
ACTION_HORIZON = 8  # 预测未来 8 步的动作
STATE_DIM = 10
ACTION_DIM = 2

# 我们只提取每个步的当前状态，以及它未来 8 步的动作
dp_states = []
dp_action_seqs = []

# 因为采集的数据是 50 条轨迹（每条 50 步），我们要防止跨越轨迹边界去取未来动作
EPISODE_LEN = 50
NUM_EPISODES = 50

for ep in range(NUM_EPISODES):
    start_idx = ep * EPISODE_LEN
    # 遍历该轨迹的每一步（最后几步没法看未来 8 步，所以跳过）
    for step in range(EPISODE_LEN - ACTION_HORIZON + 1):
        current_idx = start_idx + step
        
        # 当前状态 s_t
        s_t = states[current_idx]
        
        # 未来 8 步的动作序列 a_{t:t+8}
        a_seq = actions[current_idx : current_idx + ACTION_HORIZON]
        
        dp_states.append(s_t)
        dp_action_seqs.append(a_seq)

dp_states = np.array(dp_states)           # 形状：(2100, 10)
dp_action_seqs = np.array(dp_action_seqs) # 形状：(2100, 8, 2)

print(f"改造后的状态形状: {dp_states.shape}")
print(f"改造后的动作序列形状: {dp_action_seqs.shape}")

# 保存给扩散模型用
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "dp_data.npz")
np.savez(save_path, states=dp_states, action_seqs=dp_action_seqs)
print(f"数据已保存在: {save_path}")
print("数据已保存为 dp_data.npz")
# 在 prepare_data_dp.py 的最后，保存之前加上这几行：
action_mean = np.mean(dp_action_seqs)
action_std = np.std(dp_action_seqs) + 1e-6 # 防止除0
dp_action_seqs_normalized = (dp_action_seqs - action_mean) / action_std

# 保存归一化后的动作
np.savez("dp_data.npz", states=dp_states, action_seqs=dp_action_seqs_normalized)
print(f"动作归一化参数: mean={action_mean:.4f}, std={action_std:.4f}")