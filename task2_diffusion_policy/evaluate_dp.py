import gymnasium as gym
import torch
from torch import nn
import numpy as np
import time
import math
import os

# === 1. 必须把刚才训练的模型结构原封不动搬过来 ===
T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, 0].unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = nn.SiLU()(h)
        h = self.conv1(h)
        time_emb = self.time_mlp(t_emb)[:, :, None]
        h = h + time_emb
        h = self.norm2(h)
        h = nn.SiLU()(h)
        h = self.conv2(h)
        return h + self.res_conv(x)

class TrueConditionalUNet(nn.Module):
    def __init__(self, in_channels=2, state_dim=10, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.down1 = ResidualBlock(32, 64, time_emb_dim)
        self.down2 = ResidualBlock(64, 128, time_emb_dim)
        self.up1 = ResidualBlock(128 + 64, 64, time_emb_dim)
        self.up2 = ResidualBlock(64 + 32, 32, time_emb_dim)
        self.final_conv = nn.Conv1d(32, in_channels, 1)

    def forward(self, x, t, state):
        t_emb = self.time_mlp(t)
        s_emb = self.state_mlp(state)
        cond_emb = t_emb + s_emb
        x1 = self.init_conv(x)
        x2 = self.down1(x1, cond_emb)
        x3 = self.down2(x2, cond_emb)
        h = torch.cat([x3, x2], dim=1)
        h = self.up1(h, cond_emb)
        h = torch.cat([h, x1], dim=1)
        h = self.up2(h, cond_emb)
        return self.final_conv(h)

# === 2. 核心魔法：去噪采样过程 ===
@torch.no_grad()
def sample_dp_action(model, state, device):
    # 1. 初始化为纯高斯噪声 (1, 2, 8) -> 1个Batch, 2维动作, 8步序列
    x = torch.randn((1, 2, 8)).to(device)
    state = state.to(device)
    
    # 2. 循环 T 步，一步步去噪（从 T-1 到 0）
    for t in reversed(range(T)):
        t_tensor = torch.tensor([[t]]).to(device)
        
        # 预测噪声
        predicted_noise = model(x, t_tensor, state)
        
        # DDPM 逆向公式参数
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]
        
        # 根据公式计算均值
        if t > 0:
            noise = torch.randn_like(x).to(device)
            sigma = beta.sqrt()
        else:
            noise = torch.zeros_like(x).to(device)
            sigma = 0
            
        mean = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_bar).sqrt()) * predicted_noise)
        
        # 更新动作序列
        x = mean + sigma * noise
        
    # 返回去噪干净的 8 步动作序列
    return x.squeeze(0).cpu().numpy() # 形状 (2, 8)

# === 3. 在环境中执行 ===
device = torch.device("cpu") # 如果推理太慢，可以换回 CPU 试试，或者用 GPU
model = TrueConditionalUNet().to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(os.path.join(current_dir, "dp_policy.pth"), map_location=device))
# 在 evaluate_dp.py 的 model.load_state_dict 之后加上：
# （注意填入你刚才在 prepare_data_dp.py 终端里看到的真实 mean 和 std）
ACTION_MEAN = 0.0  # 替换为你终端里打印的真实值
ACTION_STD = 1.0   # 替换为你终端里打印的真实值
model.eval()

env = gym.make("Reacher-v5", render_mode="human")
NUM_TEST_EPISODES = 5
success_count = 0

print("开始测试 Diffusion Policy 自主执行...")
for episode in range(NUM_TEST_EPISODES):
    observation, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        # 1. 获取状态
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        
        # 2. 模型生成动作序列（此时输出的是归一化后的值）
        action_seq = sample_dp_action(model, state_tensor, device) 
        
        # 3. 只取第一步动作
        action = action_seq[:, 0] 
        
        # === 在这里加入反归一化 ====================
        # 4. 将归一化的值还原为真实的物理力矩
        action = (action * ACTION_STD) + ACTION_MEAN
        # ==========================================
        
        # 5. 限制动作范围（防止模型偶尔发疯输出过大）
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # 6. 发送给环境
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1
        
        time.sleep(0.02)
        time.sleep(0.02) # 稍微放慢点让你看清

    print(f"回合 {episode+1}/{NUM_TEST_EPISODES}, 累计奖励: {total_reward:.2f}, 总步数: {step_count}")

env.close()