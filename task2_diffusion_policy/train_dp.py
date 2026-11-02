import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math

# === 1. 标准 DDPM 参数 ===
T = 200  # 增加去噪总步数
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# === 2. 时间步正弦位置编码（让网络更好地理解时间 t） ===
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

# === 3. 真正的残差块 (Residual Block) ===
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
        # 把时间信息注入进去
        time_emb = self.time_mlp(t_emb)[:, :, None]
        h = h + time_emb
        h = self.norm2(h)
        h = nn.SiLU()(h)
        h = self.conv2(h)
        return h + self.res_conv(x)

# === 4. 真正的条件 1D U-Net ===
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
        cond_emb = t_emb + s_emb  # 融合条件
        
        x1 = self.init_conv(x)
        x2 = self.down1(x1, cond_emb)
        x3 = self.down2(x2, cond_emb)
        
        # 跳跃连接
        h = torch.cat([x3, x2], dim=1)
        h = self.up1(h, cond_emb)
        h = torch.cat([h, x1], dim=1)
        h = self.up2(h, cond_emb)
        return self.final_conv(h)

# === 5. 加载数据 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "dp_data.npz")
data = np.load(data_path)
states = torch.tensor(data["states"], dtype=torch.float32)
action_seqs = torch.tensor(data["action_seqs"], dtype=torch.float32).transpose(1, 2)

dataset = TensorDataset(states, action_seqs)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 6. 训练 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

model = TrueConditionalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("开始训练终极版 Diffusion Policy...")
for epoch in range(500):
    epoch_loss = 0.0
    for batch_states, batch_actions in dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_size = batch_actions.shape[0]
        
        # 采样 t (0 到 T-1)
        t = torch.randint(0, T, (batch_size, 1)).to(device)
        noise = torch.randn_like(batch_actions).to(device)
        
        # 加噪
        alpha_bar = alphas_cumprod[t.cpu()].view(batch_size, 1, 1).to(device)
        noisy_actions = alpha_bar.sqrt() * batch_actions + (1 - alpha_bar).sqrt() * noise
        
        # 预测
        predicted_noise = model(noisy_actions, t, batch_states)
        
        # 计算 Loss
        loss = nn.MSELoss()(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:03d}/500, Loss: {epoch_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "dp_policy.pth")
print("神功已成！模型已保存为 dp_policy.pth")