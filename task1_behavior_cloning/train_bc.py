import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 加载刚才采集的数据
print("加载数据...")
data = np.load("expert_data.npz")
states = data["states"]    # 形状 (2500, 10)
actions = data["actions"]  # 形状 (2500, 2)

# 2. 把 NumPy 数组转换成 PyTorch 的 Tensor（张量）
# 神经网络只认识 Tensor，且通常是 float32 类型
states_tensor = torch.tensor(states, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.float32)

# 3. 构建 DataLoader，实现小批量（Mini-batch）训练
dataset = TensorDataset(states_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. 定义我们的 MLP 网络结构
# 输入维度是 10，输出维度是 2
model = nn.Sequential(
    nn.Linear(10, 128),   # 输入层到隐藏层1
    nn.ReLU(),            # 激活函数
    nn.Linear(128, 128),  # 隐藏层1到隐藏层2
    nn.ReLU(),
    nn.Linear(128, 2)     # 隐藏层2到输出层
)

# 5. 定义损失函数和优化器
loss_fn = nn.MSELoss() # 均方误差，衡量预测动作和专家动作的差距
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. 开始训练循环
epochs = 100
print("开始训练模型...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_states, batch_actions in dataloader:
        # --- 前向传播 ---
        pred_actions = model(batch_states) 
        loss = loss_fn(pred_actions, batch_actions)
        
        # --- 反向传播 ---
        optimizer.zero_grad() # 清空之前的梯度
        loss.backward()       # 计算梯度
        optimizer.step()      # 更新权重
        
        epoch_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1:03d}/{epochs}, Loss: {avg_loss:.6f}")

# 7. 保存训练好的模型权重
torch.save(model.state_dict(), "bc_policy.pth")
print("模型已保存为 bc_policy.pth")