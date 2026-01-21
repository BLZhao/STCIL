import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------
# Lorenz System Generator
# ---------------------------
def generate_lorenz_data(
    sigma=10.0, rho=28.0, beta=8/3,
    dt=0.01, num_steps=10000,
    initial_state=None
):
    """使用4阶Runge-Kutta方法生成Lorenz系统数据

    Args:
        sigma, rho, beta: Lorenz系统参数
        dt: 时间步长
        num_steps: 生成步数
        initial_state: 初始状态 [x, y, z]，默认为 [1.0, 1.0, 1.0]

    Returns:
        (num_steps, 3) numpy数组，每行为 (x, y, z)
    """
    def lorenz_deriv(state, t, sigma, rho, beta):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return np.array([dxdt, dydt, dzdt])

    def rk4_step(state, t, dt, sigma, rho, beta):
        k1 = lorenz_deriv(state, t, sigma, rho, beta)
        k2 = lorenz_deriv(state + k1 * dt / 2, t + dt / 2, sigma, rho, beta)
        k3 = lorenz_deriv(state + k2 * dt / 2, t + dt / 2, sigma, rho, beta)
        k4 = lorenz_deriv(state + k3 * dt, t + dt, sigma, rho, beta)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    if initial_state is None:
        state = np.array([1.0, 1.0, 1.0])
    else:
        state = np.array(initial_state)

    data = np.zeros((num_steps, 3))
    t = 0.0

    for i in range(num_steps):
        data[i] = state
        state = rk4_step(state, t, dt, sigma, rho, beta)
        t += dt

    return data

# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


# ---------------------------
# Transformer for Time Series
# ---------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(d_model, output_dim if output_dim else input_dim)  # 预测输出维度

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]           # 取最后一个时间步
        out = self.regressor(out)  # (batch, 1)
        return out
    

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len, noise_std=0.0, train=True):
        """
        series: (N, dim) 数组，N为序列长度，dim为维度（1或3）
        noise_std: 高斯噪声标准差, 比如 0.01 / 0.05 / 0.1;
        train: 训练集 True, 加噪声; 测试集 False, 不加.
        """
        self.series = series
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.train = train

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]  # (seq_len, dim)
        y = self.series[idx+self.seq_len]      # (dim,)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.train and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, y



def predict_future(model, init_seq, steps=30, device="cpu"):
    """
    init_seq: (seq_len, dim) tensor, dim为输入维度
    returns: (steps, dim) numpy数组
    """
    model.eval()
    preds = []
    seq = init_seq.clone().unsqueeze(0).to(device)  # (1, seq_len, dim)

    with torch.no_grad():
        for _ in range(steps):
            pred = model(seq)           # (1, output_dim)
            preds.append(pred.cpu().numpy())

            # 拼接预测值到序列末尾，丢弃最旧一步
            pred_expanded = pred.unsqueeze(1)       # (1, 1, output_dim)
            seq = torch.cat([seq[:, 1:, :], pred_expanded], dim=1)

    return np.array(preds).squeeze()  # (steps, output_dim)


# ---------------------------
# 生成 Lorenz 系统数据
# ---------------------------
lorenz_data = generate_lorenz_data(
    sigma=10.0, rho=28.0, beta=8/3,
    dt=0.01, num_steps=10000,
    initial_state=[1.0, 1.0, 1.0]
)

seq_len = 50

train_dataset = TimeSeriesDataset(lorenz_data, seq_len, noise_std=0.5, train=True)
test_dataset  = TimeSeriesDataset(lorenz_data, seq_len, noise_std=0.0, train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=3, output_dim=3).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(20):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # y: (batch, 3)
        optimizer.zero_grad()
        pred = model(x)  # (batch, 3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

model.eval()
with torch.no_grad():
    x, y = test_dataset[0]
    x = x.unsqueeze(0).to(device)   # (1, seq_len, 3)
    pred = model(x)                 # (1, 3)
    print("True:", y.numpy(), "Pred:", pred.cpu().numpy().squeeze())

# 选择起始点
start_idx = 400
x_init, _ = test_dataset[start_idx]
x_init = x_init.to(device)

future_steps = 30
future_preds = predict_future(model, x_init, steps=future_steps, device=device)  # (steps, 3)

history = x_init.cpu().numpy()                    # (seq_len, 3)
true_future = lorenz_data[start_idx+seq_len:start_idx+seq_len+future_steps]  # (steps, 3)

# 计算误差
errors = np.abs(future_preds - true_future)  # (steps, 3)
mean_errors = errors.mean(axis=0)  # (3,)

print(f"\nMean prediction error per dimension:")
print(f"  x: {mean_errors[0]:.4f}")
print(f"  y: {mean_errors[1]:.4f}")
print(f"  z: {mean_errors[2]:.4f}")

# ---------------------------
# 双重可视化
# ---------------------------
fig = plt.figure(figsize=(16, 7))

# 子图1: 3D空间轨迹图
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# 历史轨迹
ax1.plot(history[:, 0], history[:, 1], history[:, 2],
         label='History', color='blue', linewidth=2)

# 真实未来轨迹
ax1.plot(true_future[:, 0], true_future[:, 1], true_future[:, 2],
         label='True Future', color='green', marker='o', markersize=4)

# 预测轨迹
ax1.plot(future_preds[:, 0], future_preds[:, 1], future_preds[:, 2],
         label='Predicted', color='red', marker='x', markersize=4)

# 连接线
ax1.plot([history[-1, 0], true_future[0, 0]],
         [history[-1, 1], true_future[0, 1]],
         [history[-1, 2], true_future[0, 2]], 'k--', alpha=0.3)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'3D Trajectory Prediction (Next {future_steps} Steps)')
ax1.legend()

# 子图2: 3个分量的2D时间序列图
ax2 = fig.add_subplot(1, 2, 2)
colors = ['r', 'g', 'b']
labels = ['x', 'y', 'z']

history_time = np.arange(seq_len)
future_time = np.arange(seq_len, seq_len + future_steps)

for dim in range(3):
    # 历史
    ax2.plot(history_time, history[:, dim],
             color=colors[dim], alpha=0.7, linestyle='-', linewidth=2)
    # 真实未来
    ax2.plot(future_time, true_future[:, dim],
             color=colors[dim], alpha=0.7, marker='o', label=f'True {labels[dim]}')
    # 预测
    ax2.plot(future_time, future_preds[:, dim],
             color=colors[dim], alpha=0.7, marker='x', linestyle='--',
             label=f'Pred {labels[dim]}', markersize=4)

ax2.axvline(seq_len - 1, linestyle='--', color='gray', alpha=0.5)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Value')
ax2.set_title(f'Component-wise Prediction (Next {future_steps} Steps)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
