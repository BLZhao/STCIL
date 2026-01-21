import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
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
        self.regressor = nn.Linear(d_model, 1)  # 预测一个时间步

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
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        if self.train and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, y



def predict_future(model, init_seq, steps=30, device="cpu"):
    """
    init_seq: (seq_len, 1) tensor
    returns: numpy array of length = steps
    """
    model.eval()
    preds = []
    seq = init_seq.clone().unsqueeze(0).to(device)  # (1, seq_len, 1)

    with torch.no_grad():
        for _ in range(steps):
            pred = model(seq)           # (1, 1)
            preds.append(pred.item())

            # 拼接预测值到序列末尾，丢弃最旧一步
            pred_expanded = pred.unsqueeze(-1)       # (1, 1, 1)
            seq = torch.cat([seq[:, 1:, :], pred_expanded], dim=1)

    return np.array(preds)


# 生成一个简单正弦序列
t = np.arange(0, 1000, 0.1)
series = np.sin(t)

seq_len = 50

train_dataset = TimeSeriesDataset(series, seq_len, noise_std=1, train=True)
test_dataset  = TimeSeriesDataset(series, seq_len, noise_std=1,  train=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(20):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

model.eval()
with torch.no_grad():
    x, y = test_dataset[0]
    x = x.unsqueeze(0).to(device)   # (1, seq_len, 1)
    pred = model(x)
    print("True:", y.item(), "Pred:", pred.item())

# 选择起始点
start_idx = 400
x_init, _ = test_dataset[start_idx]
x_init = x_init.to(device)

future_steps = 30
future_preds = predict_future(model, x_init, steps=future_steps, device=device)

history = x_init.cpu()
true_future = series[start_idx+seq_len:start_idx+seq_len+future_steps]

# 计算误差
errors = np.abs(future_preds - true_future)

plt.figure(figsize=(12, 6))

# 历史 & 未来
plt.plot(range(seq_len), history, label="History Input", linewidth=2)
plt.plot(range(seq_len, seq_len+future_steps), true_future, label="True Future", marker="o")
plt.plot(range(seq_len, seq_len+future_steps), future_preds, label="Predicted Future", marker="x")

# 误差标注
for i in range(future_steps):
    t = seq_len + i
    plt.text(t, future_preds[i],
             f"{errors[i]:.3f}",
             fontsize=8,
             ha="center",
             va="bottom")

plt.axvline(seq_len-1, linestyle="--", color="gray")
plt.legend()
plt.title("Transformer Forecast: Next 30 Steps with Per-Step Error")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(True)
plt.show()
