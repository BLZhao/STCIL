"""
Lorenz 系统预测改进方案对比实验 (修改版)

修改参数:
- 噪声强度: 0.0 (无噪声)
- 不使用数据归一化
- 序列长度: 100
- 预测步数: 20

对比三种改进方案:
1. 方案1: 仅单步预测 (基准)
2. 方案2: Seq2Seq多步预测
3. 方案3: Seq2Seq + 自适应学习率
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# ---------------------------
# Lorenz System Generator
# ---------------------------
def generate_lorenz_data(
    sigma=10.0, rho=28.0, beta=8/3,
    dt=0.01, num_steps=10000,
    initial_state=None
):
    """使用4阶Runge-Kutta方法生成Lorenz系统数据"""
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
# Data Normalizer (方案1)
# ---------------------------
class DataNormalizer:
    """数据标准化类"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """计算数据的均值和标准差"""
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        return self

    def transform(self, data):
        """标准化数据"""
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """反标准化数据"""
        if data.ndim == 1:
            return data * self.std + self.mean
        return data * self.std + self.mean

    def fit_transform(self, data):
        """拟合并转换数据"""
        self.fit(data)
        return self.transform(data)


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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------
# Baseline Model: 单步预测 Transformer
# ---------------------------
class BaselineTransformer(nn.Module):
    """基准模型: 单步预测"""
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, output_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]
        out = self.regressor(out)
        return out


# ---------------------------
# Seq2Seq Model: 多步预测 Transformer (方案2)
# ---------------------------
class Seq2SeqTransformer(nn.Module):
    """Seq2Seq模型: 一次性预测未来 N 步"""
    def __init__(self, input_dim=3, pred_steps=30, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1, output_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pred_steps = pred_steps
        self.output_dim = output_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(d_model, pred_steps * output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]
        out = self.regressor(out)
        out = out.view(x.size(0), self.pred_steps, self.output_dim)
        return out


# ---------------------------
# Dataset Classes
# ---------------------------
class SingleStepDataset(Dataset):
    """单步预测数据集"""
    def __init__(self, series, seq_len, noise_std=0.0, train=True):
        self.series = series
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.train = train

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if self.train and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x, y


class Seq2SeqDataset(Dataset):
    """多步预测数据集"""
    def __init__(self, series, seq_len, pred_steps, noise_std=0.0, train=True):
        self.series = series
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.noise_std = noise_std
        self.train = train

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_steps + 1

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len:idx+self.seq_len+self.pred_steps]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if self.train and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x, y


# ---------------------------
# Prediction Functions
# ---------------------------
def predict_baseline(model, init_seq, steps=30, device="cpu"):
    """基准模型: 自回归预测"""
    model.eval()
    preds = []
    seq = init_seq.clone().unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(steps):
            pred = model(seq)
            preds.append(pred.cpu().numpy())
            pred_expanded = pred.unsqueeze(1)
            seq = torch.cat([seq[:, 1:, :], pred_expanded], dim=1)
    return np.array(preds).squeeze()


def predict_seq2seq(model, init_seq, steps=30, device="cpu"):
    """Seq2Seq模型: 一次性预测"""
    model.eval()
    with torch.no_grad():
        init_seq = init_seq.unsqueeze(0).to(device)
        pred = model(init_seq)
        return pred.cpu().numpy().squeeze()


# ---------------------------
# Experiment Functions
# ---------------------------
def run_experiment_baseline(lorenz_data, seq_len=100, epochs=20, device="cpu"):
    """实验1: 基准模型 (单步预测 + 固定LR)"""
    print("  [训练中...] 方案1: 单步预测 + 固定LR")

    dataset = SingleStepDataset(lorenz_data, seq_len, noise_std=0.0, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BaselineTransformer(input_dim=3, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    return model


def run_experiment_seq2seq(lorenz_data, seq_len=100, pred_steps=20, epochs=20, device="cpu"):
    """实验2: Seq2Seq多步预测 + 固定LR"""
    print("  [训练中...] 方案2: Seq2Seq多步预测 + 固定LR")

    dataset = Seq2SeqDataset(lorenz_data, seq_len, pred_steps, noise_std=0.0, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Seq2SeqTransformer(input_dim=3, pred_steps=pred_steps, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    return model


def run_experiment_seq2seq_scheduler(lorenz_data, seq_len=100, pred_steps=20, epochs=20, device="cpu"):
    """实验3: Seq2Seq + 自适应学习率"""
    print("  [训练中...] 方案3: Seq2Seq + 自适应LR")

    dataset = Seq2SeqDataset(lorenz_data, seq_len, pred_steps, noise_std=0.0, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Seq2SeqTransformer(input_dim=3, pred_steps=pred_steps, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 自适应学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def compute_metrics(pred, true):
    """计算评估指标"""
    mae = np.abs(pred - true).mean(axis=0)
    rmse = np.sqrt(((pred - true) ** 2).mean(axis=0))
    total_mae = mae.mean()
    total_rmse = rmse.mean()
    return mae, rmse, total_mae, total_rmse


# ---------------------------
# Main Experiment
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成数据
    print("Generating Lorenz system data...")
    lorenz_data = generate_lorenz_data(
        sigma=10.0, rho=28.0, beta=8/3,
        dt=0.01, num_steps=10000,
        initial_state=[1.0, 1.0, 1.0]
    )

    seq_len = 100  # 修改: 从 50 增加到 100
    pred_steps = 20  # 修改: 从 30 减少到 20
    start_idx = 400
    epochs = 20

    # 准备测试数据
    test_x = torch.tensor(lorenz_data[start_idx:start_idx+seq_len], dtype=torch.float32)
    true_future = lorenz_data[start_idx+seq_len:start_idx+seq_len+pred_steps]

    results = {}
    predictions = {}

    print("=" * 60)
    print("Lorenz System Prediction - Method Comparison")
    print("=" * 60)
    print(f"Data: {len(lorenz_data)}, Seq Len: {seq_len}, Pred Steps: {pred_steps}")
    print(f"Epochs: {epochs}, Device: {device}, Noise: 0.0 (no normalization)")
    print("-" * 60)

    # 实验1: 基准模型 (单步预测 + 固定LR)
    print("\n[Exp 1] Baseline: Single-step + Fixed LR")
    model_baseline = run_experiment_baseline(lorenz_data, seq_len, epochs, device)
    pred_baseline = predict_baseline(model_baseline, test_x, pred_steps, device)
    mae, rmse, total_mae, total_rmse = compute_metrics(pred_baseline, true_future)
    results['baseline'] = {'mae': mae, 'rmse': rmse, 'total_mae': total_mae, 'total_rmse': total_rmse}
    predictions['baseline'] = pred_baseline
    print(f"  MAE: x={mae[0]:.4f}, y={mae[1]:.4f}, z={mae[2]:.4f}, total={total_mae:.4f}")

    # 实验2: Seq2Seq + 固定LR
    print("\n[Exp 2] Seq2Seq Multi-step + Fixed LR")
    model_seq2seq = run_experiment_seq2seq(lorenz_data, seq_len, pred_steps, epochs, device)
    pred_seq2seq = predict_seq2seq(model_seq2seq, test_x, pred_steps, device)
    mae, rmse, total_mae, total_rmse = compute_metrics(pred_seq2seq, true_future)
    results['seq2seq'] = {'mae': mae, 'rmse': rmse, 'total_mae': total_mae, 'total_rmse': total_rmse}
    predictions['seq2seq'] = pred_seq2seq
    print(f"  MAE: x={mae[0]:.4f}, y={mae[1]:.4f}, z={mae[2]:.4f}, total={total_mae:.4f}")
    print(f"  vs Baseline: {(1 - total_mae/results['baseline']['total_mae'])*100:+.2f}%")

    # 实验3: Seq2Seq + 自适应LR
    print("\n[Exp 3] Seq2Seq + Adaptive LR Scheduler")
    model_scheduler = run_experiment_seq2seq_scheduler(lorenz_data, seq_len, pred_steps, epochs, device)
    pred_scheduler = predict_seq2seq(model_scheduler, test_x, pred_steps, device)
    mae, rmse, total_mae, total_rmse = compute_metrics(pred_scheduler, true_future)
    results['scheduler'] = {'mae': mae, 'rmse': rmse, 'total_mae': total_mae, 'total_rmse': total_rmse}
    predictions['scheduler'] = pred_scheduler
    print(f"  MAE: x={mae[0]:.4f}, y={mae[1]:.4f}, z={mae[2]:.4f}, total={total_mae:.4f}")
    print(f"  vs Baseline: {(1 - total_mae/results['baseline']['total_mae'])*100:+.2f}%")

    # 打印汇总表格
    print("\n" + "=" * 60)
    print("Results Summary (MAE)")
    print("=" * 60)
    print(f"{'Method':<30} {'x':>8} {'y':>8} {'z':>8} {'Total':>8} {'Improvement':>12}")
    print("-" * 60)

    baseline_mae = results['baseline']['total_mae']
    for name, label in [
        ('baseline', 'Baseline (Single-step, Fixed LR)'),
        ('seq2seq', 'Seq2Seq (Multi-step, Fixed LR)'),
        ('scheduler', 'Seq2Seq + Adaptive LR')
    ]:
        r = results[name]
        improvement = (1 - r['total_mae']/baseline_mae) * 100 if name != 'baseline' else 0
        imp_str = f"{improvement:+.1f}%" if name != 'baseline' else "-"
        print(f"{label:<30} {r['mae'][0]:>8.4f} {r['mae'][1]:>8.4f} {r['mae'][2]:>8.4f} {r['total_mae']:>8.4f} {imp_str:>12}")

    print("=" * 60)

    # 可视化
    print("\nGenerating visualization...")
    visualize_results(true_future, predictions, results)

    return results, predictions


def visualize_results(true_future, predictions, results):
    """可视化结果对比"""
    fig = plt.figure(figsize=(18, 10))

    colors = {'baseline': 'red', 'seq2seq': 'blue', 'scheduler': 'green'}
    labels = {
        'baseline': 'Baseline (Single-step)',
        'seq2seq': 'Seq2Seq (Multi-step)',
        'scheduler': 'Seq2Seq + Scheduler'
    }

    # 子图1: 3D轨迹对比
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # 真实轨迹
    ax1.plot(true_future[:, 0], true_future[:, 1], true_future[:, 2],
             label='True Future', color='black', linewidth=3, alpha=0.7)

    # 各方案预测轨迹
    for key, pred in predictions.items():
        ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2],
                 label=labels[key], color=colors[key], linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend(fontsize=7)

    # 子图2: MAE 柱状图
    ax2 = fig.add_subplot(2, 2, 2)
    methods = ['Baseline\n(Single-step)', 'Seq2Seq\n(Multi-step)', 'Seq2Seq +\nScheduler']
    x_pos = np.arange(len(methods))
    maes = [results[k]['total_mae'] for k in ['baseline', 'seq2seq', 'scheduler']]
    bars = ax2.bar(x_pos, maes, color=[colors[k] for k in ['baseline', 'seq2seq', 'scheduler']], alpha=0.7)

    # 在柱子上标注数值
    for i, (bar, mae) in enumerate(zip(bars, maes)):
        height = bar.get_height()
        improvement = (1 - mae/maes[0]) * 100 if i > 0 else 0
        text = f'{mae:.3f}'
        if i > 0:
            text += f'\n{improvement:+.1f}%'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 text, ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Method')
    ax2.set_ylabel('MAE (Mean Absolute Error)')
    ax2.set_title('Prediction Error Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.grid(axis='y', alpha=0.3)

    # 子图3: X 分量对比
    ax3 = fig.add_subplot(2, 2, 3)
    time = np.arange(len(true_future))
    ax3.plot(time, true_future[:, 0], label='True', color='black', linewidth=2, alpha=0.7)
    for key, pred in predictions.items():
        ax3.plot(time, pred[:, 0], label=labels[key], color=colors[key], linestyle='--', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('X Value')
    ax3.set_title('X Component Comparison')
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # 子图4: Y 分量对比
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time, true_future[:, 1], label='True', color='black', linewidth=2, alpha=0.7)
    for key, pred in predictions.items():
        ax4.plot(time, pred[:, 1], label=labels[key], color=colors[key], linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Y Value')
    ax4.set_title('Y Component Comparison')
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=150, bbox_inches='tight')
    print("  Visualization saved to: experiment_results.png")
    plt.show()


if __name__ == "__main__":
    results, predictions = main()
