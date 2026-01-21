# CausalGNN Multidim v3 项目汇总文档

> **项目概述**: 基于图神经网络的多变量时间序列异常检测方法，引入节点与连边注意力机制，并在训练时采用因果干预正则项提升模型泛化能力。

---

## 一、项目目录结构

```
CausalGNN_multidim_v3/
├── main_ATSD.py                      # 主程序入口
├── model.py                          # 模型定义（核心文件）
├── train.py                          # 基础训练逻辑
├── train_causal.py                   # 因果训练逻辑
├── opts.py                           # 参数配置
├── utils.py                          # 工具函数
├── gcn_conv.py                       # GCN卷积层实现
├── featgen.py                        # 特征生成工具
├── gengraph.py                       # 图生成工具
├── generate_ATSD_dataset.py          # ATSD数据集生成器
├── analysis_node_attention_multidim.py  # 节点注意力分析
├── synthetic_structsim.py            # 合成数据集
└── test_ATSD.py                      # 测试脚本
```

---

## 二、各文件功能详解

### 2.1 核心入口文件

#### `main_ATSD.py` (33行)
**功能**: 主程序入口

**主要逻辑**:
1. 加载ATSD数据集（训练集、验证集、测试集）
2. 根据模型类型选择训练方式：
   - GCN: 调用 `train_baseline_syn`
   - CausalGCN系列: 调用 `train_causal_syn`
3. 数据加载路径: `/disk1/xujing.zbl/CAL/CAL/data/`

**支持模型**:
- GCN（基线）
- CausalGCN_CAL_M_D
- CausalGCN_CAL_M
- CausalGCN_CAL
- CausalGCN_CAL_M_D_wo_season
- CausalGCN_CAL_Multiscale
- CausalGCN_CAL_M_Multiscale

---

### 2.2 模型定义文件

#### `model.py` (约4000行)
**功能**: 定义所有图神经网络模型

**核心组件**:

| 组件类 | 功能描述 |
|--------|----------|
| `my_Layernorm` | 自定义LayerNorm，用于季节性特征处理 |
| `moving_avg` | 移动平均层 |
| `series_decomp` | 时间序列分解（趋势-季节分解） |
| `series_decomp_multi` | 多尺度时间序列分解 |
| `DFT_series_decomp` | 基于DFT的序列分解 |
| `MultiScaleFeatureExtractor` | 多尺度特征提取器 |
| `MultiScaleFeatureExtractor_eq` | 等长多尺度特征提取 |
| `MultiScaleFeatureExtractor_last` | 最后层多尺度特征提取 |

**主要模型类**:

| 模型类 | 特点 |
|--------|------|
| `CausalGCN_best_1` | 基础因果GCN，仅最后做一次node attention |
| `CausalGCN_CAL_M` | **核心模型**，多维度node attention + edge attention + 随机干预 |
| `CausalGCN_CAL_M_D` | 带季节分解的CausalGCN_CAL_M |
| `CausalGCN_CAL_M_D_wo_season` | 不含季节分层的版本 |
| `CausalGCN_CAL_M_Multiscale` | **核心模型** + 多尺度特征提取 |
| `CausalGCN_CAL` | 不含随机干预的版本 |
| `CausalGCN_CAL_Multiscale` | CAL + 多尺度特征提取 |
| `GCNNet/GINNet/GATNet` | 基线模型 |

**模型关键机制**:

```
输入数据 (x, edge_index, edge_attr)
    ↓
BatchNorm + Linear变换
    ↓
【Edge Attention】
edge_rep = [x[row], x[col]]
edge_att = Softmax(MLP(edge_rep))
edge_weight_c = edge_att[:, 0]  # context权重
edge_weight_o = edge_att[:, 1]  # object权重
    ↓
【Node Attention (Multi-dimensional)】
node_att = MLP(x)  # [num_nodes, 2*hidden]
node_att = Softmax(node_att, dim=1)
node_att_c = node_att[:, 0, :]  # [num_nodes, hidden]
node_att_o = node_att[:, 1, :]
xc = node_att_c * x
xo = node_att_o * x
    ↓
【GCN Propagation】
xc = ReLU(GCN_conv(xc, edge_index, edge_weight_c))
xo = ReLU(GCN_conv(xo, edge_index, edge_weight_o))
    ↓
【Readout Layers】
xc_logits = context_readout_layer(xc)  # 预测为均匀分布
xo_logits = objects_readout_layer(xo)  # 预测为真实标签
xco_logits, xoc_logits = random_readout_layer(xc, xo)  # 因果干预
    ↓
输出: (xc_logits, xo_logits, xco_logits, xoc_logits)
```

**因果干预机制** (CausalGCN_CAL_M):

```python
# 随机匹配所有样本的context和object
# 计算所有可能的 (batch_size × batch_size) 组合
x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [B, B, num_vars, hidden]
# 通过MLP后计算平均，实现因果干预
xc_mean = x.mean(dim=1)
xo_mean = x.mean(dim=0)
```

---

### 2.3 训练相关文件

#### `train.py` (235行)
**功能**: 基础GCN模型的训练

**主要函数**:
- `train_baseline_syn()`: 主训练循环
- `train()`: 单个epoch训练
- `eval_acc()`: 评估函数，返回Precision/Recall/F1

**评估指标**: 使用PR曲线计算最佳F1分数

#### `train_causal.py` (553行)
**功能**: 因果GCN模型的训练

**主要函数**:
- `train_causal_syn()`: 因果模型主训练循环
- `train_causal_epoch()`: 单个epoch训练
- `eval_acc_causal()`: 因果模型评估
- `test_causal_syn()`: 测试并保存attention权重

**损失函数**:
```python
loss = args.c * c_loss + args.o * o_loss + args.co * co_loss + args.oc * oc_loss
```

其中：
- `c_loss`: KL散度，预测为数据分布（最小化因果影响）
- `o_loss`: NLL损失，预测为真实标签
- `co_loss`: NLL损失，融合预测
- `oc_loss`: KL散度，反向融合预测

---

### 2.4 配置文件

#### `opts.py` (201行)
**功能**: 参数解析和模型工厂

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | CausalGCN_CAL_M_Multiscale | 模型类型 |
| `--feature_dim` | 100 | 时间序列长度（滑动窗口） |
| `--node_num` | 85 | 监控指标个数 |
| `--hidden` | 256 | 隐藏层维度 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--c` | 0.5 | c_loss权重 |
| `--o` | 1.0 | o_loss权重 |
| `--co` | 0.5 | co_loss权重 |
| `--oc` | 0.5 | oc_loss权重 |
| `--lr` | 0.0001 | 学习率 |
| `--seed` | 1111 | 随机种子 |

---

### 2.5 数据处理文件

#### `generate_ATSD_dataset.py` (386行)
**功能**: 生成ATSD数据集的图结构

**数据格式**:
- 输入: ATSD标准格式CSV文件
- 输出: PyTorch Geometric Data对象列表

**图构建逻辑**:
```python
# 1. 计算特征相关性矩阵
correlation_matrix = np.corrcoef(features)

# 2. 根据阈值构建边
edge_index = []
for i, j where abs(correlation_matrix[i,j]) > threshold:
    edge_index.append([i, j])
    edge_index.append([j, i])

# 3. 边属性为相关系数
edge_attr = correlation_matrix[i, j]
```

**数据类** `ATSDSegLoader`:
- 支持按设备归一化 (`by_device`)
- 滑动窗口采样
- 训练/验证/测试集划分

---

### 2.6 工具文件

#### `utils.py` (205行)
**功能**: 数据集生成和划分工具

**主要函数**:
- `k_fold()`: K折交叉验证划分
- `creat_one_pyg_graph()`: 创建单个图数据
- `graph_dataset_generate()`: 生成合成图数据集
- `dataset_bias_split()`: 按偏斜比例划分数据集
- `print_dataset_info()`: 打印数据集统计信息

#### `gcn_conv.py` (108行)
**功能**: 自定义GCN卷积层

**特点**:
- 支持边归一化 (`edge_norm`)
- 支持特征变换模式 (`gfn`)
- 继承自 `MessagePassing`

#### `featgen.py` (77行)
**功能**: 节点特征生成器

**类**:
- `FeatureGen`: 抽象基类
- `ConstFeatureGen`: 常数特征
- `GaussianFeatureGen`: 高斯分布特征
- `GridFeatureGen`: 网格特征

#### `gengraph.py` (79行)
**功能**: 图结构生成器

**支持的形状**:
- house, cycle, grid, diamond
- BA无标度图、树结构

#### `synthetic_structsim.py` (289行)
**功能**: 合成图结构的形状定义

---

### 2.7 分析文件

#### `analysis_node_attention_multidim.py` (511行)
**功能**: 分析节点和边的注意力权重

**主要分析**:
1. 加载保存的attention结果
2. 提取Top-K节点和边
3. 计算异常覆盖率
4. 绘制热力图

**关键函数**:
- `split_edges_and_attention()`: 按样本分割边
- `get_top_k_edges_and_attention()`: 获取Top-K边
- `calculate_total_match_ratio()`: 计算覆盖率
- `plot_connected_timeseries()`: 绘制时序图

#### `test_ATSD.py` (46行)
**功能**: 测试脚本，加载模型并评估

---

## 三、项目整体逻辑流程

```
┌─────────────────────────────────────────────────────────────┐
│                    1. 数据准备阶段                           │
├─────────────────────────────────────────────────────────────┤
│  generate_ATSD_dataset.py                                   │
│    ├── 读取ATSD原始CSV数据                                   │
│    ├── 按设备归一化                                          │
│    ├── 滑动窗口分割 (win_size=100)                           │
│    ├── 计算相关性矩阵构建图结构 (cor_threshold=0.2)          │
│    └── 保存为.pt格式                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. 模型训练阶段                           │
├─────────────────────────────────────────────────────────────┤
│  main_ATSD.py → train_causal.py                             │
│    ├── 加载数据集 (train/val/test)                          │
│    ├── 初始化模型 (CausalGCN_CAL_M_Multiscale)              │
│    ├── 训练循环                                             │
│    │   ├── Edge Attention: 学习边的因果/非因果权重           │
│    │   ├── Node Attention: 多维特征级别的注意力              │
│    │   ├── GCN传播: 分别在context/object空间传播             │
│    │   ├── 因果干预: 随机匹配context和object                 │
│    │   └── 损失计算: c_loss + o_loss + co_loss + oc_loss     │
│    └── 保存最佳模型                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. 测试与分析阶段                         │
├─────────────────────────────────────────────────────────────┤
│  test_ATSD.py / analysis_node_attention_multidim.py        │
│    ├── 加载最佳模型                                          │
│    ├── 在测试集上评估                                        │
│    ├── 保存attention权重                                     │
│    └── 分析attention与异常的关系                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、核心技术创新点

### 4.1 多维度节点注意力 (Multi-dimensional Node Attention)

传统模型使用单一注意力权重，本项目实现了**特征级别的注意力**:

```python
# 传统: [num_nodes, 2] - 每个节点一个权重
node_att = Softmax(MLP(x))  # shape: [N, 2]
xc = node_att[:, 0:1] * x
xo = node_att[:, 1:2] * x

# 本项目: [num_nodes, 2, hidden] - 每个特征维度独立权重
node_att = MLP(x).view(N, 2, hidden)
node_att = Softmax(node_att, dim=1)
node_att_c = node_att[:, 0, :]  # [N, hidden]
node_att_o = node_att[:, 1, :]
xc = node_att_c * x  # 逐元素相乘
xo = node_att_o * x
```

### 4.2 因果干预正则项

通过随机匹配context和object特征，实现因果关系的干预:

```python
# 跨样本所有可能的组合
x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [B, B, num_vars, hidden]
# 计算平均，打破虚假关联
xc_mean = x.mean(dim=1)
xo_mean = x.mean(dim=0)
```

### 4.3 多尺度特征提取

```python
# 下采样获取多尺度表示
scale_list = [x, downsample(x, 2), downsample(x, 4), downsample(x, 8)]
# 自顶向上融合
for i in range(len(scales) - 1):
    low_res = upsample(low_res)
    high_res = high_res + low_res
```

### 4.4 边注意力机制

```python
edge_rep = concat([x[row], x[col]])  # 边两端节点特征拼接
edge_att = Softmax(MLP(edge_rep))  # [num_edges, 2]
edge_weight_c = edge_att[:, 0]  # context传播权重
edge_weight_o = edge_att[:, 1]  # object传播权重
```

---

## 五、数据格式说明

### 5.1 输入数据格式 (ATSD)

**CSV文件格式**:
```
device, timestamp_(min), metric1, metric2, ..., metric85
dev001, 0, 0.52, 0.38, ..., 0.91
dev001, 1, 0.51, 0.39, ..., 0.92
...
```

**标签格式**:
```
device, timestamp_(min), label1, label2, ..., label85
dev001, 0, 0, 0, ..., 1  # 第85个指标异常
```

### 5.2 图数据格式 (PyTorch Geometric)

```python
Data(
    x: Tensor [num_nodes, feature_dim],  # [85, 100]
    edge_index: Tensor [2, num_edges],   # 边索引
    edge_attr: Tensor [num_edges, 1],    # 边权重（相关系数）
    y: Tensor [num_nodes],               # 节点标签
    mask: Tensor [num_nodes]             # 节点掩码
)
```

---

## 六、实验设置

### 6.1 训练参数

| 参数 | 值 |
|------|-----|
| 模型 | CausalGCN_CAL_M_Multiscale |
| 特征维度 | 100 (时间窗口) |
| 节点数 | 85 (监控指标) |
| 隐藏层维度 | 256 |
| 批大小 | 64 |
| 学习率 | 0.0001 |
| 训练轮数 | 100 |
| 优化器 | Adam |
| 学习率调度 | CosineAnnealingLR |

### 6.2 损失权重

```python
c = 0.5    # context损失权重
o = 1.0    # object损失权重
co = 0.5   # 融合损失权重
oc = 0.5   # 反向融合损失权重
```

### 6.3 评估指标

- Precision
- Recall
- F1 Score (PR曲线最佳阈值)

---

## 七、运行示例

### 训练模型
```bash
python main_ATSD.py --model CausalGCN_CAL_M_Multiscale \
    --feature_dim 100 \
    --node_num 85 \
    --hidden 256 \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0001 \
    --seed 1111
```

### 测试模型
```bash
python test_ATSD.py --model CausalGCN_CAL_M_Multiscale \
    --feature_dim 100 \
    --node_num 85
```

### 分析注意力
```bash
python analysis_node_attention_multidim.py
```

---

## 八、依赖环境

```
torch
torch-geometric
torch-scatter
numpy
pandas
scikit-learn
matplotlib
seaborn
networkx
scipy
tqdm
```

---

## 九、注意事项

1. **数据路径**: 代码中硬编码了服务器路径 `/disk1/xujing.zbl/CAL/CAL/data/`，使用前需要修改
2. **GPU设置**: 部分代码硬编码了 `cuda:3` 或 `cuda:4`，需要根据实际GPU调整
3. **模型保存位置**: `checkpoints/` 目录
4. **日志保存位置**: `logs/` 目录
5. **Attention保存**: 测试时会创建 `{model_name}_.../` 目录保存attention权重

---

*文档生成日期: 2025年*
