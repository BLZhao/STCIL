import os
import heapq
import torch
import pickle
import opts
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import DataLoader
import pickle
import seaborn as sns

def save_with_pickle(array_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(array_list, f)

def load_with_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def split_edges_and_attention(edge_index, edge_attention, nodes_per_graph=85):
    # 根据总节点数计算batch size
    total_nodes = edge_index.max().item()
    batch_size = total_nodes // nodes_per_graph + 1
    
    # 创建空列表存储每个样本的边和attention
    batch_edges = []
    batch_attentions = []
    
    for i in range(batch_size):
        # 计算当前样本节点的范围
        start_idx = i * nodes_per_graph
        end_idx = (i + 1) * nodes_per_graph
        
        # 找到属于当前样本的边
        mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) & 
                (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
        
        # 提取当前样本的边和对应的attention
        current_edges = edge_index[:, mask]
        current_attention = edge_attention[mask, -1]
        
        # 将节点索引转换为0-84的范围
        current_edges = current_edges - start_idx
        
        batch_edges.append(current_edges)
        batch_attentions.append(current_attention)
    
    return batch_edges, batch_attentions

def get_top_k_edges_and_attention(batch_edges, batch_attentions, k=5):
    result_edges = []
    result_attentions = []
    
    for edges, attention in zip(batch_edges, batch_attentions):
        # 获取前k个最大值的索引
        if len(attention) >= k:
            # 如果边数大于等于k，取前k个最大值
            top_k_values, top_k_indices = torch.topk(torch.tensor(attention), k)
        else:
            # 如果边数小于k，取所有的边
            top_k_values = attention
            top_k_indices = torch.arange(len(attention))
        
        # 获取对应的边
        selected_edges = edges[:, top_k_indices]
        
        result_edges.append(selected_edges)
        result_attentions.append(top_k_values)
    
    return result_edges, result_attentions

def plot_connected_timeseries(top_edges, data_test, chosen_idx):   
    # 获取边的数量
    num_edges = top_edges.shape[1]  # 8条边
    
    # 创建子图布局 (2x4)
    fig, axes = plt.subplots((num_edges+1)//2, 2, figsize=(20, (num_edges+1)//2*5))
    fig.suptitle('Connected Nodes Time Series Analysis', fontsize=16)
    
    # 将axes展平以便迭代
    axes = axes.flatten()
    
    # 为每条边创建一个子图
    for i in range(num_edges):
        ax1 = axes[i]
        node1_idx = top_edges[0][i]
        node2_idx = top_edges[1][i]
        
        # 创建第二个Y轴
        ax2 = ax1.twinx()
        
        # 绘制第一个节点的时序数据 (左Y轴)
        line1 = ax1.plot(data_test[chosen_idx, node1_idx], color='blue', label=f'Node {node1_idx}')
        ax1.set_ylabel(f'Node {node1_idx} Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 绘制第二个节点的时序数据 (右Y轴)
        line2 = ax2.plot(data_test[chosen_idx, node2_idx], color='red', label=f'Node {node2_idx}')
        ax2.set_ylabel(f'Node {node2_idx} Value', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 设置标题和X轴标签
        ax1.set_title(f'Edge {i+1}: Node {node1_idx} - Node {node2_idx}')
        ax1.set_xlabel('Time')
        
        # 合并两个轴的图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # 添加网格
        ax1.grid(True)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    save_path = "/disk1/xujing.zbl/CAL/CAL/visualization"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/edge_attention_timeseries_v4.1_{chosen_idx}.png')
    plt.close()

def calculate_total_match_ratio(indices_max10, true_labels_2d):
    # 验证输入维度
    assert indices_max10.shape[0] == true_labels_2d.shape[0]
    # assert indices_max10.shape[1] == 10
    
    # 找出所有样本中1的实际位置
    total_true_indices = []
    total_matches = 0
    total_ones = 0
    
    # 遍历所有样本
    for i in range(len(indices_max10)):
        # 获取当前样本中1的索引位置
        true_indices = np.where(true_labels_2d[i] == 1)[0]
        total_ones += len(true_indices)
        
        # 计算当前样本中1的索引在预测索引中的匹配数
        matches = np.isin(true_indices, indices_max10[i]).sum()
        total_matches += matches
    
    # 计算总体匹配比例
    total_ratio = total_matches / total_ones if total_ones > 0 else 0
    
    return total_ratio

# 使用示例：
# ratio = calculate_total_match_ratio(indices_max10, true_labels_2d)
# print(f"总体匹配比例: {ratio:.4f}")

def calculate_total_edge_match_ratio(top_edges, true_labels_2d):
    # 验证输入维度
    assert len(top_edges) == true_labels_2d.shape[0]
    # assert top_edges.shape[1] == 2
    # assert top_edges.shape[2] == 20
    
    total_matches = 0
    total_ones = 0
    unique_nodes_counts = []  # 存储每个样本的连边中去重节点数量
    
    # 遍历所有样本
    for i in range(len(top_edges)):
        # 获取当前样本中1的索引位置
        true_indices = np.where(true_labels_2d[i] == 1)[0]
        total_ones += len(true_indices)
        
        # 获取当前样本的所有边的节点索引（去重）
        edge_nodes = np.unique(top_edges[i].reshape(-1))
        unique_nodes_counts.append(len(edge_nodes))  # 记录去重后的节点数量
        
        # 计算当前样本中1的索引在边的节点中的匹配数
        matches = np.isin(true_indices, edge_nodes).sum() 
        total_matches += matches
    
    # 计算总体匹配比例
    total_ratio = total_matches / total_ones if total_ones > 0 else 0
    unique_nodes_array = np.array(unique_nodes_counts)
    
    return total_ratio, unique_nodes_array


def calculate_combined_edge_match_ratio(top_edges, true_labels_2d, binary_preds_labels_2d):
    # 验证输入维度
    assert len(top_edges) == true_labels_2d.shape[0]
    # assert len(indices_max) == true_labels_2d.shape[0]
    assert true_labels_2d.shape == binary_preds_labels_2d.shape
    
    total_matches = 0
    total_ones = 0
    unique_nodes_counts = []
    
    # 遍历所有样本
    for i in range(len(top_edges)):
        # 获取当前样本中1的索引位置
        true_indices = np.where(true_labels_2d[i] == 1)[0]
        total_ones += len(true_indices)
        
        # 获取当前样本的所有边的节点索引（去重）
        edge_nodes = np.unique(top_edges[i].reshape(-1))
        
        # 获取binary_preds中预测为1的索引
        pred_indices = np.where(binary_preds_labels_2d[i] == 1)[0]
        
        # 合并edge_nodes和pred_indices并去重
        combined_nodes = np.unique(np.concatenate([edge_nodes, pred_indices]))
        unique_nodes_counts.append(len(combined_nodes))
        
        # 计算当前样本中1的索引在合并节点中的匹配数
        matches = np.isin(true_indices, combined_nodes).sum()
        total_matches += matches
    
    # 计算总体匹配比例
    total_ratio = total_matches / total_ones if total_ones > 0 else 0
    unique_nodes_array = np.array(unique_nodes_counts)
    
    return total_ratio, unique_nodes_array


def get_device_indices(test_device):
    result = {}
    if not test_device:  # 处理空列表的情况
        return result
    
    start_idx = 0
    current_device = test_device[0]
    
    for i in range(1, len(test_device)):
        if test_device[i] != current_device:
            # 当遇到不同的设备时，记录前一个设备的区间
            result[current_device] = (start_idx, i-1)
            # 更新新设备的起始位置和当前设备
            start_idx = i
            current_device = test_device[i]
    
    # 处理最后一个设备
    result[current_device] = (start_idx, len(test_device)-1)
    
    return result



with open('ATSD_v4.3_CausalGCN_CAL_M_Multiscale_last_rand_multidim_cor0.2_modify_norm_win100_by_device_seed999/attention_results.pkl', 'rb') as f:  # 'rb'表示以二进制读取模式打开文件
    load_data = pickle.load(f)

with open('ATSD_v4.3_CausalGCN_CAL_M_Multiscale_last_rand_multidim_cor0.2_modify_norm_win100_by_device_seed999/test_results.pkl', 'rb') as f:
    results = pickle.load(f)

args = opts.parse_args()
top_node_K = 20 ## 选取最大的top_node_K个node attention
top_edge_K = 100 ## 选取最大的top_edge_K个edge attention
save_path = "/disk1/xujing.zbl/CAL/CAL/data"
test_set = torch.load(save_path + "/ATSD_v4.3_test_dataset_with_mask_multidim_cor0.2_modify_norm_by_device.pt", weights_only=False)
test_device = torch.load(save_path + "/ATSD_v4.3_test_device_with_mask_multidim_cor0.5_modify_norm_by_device.pt", weights_only=False)
test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

device_indices = get_device_indices(test_device)

preds = results['predictions']
true_labels = results['true_labels']
# if mask
all_masks = results['masks']
output_preds = results['output_predictions']
threshold = results['threshold']
binary_preds = np.array([1 if x > threshold else 0 for x in output_preds])

print('Test Precision:', results['precision'])
print('Test recall:', results['recall'])
print('Test f1:', results['f1'])
print('Test threshold:', results['threshold'])
print('output_predictions:', len(output_preds))
print('true_labels:', len(true_labels))
print('test results:', len(preds))
print('load_data:', len(load_data))

for i in range(len(load_data)):
    cur_batch = load_data[i]
    print(cur_batch['batch'])
    print('edge_attention', len(cur_batch['edge_attention']))
    print('Max_edge_attention', np.max(cur_batch['edge_attention'][:, -1]))
    print('Min_edge_attention', np.min(cur_batch['edge_attention'][:, -1]))
    print('Mean_edge_attention', np.mean(cur_batch['edge_attention'][:, -1]))
    print('node_attention', len(cur_batch['node_attention']))
    print('edge_index', len(cur_batch['edge_index']), len(cur_batch['edge_index'][0]))
    print('edge_index_example', cur_batch['edge_index'][:, -40:])
    break

##### node attention
result = []
for j in range(len(load_data)):
    array_dict = load_data[j]
    node_attention = array_dict['node_attention']  # shape: (10000, 2)
    batch_info = array_dict['batch']  # shape: (10000,)
    for i in range(max(batch_info)+1):  # 100个batch
        mask = (batch_info == i)
        batch_attention = node_attention[mask]
        result.append(batch_attention)
print('batch_attention:----------------', batch_attention.shape)

reshaped_attention = np.stack(result)  # shape: (2331, 100, 2)
# print(reshaped_attention[186])
reshaped_attention = np.mean(reshaped_attention[:, 1, :], axis=-1)
indices_max20 = np.argsort(reshaped_attention, axis=1)[:, -top_node_K:]
print('shape of indices_max20:', indices_max20.shape)
print('Instance of indices_max20:', indices_max20[-5:])
print('corresponding attention:', reshaped_attention[indices_max20])

# ##### 保存node attention #####
# save_dir_node = 'top_node_attention'
# if not os.path.exists(save_dir_node):
#     os.makedirs(save_dir_node)
# np.save('top_node_attention/model_CausalGIN_multidim_v4.1_by_device_top_20_nodes.npy', indices_max20)

# #### 测试node_attention保存是否正确 #####
# load_node = np.load('top_node_attention/model_CausalGIN_multidim_v4.1_by_device_top_20_nodes.npy')
# print('load-------------:', load_node.shape, load_node[-5:])


##### 将保存的展平的 真实/预测的label重新转化为(batch size, metrics)维度 #####
true_labels_2d = np.array(true_labels).reshape(-1, 85)
preds_labels_2d = np.array(preds).reshape(-1, 85)
binary_preds_labels_2d = np.array(binary_preds).reshape(-1, 85)
print(np.where(np.array(true_labels)==1))
print(np.where(true_labels_2d==1))
true_anonaly_instance = np.where(true_labels_2d==1)[0] ## 发生异常的所有样本
true_unique_anonaly_instance = np.unique(true_anonaly_instance) ## 发生异常的所有样本（不同位置发生异常只算一次）
print('Shape of true_labels_2d:', true_labels_2d.shape)
print('Shape of preds_labels_2d:', preds_labels_2d.shape)
indices_group1 = np.where((true_labels_2d[:, [42, 43]] == 1).all(axis=1))[0]
indices_group1_pred = np.where((binary_preds_labels_2d[:, [42, 43]] == 1).all(axis=1))[0]
common_indices = np.intersect1d(indices_group1, indices_group1_pred)
print('common idx for AD in group1---------------------------------------------:', common_indices)
print('Unique_anonaly_instance:', len(true_unique_anonaly_instance))
print('Unique_anonaly_instance:', true_unique_anonaly_instance[:100])

print(np.where(np.array(binary_preds)==1))
print(np.where(binary_preds_labels_2d==1))
pred_anonaly_instance = np.where(binary_preds_labels_2d==1)[0] ## 模型检测到异常的所有样本
pred_unique_anonaly_instance = np.unique(pred_anonaly_instance)  ## 模型检测到异常的所有样本（不同位置发生异常只算一次）
print('Unique_anonaly_instance:', len(pred_unique_anonaly_instance))
print('Unique_anonaly_instance:', pred_unique_anonaly_instance[:100])

true_positive_instance = [x for x in true_unique_anonaly_instance if x in pred_unique_anonaly_instance] ## 真阳的例子（对样本而言，维度不一定判断准确）
print('true_positive_instance number:', len(true_positive_instance))
print('true_positive_instance number:', true_positive_instance[:100])

true_positive_instance_multidim = np.where(true_labels_2d*binary_preds_labels_2d==1)[0] ## 真阳的例子（维度也必须判断准确）
true_positive_unique_instance_multidim = np.unique(true_positive_instance_multidim)
print('true_positive_instance_multidim number:', len(true_positive_instance_multidim))
print('true_positive_instance_multidim number:', true_positive_instance_multidim[:100])
print('true_positive_unique_instance_multidim number:', len(true_positive_unique_instance_multidim))
print('true_positive_unique_instance_multidim number:', true_positive_unique_instance_multidim[100:200])

print(np.sum(true_labels_2d==preds_labels_2d))
print(np.sum((true_labels_2d==1)*(binary_preds_labels_2d==1))/np.sum(true_labels_2d==1))
print(np.sum((true_labels_2d==1)*(binary_preds_labels_2d==1))/np.sum(binary_preds_labels_2d==1))

##### 构建连边集合 #####
# 假设edge_index是原始的边索引张量
batch_edges_all = []
batch_edges_attention_all = []
edges_count_list = []
for i in range(len(load_data)):
    cur_batch = load_data[i]
    # batch_edges中的每个元素都是一个(2, num_edges)的张量
    # 其中节点索引已经被转换到0-84的范围内
    batch_edges,  batch_edges_attention= split_edges_and_attention(cur_batch['edge_index'], cur_batch['edge_attention'], nodes_per_graph=85)
    # print(len(batch_edges), len(batch_edges[0]), len(batch_edges[0][0]),len(batch_edges[1][0]))
    # print(len(batch_edges_attention), len(batch_edges_attention[0]), len(batch_edges_attention[1]))
    # 统计每个样本的边数量
    for edges in batch_edges:
        edges_count_list.append(edges[0].shape[0])  # 记录每个样本的边数
    batch_edges_all.extend(batch_edges)
    batch_edges_attention_all.extend(batch_edges_attention)
# print(len(batch_edges_all))
# print(len(batch_edges_attention_all))
edges_count_array = np.array(edges_count_list)
print(f"样本总数: {len(edges_count_list)}")
print(f"边的总数: {np.sum(edges_count_array)}")
print(f"每个样本平均边数: {np.mean(edges_count_array):.2f}")
print(f"最大边数: {np.max(edges_count_array)}")
print(f"最小边数: {np.min(edges_count_array)}")
print(f"边数标准差: {np.std(edges_count_array):.2f}")

# 获取每个样本中attention最高的前5个边及其对应的attention值
top_edges, top_attentions = get_top_k_edges_and_attention(batch_edges_all, batch_edges_attention_all, k=top_edge_K)

# top_edges[i]表示第i个样本中attention最高的k条边
# top_attentions[i]表示第i个样本中最高的k个attention值
# print(len(top_edges), len(top_edges[0]), len(top_edges[0][0]))
# print(len(top_attentions), len(top_attentions[0]))
# print(top_edges[800:801])
# print(top_attentions[800:801])


# ##### 绘制热力图展示Causal Attention #####
plt_idx = 0
for device_id, (start_idx, end_idx) in device_indices.items():
    print('---------------------------------------:', plt_idx)
    plt_idx += 1
    correlation_matrix = np.zeros((85, 85))
    for plt_idx in range(start_idx, end_idx+1):
        print(test_device[plt_idx])
        top_edge_sample = top_edges[plt_idx]
        top_att_sample = top_attentions[plt_idx]
        print('True AD for plot:', np.where(true_labels_2d[plt_idx]==1))
        print('CAL AD for plot:', np.where(binary_preds_labels_2d[plt_idx]==1))
        # 创建一个85x85的零矩阵
        # correlation_matrix = np.zeros((85, 85))
        # 将连边的相关性填充到矩阵中
        for i in range(top_edge_sample.shape[1]):
            node1, node2 = top_edge_sample[0, i], top_edge_sample[1, i]
            correlation_matrix[node1, node2] += top_att_sample[i]
    correlation_matrix = correlation_matrix/(end_idx-start_idx)
    plt.figure(figsize=(12, 10))
    # print(correlation_matrix[71, 5], correlation_matrix[5, 71])
    # 绘制热力图
    sns.heatmap(correlation_matrix, 
                cmap='coolwarm',  # 设置颜色映射
                center=0,         # 设置颜色中心点
                annot=False,      # 不显示具体数值
                square=True,      # 保持正方形
                linewidths=0.5)   # 设置网格线宽度
    # 设置标题
    plt.title(f'Heatmap of Edge Causal Attention Scores for {device_id}')

    # 设置坐标轴标签
    plt.xlabel('Target')
    plt.ylabel('Source')

    # 添加颜色条
    # plt.colorbar(label='相关性系数')
    # 绘制完热力图后
    plt.savefig(f'heatmap/{device_id}_heatmap_{start_idx}_{end_idx}.png', 
                dpi=300,          # 设置分辨率
                bbox_inches='tight',  # 去除图片周围的空白部分
                pad_inches=0.1)    # 设置边距


##### 保存topK的连边索引以及连边attentionn，别保存为两个.npy文件 #####
# save_dir = 'top_edge_attention'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_with_pickle(top_edges, 'top_edge_attention/LLM_model_CausalGIN_multidim_v4.3_cor0.2_top_{}_edges.npy'.format(top_edge_K))
# save_with_pickle(top_attentions, 'top_edge_attention/LLM_model_CausalGIN_multidim_v4.3_cor0.2_top_{}_edge_attentions.npy'.format(top_edge_K))

# #### 检验保存对错 #####
# load1 = load_with_pickle('top_edge_attention/LLM_model_CausalGIN_multidim_v4.3_cor0.2_top_{}_edges.npy'.format(top_edge_K))
# load2 = load_with_pickle('top_edge_attention/LLM_model_CausalGIN_multidim_v4.3_cor0.2_top_{}_edge_attentions.npy'.format(top_edge_K))
# print(len(load1))
# print(len(load2))


# ##### 取出一个true positive的例子可视化[topK edge_attention中包含异常节点的连边] #####
# chosen_idx = 312
# print('Instance {}:'.format(chosen_idx), top_edges[chosen_idx])
# print('Instance {}:'.format(chosen_idx), top_attentions[chosen_idx])
# print('True Label {}:'.format(chosen_idx), true_labels_2d[chosen_idx])
# print('True Label {}:'.format(chosen_idx), np.where(true_labels_2d[chosen_idx]==1))
# true_labels_for_chosen_idx = np.where(true_labels_2d[chosen_idx]==1)
# mask = np.isin(top_edges[chosen_idx], true_labels_for_chosen_idx)
# top_edges_for_chosen_idx = top_edges[chosen_idx][:, np.any(mask, axis=0)]
# print('top_edges_for_chosen_idx:', top_edges_for_chosen_idx)

# data_test = []
# label_test = []
# for batch_idx, data in enumerate(test_loader):
#     batch_info = data.batch
#     node_data = data.x
#     label_test.extend(data.y)
#     # print(data)
#     for i in range(max(batch_info) + 1):  # 100个batch
#         mask = (batch_info == i)
#         batch_data = node_data[mask]
#         data_test.append(batch_data)
# data_test = np.stack(data_test)
# label_test = np.array(label_test)
# print(data_test.shape)
# print(label_test.shape)
# print(label_test[chosen_idx*85:(chosen_idx+1)*85])

# plot_connected_timeseries(top_edges_for_chosen_idx, data_test, chosen_idx)

#### 计算node attention的异常覆盖率/edge attention的异常覆盖率/node+edge attention的异常覆盖率 #####
node_ratio = calculate_total_match_ratio(indices_max20, true_labels_2d)
print(f"Node Attention总体异常覆盖比例: {node_ratio:.4f}")

edge_ratio, unique_nodes_array = calculate_total_edge_match_ratio(top_edges, true_labels_2d)
stats = {
        'mean_nodes': np.mean(unique_nodes_array),
        'max_nodes': np.max(unique_nodes_array),
        'min_nodes': np.min(unique_nodes_array),
        'std_nodes': np.std(unique_nodes_array),
        'match_ratio': edge_ratio
    }

print(f"Edge Attention总体异常覆盖比例: {edge_ratio:.4f}")
print(stats)

total_ratio, total_unique_nodes_array = calculate_combined_edge_match_ratio(top_edges, true_labels_2d, binary_preds_labels_2d)
stats_combined = {
        'total_mean_nodes': np.mean(total_unique_nodes_array),
        'total_max_nodes': np.max(total_unique_nodes_array),
        'total_min_nodes': np.min(total_unique_nodes_array),
        'total_std_nodes': np.std(total_unique_nodes_array),
        'total_match_ratio': total_ratio
    }

print(f"CAL检测异常节点与Edge Attention总体异常覆盖比例: {total_ratio:.4f}")
print(stats_combined)