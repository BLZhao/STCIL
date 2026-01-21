from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import random
import pdb

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1])
        return x_hat - bias


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: [batch*num_vars, seq_len]
        # 需要增加channel维度用于1D卷积
        x = x.unsqueeze(1)  # [batch*num_vars, 1, seq_len]
        
        # padding
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        
        x = self.avg(x)  # 进行移动平均
        x = x.squeeze(1)  # 去掉channel维度，恢复原始形状
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x shape: [batch*num_vars, seq_len]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean
    
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf, dim=-1)
        x_trend = x - x_season
        return x_season, x_trend




class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_len, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        
        # 创建上采样层
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len // (2 ** (i + 1)), input_len // (2 ** i)),
                nn.GELU(),
                nn.Linear(input_len // (2 ** i), input_len // (2 ** i))
            ) for i in reversed(range(num_scales-1))
        ])
            
    def downsample(self, x):
        """多尺度下采样"""
        scale_list = []
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 原始序列
        scale_list.append(x)
        
        # 生成多尺度序列
        for i in range(1, self.num_scales):
            kernel_size = 2 ** i
            # 计算保留多少个时间点
            valid_len = (seq_len // kernel_size) * kernel_size
            # 从末尾截取
            valid_x = x[:, -valid_len:]
            # 使用平均池化进行下采样
            down_sampled = F.avg_pool1d(
                valid_x, 
                kernel_size=kernel_size, 
                stride=kernel_size
            )
            scale_list.append(down_sampled)
            
        return scale_list
        
    def forward(self, x):
        """
        输入: x shape (batch_size*num_variables, seq_len)
        输出: 处理后的时间序列 shape (batch_size*num_variables, seq_len)
        """
        # 1. 下采样获取多尺度特征
        scale_list = self.downsample(x)
        
        # 2. 自顶向下融合特征
        scale_list_reverse = scale_list.copy()
        scale_list_reverse.reverse()
        
        out_low = scale_list_reverse[0]
        out_high = scale_list_reverse[1]
        out_list = [out_low]
        
        # 逐层融合
        for i in range(len(scale_list_reverse) - 1):
            # 上采样低尺度特征
            out_high_res = self.up_sampling_layers[i](out_low)
            # 残差连接
            out_high = out_high + out_high_res
            out_low = out_high
            
            # 如果还有更高尺度，则获取下一层
            if i + 2 < len(scale_list_reverse):
                out_high = scale_list_reverse[i + 2]
            out_list.append(out_low)
            
        out_list.reverse()
            
        return out_list
    

class MultiScaleFeatureExtractor_eq(nn.Module):
    def __init__(self, input_len, hidden, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        
        # 创建上采样层
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len // (2 ** (i)), hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden)
            ) for i in range(num_scales)
        ])
            
    def downsample(self, x):
        """多尺度下采样"""
        scale_list = []
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 原始序列
        scale_list.append(x)
        
        # 生成多尺度序列
        for i in range(1, self.num_scales):
            kernel_size = 2 ** i
            # 计算保留多少个时间点
            valid_len = (seq_len // kernel_size) * kernel_size
            # 从末尾截取
            valid_x = x[:, -valid_len:]
            # 使用平均池化进行下采样
            # down_sampled = F.avg_pool1d(
            #     valid_x, 
            #     kernel_size=kernel_size, 
            #     stride=kernel_size
            # )
            down_sampled = valid_x[:, kernel_size-1::kernel_size]
            scale_list.append(down_sampled)
            
        return scale_list
        
    def forward(self, x):
        """
        输入: x shape (batch_size*num_variables, seq_len)
        输出: 处理后的时间序列 shape (batch_size*num_variables, seq_len)
        """
        # 1. 下采样获取多尺度特征
        scale_list = self.downsample(x)
        
        # 2. 自顶向下融合特征
        
        out_org = scale_list[0]
        out = self.up_sampling_layers[0](out_org)
        
        # 逐层融合
        for i in range(1, len(scale_list)):
            # 上采样低尺度特征
            out_org = scale_list[i]
            out += self.up_sampling_layers[i](out_org)
            
        return out
    
class MultiScaleFeatureExtractor_last(nn.Module):
    def __init__(self, input_len, hidden, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        
        # 创建上采样层
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len // (2 ** (i)), hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden)
            ) for i in range(num_scales)
        ])
            
    def downsample(self, x):
        """多尺度下采样"""
        scale_list = []
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 原始序列
        scale_list.append(x)
        
        # 生成多尺度序列
        for i in range(1, self.num_scales):
            kernel_size = 2 ** i
            # 计算保留多少个时间点
            valid_len = seq_len // kernel_size
            # 从末尾截取
            valid_x = x[:, -valid_len:]
            # 使用平均池化进行下采样
            # down_sampled = F.avg_pool1d(
            #     valid_x, 
            #     kernel_size=kernel_size, 
            #     stride=kernel_size
            # )
            down_sampled = valid_x
            scale_list.append(down_sampled)
            
        return scale_list
        
    def forward(self, x):
        """
        输入: x shape (batch_size*num_variables, seq_len)
        输出: 处理后的时间序列 shape (batch_size*num_variables, seq_len)
        """
        # 1. 下采样获取多尺度特征
        scale_list = self.downsample(x)
        
        # 2. 自顶向下融合特征
        
        out_org = scale_list[0]
        out = self.up_sampling_layers[0](out_org)
        
        # 逐层融合
        for i in range(1, len(scale_list)):
            # 上采样低尺度特征
            out_org = scale_list[i]
            out += self.up_sampling_layers[i](out_org)
            
        return out






class CausalGCN_best_1(torch.nn.Module):
    """GCN with BN and residual connection. ------仅仅在最后对表征做一次node attentionn"""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_best_1, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        cor = data.edge_attr
        
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, torch.abs(cor.squeeze(1))))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index, torch.abs(cor.squeeze(1))))
        
        # edge_rep = torch.cat([x[row], x[col]], dim=-1)

        # if self.without_edge_attention:
        #     edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        # else:
        #     edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        # edge_weight_c = edge_att[:, 0] * torch.abs(cor.squeeze(1))
        # edge_weight_o = edge_att[:, 1] * torch.abs(cor.squeeze(1))

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        # xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        # xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        xco_logis, xoc_logis = self.random_readout_layer(xc, xo, n_times=100)

        return xc_logis, xo_logis, xco_logis, xoc_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, eval_random):

    #     num = xc.shape[0]
    #     l = [i for i in range(num)]
    #     if self.with_random:
    #         if eval_random:
    #             random.shuffle(l)
    #     random_idx = torch.tensor(l)
    #     if self.args.cat_or_add == "cat":
    #         x = torch.cat((xc[random_idx], xo), dim=1)
    #     else:
    #         x = xc[random_idx] + xo

    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
    #     return x_logis

    def random_readout_layer(self, xc, xo, n_times=100):
        num = xc.shape[0]
        xc_results = []
        xo_results = []
        
        # 进行100次随机匹配
        for _ in range(n_times):
            # 使用torch.randperm替代random.shuffle
            random_idx = torch.randperm(num)
            
            if self.args.cat_or_add == "cat":
                x = torch.cat((xc[random_idx], xo), dim=1)
            else:
                x = xc[random_idx] + xo
                
            x = self.fc1_bn_co(x)
            x = self.fc1_co(x)
            x = F.relu(x)
            x = self.fc2_bn_co(x)
            x = self.fc2_co(x)
            x_logis = F.log_softmax(x, dim=-1)
            
            # 存储每次的结果
            xc_results.append(x_logis)
            xo_results.append(x_logis[torch.argsort(random_idx)])
        
        # 计算平均值
        xc_mean = torch.stack(xc_results).mean(dim=0)  # 对xc的所有匹配结果取平均
        xo_mean = torch.stack(xo_results).mean(dim=0)  # 对xo的所有匹配结果取平均
        
        return xc_mean, xo_mean

    
class CausalGCN_CAL_M(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_CAL_M, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        self.num_vars = args.node_num
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         # ReLU(),
        #                         # Linear(hidden, hidden),
        #                         ReLU())) # GIN
        self.bns_conv = torch.nn.ModuleList()

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)  # 输出维度变为 2*hidden
        )
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        # 生成每个feature维度的node attention权重
        node_att = self.node_att_mlp(x)  # [num_nodes, 2*hidden]
        node_att = node_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        node_att = F.softmax(node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        node_att_c = node_att[:, 0, :]  # [num_nodes, hidden_in]
        node_att_o = node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用node attention
        xc = node_att_c * x 
        xo = node_att_o * x

        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch) # 单维
        # xo = self.global_pool(xo, batch) # 单维
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis, xoc_logis = self.random_readout_layer(xc, xo, n_times=100)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, xoc_logis, attention_dict

        return xc_logis, xo_logis, xco_logis, xoc_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     num = xc.shape[0]
    #     xc_results = []
    #     xo_results = []
        
    #     # 进行100次随机匹配
    #     for _ in range(n_times):
    #         # 使用torch.randperm替代random.shuffle
    #         random_idx = torch.randperm(num)
            
    #         if self.args.cat_or_add == "cat":
    #             x = torch.cat((xc[random_idx], xo), dim=1)
    #         else:
    #             x = xc[random_idx] + xo
                
    #         x = self.fc1_bn_co(x)
    #         x = self.fc1_co(x)
    #         x = F.relu(x)
    #         x = self.fc2_bn_co(x)
    #         x = self.fc2_co(x)
    #         x_logis = F.log_softmax(x, dim=-1)
            
    #         # 存储每次的结果
    #         xc_results.append(x_logis)
    #         xo_results.append(x_logis[torch.argsort(random_idx)])
        
    #     # 计算平均值
    #     xc_mean = torch.stack(xc_results).mean(dim=0)  # 对xc的所有匹配结果取平均
    #     xo_mean = torch.stack(xo_results).mean(dim=0)  # 对xo的所有匹配结果取平均
        
    #     return xc_mean, xo_mean

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     batch_size = xc.shape[0] // self.num_vars
    #     num_vars = self.num_vars
        
    #     # 重塑输入tensor
    #     xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
    #     xo = xo.view(batch_size, num_vars, -1)
        
    #     if self.args.cat_or_add == "cat":
    #         # 准备所有可能的组合
    #         xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
    #         xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
    #         # 直接进行所有组合的拼接
    #         x = torch.cat((
    #             xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
    #             xo_expanded.expand(batch_size, -1, -1, -1)
    #         ), dim=-1)
    #     else:
    #         # 直接进行所有组合的相加
    #         x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
    #     # 重塑以便于通过网络处理
    #     x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
    #     # 通过网络层
    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
        
    #     # 重塑回原始形状并计算平均值
    #     x_logis = x_logis.view(batch_size, batch_size, num_vars, -1)
    #     xc_mean = x_logis.mean(dim=1)  # [batch_size, num_vars, output_dim]
    #     xo_mean = x_logis.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
    #     # 重塑回最终所需形状
    #     xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
    #     xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
    #     return xo_mean, xc_mean
    
    def random_readout_layer(self, xc, xo, n_times=100):
        batch_size = xc.shape[0] // self.num_vars
        num_vars = self.num_vars
        
        # 重塑输入tensor
        xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
        xo = xo.view(batch_size, num_vars, -1)
        
        if self.args.cat_or_add == "cat":
            # 准备所有可能的组合
            xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
            xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
            # 直接进行所有组合的拼接
            x = torch.cat((
                xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
                xo_expanded.expand(batch_size, -1, -1, -1)
            ), dim=-1)
        else:
            # 直接进行所有组合的相加
            x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
        # 重塑以便于通过网络处理
        x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
        # 通过网络层
        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        
        # 先进行softmax
        x_soft = F.softmax(x, dim=-1)
        
        # 重塑形状
        x_soft = x_soft.view(batch_size, batch_size, num_vars, -1)
        
        # 计算平均值
        xc_mean = x_soft.mean(dim=1)  # [batch_size, num_vars, output_dim]
        xo_mean = x_soft.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
        # 重塑回最终所需形状
        xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
        xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
        # 最后进行log运算
        xc_mean = torch.log(xc_mean + 1e-10)  # 添加小量防止log(0)
        xo_mean = torch.log(xo_mean + 1e-10)
        
        return xo_mean, xc_mean
    

class CausalGCN_CAL_M_Multiscale(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_CAL_M_Multiscale, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        self.num_vars = args.node_num
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        # self.multiscale_layer = MultiScaleFeatureExtractor(
        #     input_len=hidden_in,
        #     num_scales=4
        #     )
        self.multiscale_layer = MultiScaleFeatureExtractor_last(
            input_len=hidden_in,
            hidden=hidden,
            num_scales=4
            )
        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=hidden_in, out_features=hidden),
            nn.GELU(),
            nn.Linear(in_features=hidden, out_features=hidden),
            )
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         # ReLU(),
        #                         # Linear(hidden, hidden),
        #                         ReLU())) # GIN
        self.bns_conv = torch.nn.ModuleList()

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)  # 输出维度变为 2*hidden
        )
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        ##### 原multiscale代码
        # x_list = self.multiscale_layer(x)
        # x = x_list[0]
        # x = self.cross_layer(x)
        # # x = F.relu(self.conv_feat(x, edge_index))
        ##### 修改multiscale代码
        x = self.multiscale_layer(x)
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        # 生成每个feature维度的node attention权重
        node_att = self.node_att_mlp(x)  # [num_nodes, 2*hidden]
        node_att = node_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        node_att = F.softmax(node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        node_att_c = node_att[:, 0, :]  # [num_nodes, hidden_in]
        node_att_o = node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用node attention
        xc = node_att_c * x 
        xo = node_att_o * x

        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # print('shape of xc:', xc.shape)
        # print('shape of batch:', batch)

        # xc = self.global_pool(xc, batch) # 单维
        # xo = self.global_pool(xo, batch) # 单维
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis, xoc_logis = self.random_readout_layer(xc, xo, n_times=100)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, xoc_logis, attention_dict

        return xc_logis, xo_logis, xco_logis, xoc_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     num = xc.shape[0]
    #     xc_results = []
    #     xo_results = []
        
    #     # 进行100次随机匹配
    #     for _ in range(n_times):
    #         # 使用torch.randperm替代random.shuffle
    #         random_idx = torch.randperm(num)
            
    #         if self.args.cat_or_add == "cat":
    #             x = torch.cat((xc[random_idx], xo), dim=1)
    #         else:
    #             x = xc[random_idx] + xo
                
    #         x = self.fc1_bn_co(x)
    #         x = self.fc1_co(x)
    #         x = F.relu(x)
    #         x = self.fc2_bn_co(x)
    #         x = self.fc2_co(x)
    #         x_logis = F.log_softmax(x, dim=-1)
            
    #         # 存储每次的结果
    #         xc_results.append(x_logis)
    #         xo_results.append(x_logis[torch.argsort(random_idx)])
        
    #     # 计算平均值
    #     xc_mean = torch.stack(xc_results).mean(dim=0)  # 对xc的所有匹配结果取平均
    #     xo_mean = torch.stack(xo_results).mean(dim=0)  # 对xo的所有匹配结果取平均
        
    #     return xc_mean, xo_mean

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     batch_size = xc.shape[0] // self.num_vars
    #     num_vars = self.num_vars
        
    #     # 重塑输入tensor
    #     xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
    #     xo = xo.view(batch_size, num_vars, -1)
        
    #     if self.args.cat_or_add == "cat":
    #         # 准备所有可能的组合
    #         xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
    #         xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
    #         # 直接进行所有组合的拼接
    #         x = torch.cat((
    #             xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
    #             xo_expanded.expand(batch_size, -1, -1, -1)
    #         ), dim=-1)
    #     else:
    #         # 直接进行所有组合的相加
    #         x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
    #     # 重塑以便于通过网络处理
    #     x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
    #     # 通过网络层
    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
        
    #     # 重塑回原始形状并计算平均值
    #     x_logis = x_logis.view(batch_size, batch_size, num_vars, -1)
    #     xc_mean = x_logis.mean(dim=1)  # [batch_size, num_vars, output_dim]
    #     xo_mean = x_logis.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
    #     # 重塑回最终所需形状
    #     xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
    #     xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
    #     return xo_mean, xc_mean
    
    def random_readout_layer(self, xc, xo, n_times=100):
        batch_size = xc.shape[0] // self.num_vars
        num_vars = self.num_vars
        
        # 重塑输入tensor
        xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
        xo = xo.view(batch_size, num_vars, -1)
        
        if self.args.cat_or_add == "cat":
            # 准备所有可能的组合
            xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
            xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
            # 直接进行所有组合的拼接
            x = torch.cat((
                xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
                xo_expanded.expand(batch_size, -1, -1, -1)
            ), dim=-1)
        else:
            # 直接进行所有组合的相加
            x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
        # 重塑以便于通过网络处理
        x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
        # 通过网络层
        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        
        # 先进行softmax
        x_soft = F.softmax(x, dim=-1)
        
        # 重塑形状
        x_soft = x_soft.view(batch_size, batch_size, num_vars, -1)
        
        # 计算平均值
        xc_mean = x_soft.mean(dim=1)  # [batch_size, num_vars, output_dim]
        xo_mean = x_soft.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
        # 重塑回最终所需形状
        xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
        xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
        # 最后进行log运算
        xc_mean = torch.log(xc_mean + 1e-10) 
        xo_mean = torch.log(xo_mean + 1e-10)
        
        return xo_mean, xc_mean



    

class CausalGCN_CAL(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_CAL, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        self.num_vars = args.node_num
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         # ReLU(),
        #                         # Linear(hidden, hidden),
        #                         ReLU())) # GIN
        self.bns_conv = torch.nn.ModuleList()

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)  # 输出维度变为 2*hidden
        )
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        # 生成每个feature维度的node attention权重
        node_att = self.node_att_mlp(x)  # [num_nodes, 2*hidden]
        node_att = node_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        node_att = F.softmax(node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        node_att_c = node_att[:, 0, :]  # [num_nodes, hidden_in]
        node_att_o = node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用node attention
        xc = node_att_c * x 
        xo = node_att_o * x

        ##### 单维node attention
        # node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # xc = node_att[:, 0].view(-1, 1) * x
        # xo = node_att[:, 1].view(-1, 1) * x

        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch) # 单维
        # xo = self.global_pool(xo, batch) # 单维
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, eval_random):

    #     num = xc.shape[0]
    #     l = [i for i in range(num)]
    #     if self.with_random:
    #         if eval_random:
    #             random.shuffle(l)
    #     random_idx = torch.tensor(l)
    #     if self.args.cat_or_add == "cat":
    #         x = torch.cat((xc[random_idx], xo), dim=1)
    #     else:
    #         x = xc[random_idx] + xo

    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
    #     return x_logis

    def random_readout_layer(self, xc, xo, eval_random):
        batch_size = xc.shape[0] // self.num_vars  # 计算实际的batch size
        num_vars = self.num_vars  # 每个样本的变量个数
        
        # 重塑输入tensor以便于按样本操作
        xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
        xo = xo.view(batch_size, num_vars, -1)

        # if self.with_random and eval_random:
        # 生成随机排列的样本索引
        l = [i for i in range(batch_size)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        # 按样本随机重排
        xc = xc[random_idx]

        # 重塑回原始维度
        xc = xc.view(-1, xc.shape[-1])  # [batch_size*num_vars, feature_dim]
        xo = xo.view(-1, xo.shape[-1])

        if self.args.cat_or_add == "cat":
            x = torch.cat((xc, xo), dim=1)
        else:
            x = xc + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        
        return x_logis
    

class CausalGCN_CAL_Multiscale(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_CAL_Multiscale, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        self.num_vars = args.node_num
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.multiscale_layer = MultiScaleFeatureExtractor_last(
            input_len=hidden_in,
            hidden=hidden,
            num_scales=4
            )
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         # ReLU(),
        #                         # Linear(hidden, hidden),
        #                         ReLU())) # GIN
        self.bns_conv = torch.nn.ModuleList()

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)  # 输出维度变为 2*hidden
        )
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        # x = F.relu(self.conv_feat(x, edge_index))
        x = self.multiscale_layer(x)
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        # 生成每个feature维度的node attention权重
        node_att = self.node_att_mlp(x)  # [num_nodes, 2*hidden]
        node_att = node_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        node_att = F.softmax(node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        node_att_c = node_att[:, 0, :]  # [num_nodes, hidden_in]
        node_att_o = node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用node attention
        xc = node_att_c * x 
        xo = node_att_o * x

        ##### 单维node attention
        # node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # xc = node_att[:, 0].view(-1, 1) * x
        # xo = node_att[:, 1].view(-1, 1) * x

        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch) # 单维
        # xo = self.global_pool(xo, batch) # 单维
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, eval_random):

    #     num = xc.shape[0]
    #     l = [i for i in range(num)]
    #     if self.with_random:
    #         if eval_random:
    #             random.shuffle(l)
    #     random_idx = torch.tensor(l)
    #     if self.args.cat_or_add == "cat":
    #         x = torch.cat((xc[random_idx], xo), dim=1)
    #     else:
    #         x = xc[random_idx] + xo

    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
    #     return x_logis

    def random_readout_layer(self, xc, xo, eval_random):
        batch_size = xc.shape[0] // self.num_vars  # 计算实际的batch size
        num_vars = self.num_vars  # 每个样本的变量个数
        
        # 重塑输入tensor以便于按样本操作
        xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
        xo = xo.view(batch_size, num_vars, -1)

        # if self.with_random and eval_random:
        # 生成随机排列的样本索引
        l = [i for i in range(batch_size)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        # 按样本随机重排
        xc = xc[random_idx]

        # 重塑回原始维度
        xc = xc.view(-1, xc.shape[-1])  # [batch_size*num_vars, feature_dim]
        xo = xo.view(-1, xo.shape[-1])

        if self.args.cat_or_add == "cat":
            x = torch.cat((xc, xo), dim=1)
        else:
            x = xc + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        
        return x_logis



class CausalGCN_CAL_M_D(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_CAL_M_D, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        self.hidden = hidden
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.node_num = args.node_num
        self.num_vars = args.node_num
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         ReLU(),
        #                         Linear(hidden, hidden),
        #                         ReLU())) # GIN

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        # 将原来的node_att_mlp修改为:
        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)  # 输出维度变为 2*hidden
        )
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        ##### 新增
        # self.context_convs_season = GConv(hidden, hidden)
        # self.objects_convs_season = GConv(hidden, hidden)
        # self.context_convs_trend = GConv(hidden, hidden)
        # self.objects_convs_trend = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
        
        self.decompsition = DFT_series_decomp(5)
        self.cross_layer = nn.Sequential(
                nn.Linear(in_features=hidden_in, out_features=hidden),
                nn.GELU(),
                nn.Linear(in_features=hidden, out_features=hidden),
            )

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index

        T = x.shape[1] 
        Fe = self.node_num
        B = x.shape[0]//Fe 
        x = x.view(B, Fe, T)

        # x = self.bn_feat(x)
        # x = F.relu(self.conv_feat(x, edge_index))
        season, trend = self.decompsition(x)
        season = self.cross_layer(season)
        trend = self.cross_layer(trend)
        season = season.view(B * Fe, self.hidden)
        trend = trend.view(B * Fe, self.hidden)
        # x = x.view(B * Fe, T)

        season_node_att = self.node_att_mlp(season)  # [num_nodes, 2*hidden_in]
        season_node_att = season_node_att.view(season.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        season_node_att = F.softmax(season_node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        season_node_att_c = season_node_att[:, 0, :]  # [num_nodes, hidden_in]
        season_node_att_o = season_node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用attention
        season_xc = season_node_att_c * season  # 元素级别的乘法
        season_xo = season_node_att_o * season

        trend_node_att = self.node_att_mlp(trend)  # [num_nodes, 2*hidden_in]
        trend_node_att = trend_node_att.view(trend.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        trend_node_att = F.softmax(trend_node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        trend_node_att_c = trend_node_att[:, 0, :]  # [num_nodes, hidden_in]
        trend_node_att_o = trend_node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用attention
        trend_xc = trend_node_att_c * trend  # 元素级别的乘法
        trend_xo = trend_node_att_o * trend

        ##### 原版
        xc = trend_xc + season_xc
        xo = trend_xo + season_xo

        x = xc + xo
        #####

        # x = season + trend
        # x = x.view(B * Fe, 64)
        ##### 原版
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))
        ##### 修正
        # edge_rep_season = torch.cat([season[row], season[col]], dim=-1)
        # edge_rep_trend = torch.cat([trend[row], trend[col]], dim=-1)
        # edge_att_season = F.softmax(self.edge_att_mlp(edge_rep_season), dim=-1)
        # edge_att_trend = F.softmax(self.edge_att_mlp(edge_rep_trend), dim=-1)

        # season_c = F.relu(self.context_convs_season(self.bnc(season_xc), edge_index, edge_att_season[:, 0]))
        # season_o = F.relu(self.objects_convs_season(self.bnc(season_xo), edge_index, edge_att_season[:, 1]))
        # trend_c = F.relu(self.context_convs_trend(self.bnc(trend_xc), edge_index, edge_att_trend[:, 0]))
        # trend_o = F.relu(self.objects_convs_trend(self.bnc(trend_xo), edge_index, edge_att_trend[:, 1]))

        # xc = season_c + trend_c
        # xo = season_o + trend_o

        

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis, xoc_logis = self.random_readout_layer(xc, xo, n_times=100)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'season_node_attention': season_node_att.detach().cpu(),
                'trend_node_attention': trend_node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, xoc_logis, attention_dict

        return xc_logis, xo_logis, xco_logis, xoc_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     num = xc.shape[0]
    #     xc_results = []
    #     xo_results = []
        
    #     # 进行100次随机匹配
    #     for _ in range(n_times):
    #         # 使用torch.randperm替代random.shuffle
    #         random_idx = torch.randperm(num)
            
    #         if self.args.cat_or_add == "cat":
    #             x = torch.cat((xc[random_idx], xo), dim=1)
    #         else:
    #             x = xc[random_idx] + xo
                
    #         x = self.fc1_bn_co(x)
    #         x = self.fc1_co(x)
    #         x = F.relu(x)
    #         x = self.fc2_bn_co(x)
    #         x = self.fc2_co(x)
    #         x_logis = F.log_softmax(x, dim=-1)
            
    #         # 存储每次的结果
    #         xc_results.append(x_logis)
    #         xo_results.append(x_logis[torch.argsort(random_idx)])
        
    #     # 计算平均值
    #     xc_mean = torch.stack(xc_results).mean(dim=0)  # 对xc的所有匹配结果取平均
    #     xo_mean = torch.stack(xo_results).mean(dim=0)  # 对xo的所有匹配结果取平均
        
    #     return xc_mean, xo_mean

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     batch_size = xc.shape[0] // self.num_vars
    #     num_vars = self.num_vars
        
    #     # 重塑输入tensor
    #     xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
    #     xo = xo.view(batch_size, num_vars, -1)
        
    #     if self.args.cat_or_add == "cat":
    #         # 准备所有可能的组合
    #         xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
    #         xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
    #         # 直接进行所有组合的拼接
    #         x = torch.cat((
    #             xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
    #             xo_expanded.expand(batch_size, -1, -1, -1)
    #         ), dim=-1)
    #     else:
    #         # 直接进行所有组合的相加
    #         x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
    #     # 重塑以便于通过网络处理
    #     x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
    #     # 通过网络层
    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
        
    #     # 重塑回原始形状并计算平均值
    #     x_logis = x_logis.view(batch_size, batch_size, num_vars, -1)
    #     xc_mean = x_logis.mean(dim=1)  # [batch_size, num_vars, output_dim]
    #     xo_mean = x_logis.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
    #     # 重塑回最终所需形状
    #     xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
    #     xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
    #     return xo_mean, xc_mean

    def random_readout_layer(self, xc, xo, n_times=100):
        batch_size = xc.shape[0] // self.num_vars
        num_vars = self.num_vars
        
        # 重塑输入tensor
        xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
        xo = xo.view(batch_size, num_vars, -1)
        
        if self.args.cat_or_add == "cat":
            # 准备所有可能的组合
            xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
            xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
            # 直接进行所有组合的拼接
            x = torch.cat((
                xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
                xo_expanded.expand(batch_size, -1, -1, -1)
            ), dim=-1)
        else:
            # 直接进行所有组合的相加
            x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
        # 重塑以便于通过网络处理
        x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
        # 通过网络层
        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        
        # 先进行softmax
        x_soft = F.softmax(x, dim=-1)
        
        # 重塑形状
        x_soft = x_soft.view(batch_size, batch_size, num_vars, -1)
        
        # 计算平均值
        xc_mean = x_soft.mean(dim=1)  # [batch_size, num_vars, output_dim]
        xo_mean = x_soft.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
        # 重塑回最终所需形状
        xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
        xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
        # 最后进行log运算
        xc_mean = torch.log(xc_mean + 1e-10)  # 添加小量防止log(0)
        xo_mean = torch.log(xo_mean + 1e-10)
        
        return xo_mean, xc_mean

class CausalGCN_CAL_M_D_wo_season(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN_CAL_M_D_wo_season, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        self.hidden = hidden
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.node_num = args.node_num
        self.num_vars = args.node_num
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         ReLU(),
        #                         Linear(hidden, hidden),
        #                         ReLU())) # GIN

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        # 将原来的node_att_mlp修改为:
        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)  # 输出维度变为 2*hidden
        )
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        ##### 新增
        # self.context_convs_season = GConv(hidden, hidden)
        # self.objects_convs_season = GConv(hidden, hidden)
        # self.context_convs_trend = GConv(hidden, hidden)
        # self.objects_convs_trend = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
        
        self.decompsition = DFT_series_decomp(5)
        self.cross_layer = nn.Sequential(
                nn.Linear(in_features=hidden_in, out_features=hidden),
                nn.GELU(),
                nn.Linear(in_features=hidden, out_features=hidden),
            )

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index

        # x = self.bn_feat(x) ## 之前算法未加

        T = x.shape[1] 
        Fe = self.node_num
        B = x.shape[0]//Fe 
        x = x.view(B, Fe, T)

       
        # x = F.relu(self.conv_feat(x, edge_index))
        season, trend = self.decompsition(x)
        season = self.cross_layer(season)
        trend = self.cross_layer(trend)
        season = season.view(B * Fe, self.hidden)
        trend = trend.view(B * Fe, self.hidden)
        x = season + trend

        node_att = self.node_att_mlp(x)  # [num_nodes, 2*hidden]
        node_att = node_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        node_att = F.softmax(node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        node_att_c = node_att[:, 0, :]  # [num_nodes, hidden_in]
        node_att_o = node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用node attention
        xc = node_att_c * x 
        xo = node_att_o * x

        # season_node_att = self.node_att_mlp(season)  # [num_nodes, 2*hidden_in]
        # season_node_att = season_node_att.view(season.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        # season_node_att = F.softmax(season_node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        # season_node_att_c = season_node_att[:, 0, :]  # [num_nodes, hidden_in]
        # season_node_att_o = season_node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用attention
        # season_xc = season_node_att_c * season  # 元素级别的乘法
        # season_xo = season_node_att_o * season

        # trend_node_att = self.node_att_mlp(trend)  # [num_nodes, 2*hidden_in]
        # trend_node_att = trend_node_att.view(trend.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        # trend_node_att = F.softmax(trend_node_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        # trend_node_att_c = trend_node_att[:, 0, :]  # [num_nodes, hidden_in]
        # trend_node_att_o = trend_node_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用attention
        # trend_xc = trend_node_att_c * trend  # 元素级别的乘法
        # trend_xo = trend_node_att_o * trend

        ##### 原版
        # xc = trend_xc
        # xo = trend_xo

        # x = xc + xo
        #####

        # x = season + trend
        # x = x.view(B * Fe, 64)
        ##### 原版
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))
        ##### 修正
        # edge_rep_season = torch.cat([season[row], season[col]], dim=-1)
        # edge_rep_trend = torch.cat([trend[row], trend[col]], dim=-1)
        # edge_att_season = F.softmax(self.edge_att_mlp(edge_rep_season), dim=-1)
        # edge_att_trend = F.softmax(self.edge_att_mlp(edge_rep_trend), dim=-1)

        # season_c = F.relu(self.context_convs_season(self.bnc(season_xc), edge_index, edge_att_season[:, 0]))
        # season_o = F.relu(self.objects_convs_season(self.bnc(season_xo), edge_index, edge_att_season[:, 1]))
        # trend_c = F.relu(self.context_convs_trend(self.bnc(trend_xc), edge_index, edge_att_trend[:, 0]))
        # trend_o = F.relu(self.objects_convs_trend(self.bnc(trend_xo), edge_index, edge_att_trend[:, 1]))

        # xc = season_c + trend_c
        # xo = season_o + trend_o

        

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis, xoc_logis = self.random_readout_layer(xc, xo, n_times=100)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, xoc_logis, attention_dict

        return xc_logis, xo_logis, xco_logis, xoc_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     num = xc.shape[0]
    #     xc_results = []
    #     xo_results = []
        
    #     # 进行100次随机匹配
    #     for _ in range(n_times):
    #         # 使用torch.randperm替代random.shuffle
    #         random_idx = torch.randperm(num)
            
    #         if self.args.cat_or_add == "cat":
    #             x = torch.cat((xc[random_idx], xo), dim=1)
    #         else:
    #             x = xc[random_idx] + xo
                
    #         x = self.fc1_bn_co(x)
    #         x = self.fc1_co(x)
    #         x = F.relu(x)
    #         x = self.fc2_bn_co(x)
    #         x = self.fc2_co(x)
    #         x_logis = F.log_softmax(x, dim=-1)
            
    #         # 存储每次的结果
    #         xc_results.append(x_logis)
    #         xo_results.append(x_logis[torch.argsort(random_idx)])
        
    #     # 计算平均值
    #     xc_mean = torch.stack(xc_results).mean(dim=0)  # 对xc的所有匹配结果取平均
    #     xo_mean = torch.stack(xo_results).mean(dim=0)  # 对xo的所有匹配结果取平均
        
    #     return xc_mean, xo_mean

    # def random_readout_layer(self, xc, xo, n_times=100):
    #     batch_size = xc.shape[0] // self.num_vars
    #     num_vars = self.num_vars
        
    #     # 重塑输入tensor
    #     xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
    #     xo = xo.view(batch_size, num_vars, -1)
        
    #     if self.args.cat_or_add == "cat":
    #         # 准备所有可能的组合
    #         xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
    #         xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
    #         # 直接进行所有组合的拼接
    #         x = torch.cat((
    #             xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
    #             xo_expanded.expand(batch_size, -1, -1, -1)
    #         ), dim=-1)
    #     else:
    #         # 直接进行所有组合的相加
    #         x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
    #     # 重塑以便于通过网络处理
    #     x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
    #     # 通过网络层
    #     x = self.fc1_bn_co(x)
    #     x = self.fc1_co(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_co(x)
    #     x = self.fc2_co(x)
    #     x_logis = F.log_softmax(x, dim=-1)
        
    #     # 重塑回原始形状并计算平均值
    #     x_logis = x_logis.view(batch_size, batch_size, num_vars, -1)
    #     xc_mean = x_logis.mean(dim=1)  # [batch_size, num_vars, output_dim]
    #     xo_mean = x_logis.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
    #     # 重塑回最终所需形状
    #     xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
    #     xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
    #     return xo_mean, xc_mean

    def random_readout_layer(self, xc, xo, n_times=100):
        batch_size = xc.shape[0] // self.num_vars
        num_vars = self.num_vars
        
        # 重塑输入tensor
        xc = xc.view(batch_size, num_vars, -1)  # [batch_size, num_vars, feature_dim]
        xo = xo.view(batch_size, num_vars, -1)
        
        if self.args.cat_or_add == "cat":
            # 准备所有可能的组合
            xc_expanded = xc.unsqueeze(1)  # [batch_size, 1, num_vars, feature_dim]
            xo_expanded = xo.unsqueeze(0)  # [1, batch_size, num_vars, feature_dim]
            
            # 直接进行所有组合的拼接
            x = torch.cat((
                xc_expanded.expand(-1, batch_size, -1, -1),  # [batch_size, batch_size, num_vars, feature_dim]
                xo_expanded.expand(batch_size, -1, -1, -1)
            ), dim=-1)
        else:
            # 直接进行所有组合的相加
            x = xc.unsqueeze(1) + xo.unsqueeze(0)  # [batch_size, batch_size, num_vars, feature_dim]
        
        # 重塑以便于通过网络处理
        x = x.view(-1, x.shape[-1])  # [batch_size*batch_size*num_vars, feature_dim]
        
        # 通过网络层
        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        
        # 先进行softmax
        x_soft = F.softmax(x, dim=-1)
        
        # 重塑形状
        x_soft = x_soft.view(batch_size, batch_size, num_vars, -1)
        
        # 计算平均值
        xc_mean = x_soft.mean(dim=1)  # [batch_size, num_vars, output_dim]
        xo_mean = x_soft.mean(dim=0)  # [batch_size, num_vars, output_dim]
        
        # 重塑回最终所需形状
        xc_mean = xc_mean.reshape(-1, xc_mean.shape[-1])  # [batch_size*num_vars, output_dim]
        xo_mean = xo_mean.reshape(-1, xo_mean.shape[-1])
        
        # 最后进行log运算
        xc_mean = torch.log(xc_mean + 1e-10)
        xo_mean = torch.log(xo_mean + 1e-10)
        
        return xo_mean, xc_mean


class CausalGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        # 添加时间序列分解模块
        self.decomposition = series_decomp(kernel_size=11)
        # 为季节性和趋势性分别创建Layer Norm和attention模块
        # self.trend_norm = my_Layernorm(hidden)
        # self.seasonal_norm = my_Layernorm(hidden)
        
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        # self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.conv_feat_seasonal = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.conv_feat_trend = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.trend_node_att = nn.Sequential(
            nn.Linear(hidden_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden_in)
        )
        
        self.seasonal_node_att = nn.Sequential(
            nn.Linear(hidden_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden_in)
        )

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden_in * 2, 2)
        self.node_att_mlp = nn.Linear(hidden_in, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):
        x = data.x if data.x is not None else data.feat  # [num_nodes, seq_len, hidden]

        edge_index, batch = data.edge_index, data.batch
        # print('--------------:', x.shape)

        row, col = edge_index
        x = self.bn_feat(x)


        # 对输入进行时间序列分解
        seasonal, trend = self.decomposition(x)
        # 分别计算趋势和季节性的attention
        if self.without_node_attention:
            trend_att = 0.5 * torch.ones(trend.shape[0], 2, device=x.device)
            seasonal_att = 0.5 * torch.ones(seasonal.shape[0], 2, device=x.device)
        else:
            trend_att = self.trend_node_att(trend)
            seasonal_att = self.seasonal_node_att(seasonal)

        seasonal_edge_rep = torch.cat([seasonal[row], seasonal[col]], dim=-1)
        trend_edge_rep = torch.cat([trend[row], trend[col]], dim=-1)
        if self.without_edge_attention:
            seasonal_edge_att = 0.5 * torch.ones(seasonal_edge_rep.shape[0], 2, device=x.device)
            trend_edge_att = 0.5 * torch.ones(trend_edge_rep.shape[0], 2, device=x.device)
        else:
            seasonal_edge_att = F.softmax(self.edge_att_mlp(seasonal_edge_rep), dim=-1)
            trend_edge_att = F.softmax(self.edge_att_mlp(trend_edge_rep), dim=-1)

        seasonal_edge_weight_c = seasonal_edge_att[:, 0]
        seasonal_edge_weight_o = seasonal_edge_att[:, 1]

        trend_edge_weight_c = trend_edge_att[:, 0]
        trend_edge_weight_o = trend_edge_att[:, 1]

        # seasonal_xc = seasonal_att[:, 0].view(-1, 1) * seasonal
        # seasonal_xo = seasonal_att[:, 1].view(-1, 1) * seasonal

        seasonal_att = seasonal_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        seasonal_att = F.softmax(seasonal_att, dim=1)  # 在第1维(2)上做softmax
        trend_att = trend_att.view(x.shape[0], 2, -1)  # [num_nodes, 2, hidden_in]
        trend_att = F.softmax(trend_att, dim=1)  # 在第1维(2)上做softmax

        # 分别获取context和object的attention权重
        seasonal_att_c = seasonal_att[:, 0, :]  # [num_nodes, hidden_in]
        seasonal_att_o = seasonal_att[:, 1, :]  # [num_nodes, hidden_in]
        trend_att_c = trend_att[:, 0, :]  # [num_nodes, hidden_in]
        trend_att_o = trend_att[:, 1, :]  # [num_nodes, hidden_in]

        # 对每个feature维度分别应用attention
        seasonal_xc = seasonal_att_c * seasonal  # 元素级别的乘法
        seasonal_xo = seasonal_att_o * seasonal
        trend_xc = trend_att_c * trend  # 元素级别的乘法
        trend_xo = trend_att_o * trend
        

        # trend_xc = trend_att[:, 0].view(-1, 1) * trend
        # trend_xo = trend_att[:, 1].view(-1, 1) * trend

        seasonal_xc = F.relu(self.conv_feat_seasonal(seasonal_xc, edge_index, seasonal_edge_weight_c))
        trend_xc = F.relu(self.conv_feat_trend(trend_xc, edge_index, seasonal_edge_weight_o))

        seasonal_xo = F.relu(self.conv_feat_seasonal(seasonal_xo, edge_index, trend_edge_weight_c))
        trend_xo = F.relu(self.conv_feat_trend(trend_xo, edge_index, trend_edge_weight_o))

        
        # 合并趋势和季节性特征
        xc = trend_xc + seasonal_xc
        xo = trend_xo + seasonal_xo

        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        # xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        
        return xc_logis, xo_logis, xco_logis



    # def forward(self, data, eval_random=True, save_attention=False):

    #     x = data.x if data.x is not None else data.feat
    #     edge_index, batch = data.edge_index, data.batch
    #     cor = data.edge_attr
        
    #     row, col = edge_index
    #     x = self.bn_feat(x)

    #     if self.without_node_attention:
    #         node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
    #     else:
    #         node_att = F.softmax(self.node_att_mlp(x), dim=-1)

        
    #     edge_rep = torch.cat([x[row], x[col]], dim=-1)

    #     if self.without_edge_attention:
    #         edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
    #     else:
    #         edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
    #     edge_weight_c = edge_att[:, 0] * torch.abs(cor.squeeze(1))
    #     edge_weight_o = edge_att[:, 1] * torch.abs(cor.squeeze(1))


    #     xc = node_att[:, 0].view(-1, 1) * x
    #     xo = node_att[:, 1].view(-1, 1) * x
    #     xc = F.relu(self.conv_feat(xc, edge_index, edge_weight_c))
    #     xo = F.relu(self.conv_feat(xo, edge_index, edge_weight_o))
        

    #     # xc = self.global_pool(xc, batch)
    #     # xo = self.global_pool(xo, batch)
        
    #     xc_logis = self.context_readout_layer(xc)
    #     xo_logis = self.objects_readout_layer(xo)
    #     xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

    #     return xc_logis, xo_logis, xco_logis



    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = (xc[random_idx] + xo)/2

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis
    

class CausalGCN_begin(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes, args,
                 gfn=False,
                 collapse=False,
                 residual=False,
                 res_branch="BNConvReLU",
                 global_pool="sum",
                 dropout=0,
                 edge_norm=True):
        super(CausalGCN_begin, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv_c = torch.nn.ModuleList()
        self.convs_c = torch.nn.ModuleList()
        self.bns_conv_o = torch.nn.ModuleList()
        self.convs_o = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv_c.append(BatchNorm1d(hidden))
            self.convs_c.append(GConv(hidden, hidden))
        for i in range(num_conv_layers):
            self.bns_conv_o.append(BatchNorm1d(hidden))
            self.convs_o.append(GConv(hidden, hidden))

        # self.edge_att_mlp = nn.Linear(hidden_in * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden_in, 2)
        self.edge_att_mlp = nn.Sequential(
            nn.Linear(hidden_in * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
        self.bnc = BatchNorm1d(hidden_in)
        self.bno = BatchNorm1d(hidden_in)
        self.context_convs = GConv(hidden_in, hidden)
        self.objects_convs = GConv(hidden_in, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index


        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        for i, conv_c in enumerate(self.convs_c):
            xc = self.bns_conv_c[i](xc)
            xc = F.relu(conv_c(xc, edge_index, edge_weight_c))
        for i, conv_o in enumerate(self.convs_o):
            xo = self.bns_conv_o[i](xo)
            xo = F.relu(conv_o(xo, edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)

        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):

        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class CausalGCN_share(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes, args,
                 gfn=False,
                 collapse=False,
                 residual=False,
                 res_branch="BNConvReLU",
                 global_pool="sum",
                 dropout=0,
                 edge_norm=True):
        super(CausalGCN_share, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        # self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        # self.fc1_bn_c = BatchNorm1d(hidden)
        # self.fc1_c = Linear(hidden, hidden)
        # self.fc2_bn_c = BatchNorm1d(hidden)
        # self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=edge_rep.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.objects_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)

        xc_logis = self.objects_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis

    # def context_readout_layer(self, x):
    #
    #     x = self.fc1_bn_c(x)
    #     x = self.fc1_c(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_c(x)
    #     x = self.fc2_c(x)
    #     x_logis = F.log_softmax(x, dim=-1)
    #     return x_logis

    def objects_readout_layer(self, x):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

class CausalGIN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                gfn=False,
                edge_norm=True):
        super(CausalGIN, self).__init__()

        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(
                       Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, train_type="base", save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        # edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]
        
        
        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        xo_logis = self.objects_readout_layer(xo, train_type)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis


    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x, train_type):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        if train_type == "irm":
            return x, x_logis
        else:
            return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class CausalGIN_share(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes, args,
                 gfn=False,
                 edge_norm=True):
        super(CausalGIN_share, self).__init__()

        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    BatchNorm1d(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU())))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        # self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        # self.fc1_bn_c = BatchNorm1d(hidden)
        # self.fc1_c = Linear(hidden, hidden)
        # self.fc2_bn_c = BatchNorm1d(hidden)
        # self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, train_type="base", save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.objects_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)

        xc_logis = self.objects_readout_layer(xc, train_type)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        xo_logis = self.objects_readout_layer(xo, train_type)

        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis

    # def context_readout_layer(self, x):
    #
    #     x = self.fc1_bn_c(x)
    #     x = self.fc1_c(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_c(x)
    #     x = self.fc2_c(x)
    #     x_logis = F.log_softmax(x, dim=-1)
    #     return x_logis

    def objects_readout_layer(self, x, train_type):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        if train_type == "irm":
            return x, x_logis
        else:
            return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)

        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class CausalGIN_begin(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                gfn=False,
                edge_norm=True):
        super(CausalGIN_begin, self).__init__()

        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs_c = torch.nn.ModuleList()
        self.convs_o = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs_c.append(GINConv(
            Sequential(
                       Linear(hidden, hidden),
                       BatchNorm1d(hidden),
                       ReLU(),
                       Linear(hidden, hidden),
                       ReLU())))
        for i in range(num_conv_layers):
            self.convs_o.append(GINConv(
            Sequential(
                       Linear(hidden, hidden),
                       BatchNorm1d(hidden),
                       ReLU(),
                       Linear(hidden, hidden),
                       ReLU())))
        # self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        # self.node_att_mlp = nn.Linear(hidden, 2)
        self.edge_att_mlp = nn.Sequential(
            nn.Linear(hidden_in * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
        self.bnc = BatchNorm1d(hidden_in)
        self.bno= BatchNorm1d(hidden_in)
        self.context_convs = GConv(hidden_in, hidden)
        self.objects_convs = GConv(hidden_in, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, train_type="base", save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        # x = self.bn_feat(x)
        # x = F.relu(self.conv_feat(x, edge_index))

        # for i, conv in enumerate(self.convs):
        #     x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        # edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]


        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))
        for i, conv in enumerate(self.convs_c):
            xc = conv(xc, edge_index)
        for i, conv in enumerate(self.convs_o):
            xo = conv(xo, edge_index)
        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)

        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        xo_logis = self.objects_readout_layer(xo, train_type)

        # 如果是测试阶段且需要保存attention
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis



    def context_readout_layer(self, x):

        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x, train_type):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        if train_type == "irm":
            return x, x_logis
        else:
            return x_logis
    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)

        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class CausalGAT(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes, 
                       args, 
                       head=4, 
                       dropout=0.):
        super(CausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2, device=x.device)
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        # edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2, device=x.device)
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict
        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class CausalGAT_share(torch.nn.Module):
    def __init__(self, num_features,
                 num_classes,
                 args,
                 head=4,
                 dropout=0.):
        super(CausalGAT_share, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        # self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        # self.fc1_bn_c = BatchNorm1d(hidden)
        # self.fc1_c = Linear(hidden, hidden)
        # self.fc2_bn_c = BatchNorm1d(hidden)
        # self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, save_attention=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.objects_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)

        xc_logis = self.objects_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        if save_attention:
            attention_dict = {
                'edge_attention': edge_att.detach().cpu(),
                'node_attention': node_att.detach().cpu(),
                'batch': batch.detach().cpu(),
                'edge_index': edge_index.detach().cpu()
            }
            return xc_logis, xo_logis, xco_logis, attention_dict

        return xc_logis, xo_logis, xco_logis

    # def context_readout_layer(self, x):
    #
    #     x = self.fc1_bn_c(x)
    #     x = self.fc1_c(x)
    #     x = F.relu(x)
    #     x = self.fc2_bn_c(x)
    #     x = self.fc2_c(x)
    #     x_logis = F.log_softmax(x, dim=-1)
    #     return x_logis

    def objects_readout_layer(self, x):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)

        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

class GCNNet(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, hidden, 
                       num_feat_layers=1, 
                       num_conv_layers=1,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(GCNNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        # self.conv_feat = GINConv(Sequential(
        #                         Linear(hidden_in, hidden),
        #                         BatchNorm1d(hidden),
        #                         # ReLU(),
        #                         # Linear(hidden, hidden),
        #                         ReLU())) # GIN
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        cor = data.edge_attr

        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))


            
        # x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
    

# class GCNNet(torch.nn.Module):
#     """GCN with BN and residual connection. -----modified""" 
#     def __init__(self, num_features,
#                        num_classes, hidden, 
#                        num_feat_layers=1, 
#                        num_conv_layers=3,
#                  num_fc_layers=2, gfn=False, collapse=False, residual=False,
#                  res_branch="BNConvReLU", global_pool="sum", dropout=0, 
#                  edge_norm=True):
#         super(GCNNet, self).__init__()

#         self.global_pool = global_add_pool
#         self.dropout = dropout
#         GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

#         hidden_in = num_features
#         self.bn_feat = BatchNorm1d(hidden_in)
#         self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
#         self.bns_conv = torch.nn.ModuleList()
#         self.convs = torch.nn.ModuleList()

#         for i in range(num_conv_layers):
#             self.bns_conv.append(BatchNorm1d(hidden))
#             self.convs.append(GConv(hidden, hidden))
#         self.bn_hidden = BatchNorm1d(hidden)
#         self.bns_fc = torch.nn.ModuleList()
#         self.lins = torch.nn.ModuleList()

#         for i in range(num_fc_layers - 1):
#             self.bns_fc.append(BatchNorm1d(hidden))
#             self.lins.append(Linear(hidden, hidden))
#         self.lin_class = Linear(hidden, num_classes)

#         # BN initialization.
#         for m in self.modules():
#             if isinstance(m, (torch.nn.BatchNorm1d)):
#                 torch.nn.init.constant_(m.weight, 1)
#                 torch.nn.init.constant_(m.bias, 0.0001)

#     def forward(self, data):
        
#         x = data.x if data.x is not None else data.feat
#         edge_index, batch = data.edge_index, data.batch
        
#         x = self.bn_feat(x)
#         x = F.relu(self.conv_feat(x, edge_index))

#         x = self.bn_hidden(x)
#         x = self.lin_class(x)
#         return F.log_softmax(x, dim=-1)
    

class GINNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes,
                       hidden, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        # x, edge_index, batch = data.feat, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
        # x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))    
        x = self.bn_hidden(x)
        x = self.lin_class(x)

        prediction = F.log_softmax(x, dim=-1)
        return prediction

class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        # x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)



class CAL_M_for_toy(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CAL_M_for_toy, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        self.hidden = hidden
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.node_num = args.node_num
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden_in)
        self.fc1_c = Linear(hidden_in, hidden_out)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden_in)
        self.fc1_o = Linear(hidden_in, hidden_out)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden_in)
            self.fc1_co = Linear(hidden_in, hidden_out)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
        
        self.decompsition = DFT_series_decomp(5)
        self.cross_layer = nn.Sequential(
                nn.Linear(in_features=hidden_in, out_features=hidden),
                nn.GELU(),
                nn.Linear(in_features=hidden, out_features=hidden),
            )

    def forward(self, xc, xo):
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis, xoc_logis = self.random_readout_layer(xc, xo, n_times=100)

        return xc_logis, xo_logis, xco_logis, xoc_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, n_times=100):
        num = xc.shape[0]
        xc_results = []
        xo_results = []
        
        # 进行100次随机匹配
        for _ in range(n_times):
            # 使用torch.randperm替代random.shuffle
            random_idx = torch.randperm(num)
            
            if self.args.cat_or_add == "cat":
                x = torch.cat((xc[random_idx], xo), dim=1)
            else:
                x = xc[random_idx] + xo
                
            x = self.fc1_bn_co(x)
            x = self.fc1_co(x)
            x_logis = F.log_softmax(x, dim=-1)
            
            # 存储每次的结果
            xc_results.append(x_logis)
            xo_results.append(x_logis[torch.argsort(random_idx)])
        
        # 计算平均值
        xc_mean = torch.stack(xc_results).mean(dim=0)  # 对xc的所有匹配结果取平均
        xo_mean = torch.stack(xo_results).mean(dim=0)  # 对xo的所有匹配结果取平均
        
        return xc_mean, xo_mean
    
class CAL_for_toy(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CAL_for_toy, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        self.hidden = hidden
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.node_num = args.node_num
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden_in)
        self.fc1_c = Linear(hidden_in, hidden_out)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden_in)
        self.fc1_o = Linear(hidden_in, hidden_out)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden_in)
            self.fc1_co = Linear(hidden_in, hidden_out)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
        
        self.decompsition = DFT_series_decomp(5)
        self.cross_layer = nn.Sequential(
                nn.Linear(in_features=hidden_in, out_features=hidden),
                nn.GELU(),
                nn.Linear(in_features=hidden, out_features=hidden),
            )

    def forward(self, xc, xo):
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo)

        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo):

        num = xc.shape[0]
        l = [i for i in range(num)]
        # random.shuffle(l)
        random_idx = torch.tensor(l)

        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis