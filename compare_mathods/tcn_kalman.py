import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import os
from calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics
class SimpleKalmanFilter:
    def __init__(self, dim, Q=1e-5, R=1e-2):
        self.Q = Q  # 过程噪声
        self.R = R  # 观测噪声
        self.P = np.ones(dim)
        self.x = np.zeros(dim)

    def filter(self, z):
        # z: shape (batch, dim)
        x_hat = self.x
        P_hat = self.P + self.Q
        K = P_hat / (P_hat + self.R)
        self.x = x_hat + K * (z - x_hat)
        self.P = (1 - K) * P_hat
        return self.x

# 定义一个full卷积: 去除一维卷积后多余的padding，保证输出长度与输入一致，防止未来信息泄露。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    
    '''
    结构：
    一维卷积: conv1d
    chomp1d: 去掉多余的padding
    ReLU激活函数
    Dropout
    一维卷积: conv1d
    chomp1d: 去掉多余的padding
    ReLU激活函数
    
    Func: 实现带有残差的膨胀卷积模块，捕获长时序依赖
    '''
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    '''
    1. 多个膨胀卷积模块级联
    2. 每一层的膨胀系数为2的幂次方
    3. 每一层的参数可配置
    
    作用: 堆叠多层膨胀卷积模块, 增强时序建模
    '''
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KalmanTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(KalmanTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.kalman = SimpleKalmanFilter(dim=output_size)

    def forward(self, x):
        # TCN needs data with shape (N, C, L)
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.tcn(x)
        x = x[:, :, -1] # (N, C)
        x = self.linear(x) # (N, output_size)
        # Kalman滤波（逐batch处理）
        x_np = x.detach().cpu().numpy()
        x_kalman = np.array([self.kalman.filter(z) for z in x_np])
        x_kalman = torch.tensor(x_kalman, dtype=x.dtype, device=x.device)
        return x_kalman

def TCNModel(input_size, output_size, num_channels, kernel_size, dropout):
    return KalmanTCN(input_size, output_size, num_channels, kernel_size, dropout)

if __name__ == "__main__":
    # 定义文件路径
    from train_evaluate import train_and_evaluate
    # data_dir = "csv6"
    # train_files = [
    #     os.path.join(data_dir, '5-rectangle.csv'),
    #     os.path.join(data_dir, '6-rectangle.csv'),
    #     os.path.join(data_dir, '7-rectangle.csv')
    # ]
    # val_file = os.path.join(data_dir, '8-rectangle.csv')
    # test_file = os.path.join(data_dir, '9-rectangle.csv')

    # # 训练和评估模型
    # train_and_evaluate(train_files, val_file, test_file, method='TCN')
    data_dir = "data/wave-csv2"
    train_files = [
        os.path.join(data_dir, 'tra1.csv'),
        os.path.join(data_dir, 'tra2.csv'),
        os.path.join(data_dir, 'tra3.csv'),
        os.path.join(data_dir, 'tra4.csv'),
        os.path.join(data_dir, 'tra5.csv'),
        os.path.join(data_dir, 'tra6.csv'),
    ]
    val_file = os.path.join(data_dir, 'tra7.csv')
    test_file = os.path.join(data_dir, 'tra8.csv')
    
    # 训练和评估模型
    train_and_evaluate(train_files, val_file, test_file,epochs=30,method='KF-WT-CN')

    