import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import os
from calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics
# from train_evaluate import just_test

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

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # TCN needs data with shape (N, C, L)
        x = x.transpose(1, 2)  # Transpose to (N, C, L)
        x = self.tcn(x)
        x = x[:, :, -1] # Taking the last value of the output sequence
        return self.linear(x)

def TCNModel(input_size, output_size, num_channels, kernel_size, dropout):
    return TCN(input_size, output_size, num_channels, kernel_size, dropout)

if __name__ == "__main__":
    # 定义文件路径
       # 定义文件路径
    from train_evaluate import train_and_evaluate
    data_dir = "data/csv6"
    train_files = [
        os.path.join(data_dir, '2-spark.csv'),
        os.path.join(data_dir, '3-spark.csv'),
        os.path.join(data_dir, '3-triangle.csv'),
        os.path.join(data_dir, '4-triangle.csv'),
        os.path.join(data_dir, '5-rectangle.csv'),
        os.path.join(data_dir, '6-rectangle.csv')
    ]
    # val_files = os.path.join(data_dir, '7-rectangle.csv')
    val_file = os.path.join(data_dir, '7-rectangle.csv')
    test_file = os.path.join(data_dir, '8-rectangle.csv')
    
    # 训练和评估模型
    train_and_evaluate(train_files, val_file, test_file,method='TCN')