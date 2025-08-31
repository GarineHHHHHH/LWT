import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics

import os

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播
        out, _ = self.rnn(x, h0)
        
        # 将最后一个时间步的输出传递给全连接层
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # 定义文件路径
    from train_evaluate import train_and_evaluate
    data_dir = "csv6"
    train_files = [
        os.path.join(data_dir, '5-rectangle.csv'),
        os.path.join(data_dir, '6-rectangle.csv'),
        os.path.join(data_dir, '7-rectangle.csv')
    ]
    val_file = os.path.join(data_dir, '8-rectangle.csv')
    test_file = os.path.join(data_dir, '9-rectangle.csv')
    
    # 训练和评估模型
    train_and_evaluate(train_files, val_file, test_file,method='RNN')