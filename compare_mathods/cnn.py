import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics
import os
import matplotlib.pyplot as plt

# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, seq_length, input_size):  # 修改输入大小为7
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        def calc_output_size(seq_len):
            seq_len = seq_len - 3 + 1
            seq_len = seq_len // 2
            seq_len = seq_len - 3 + 1
            seq_len = seq_len // 2
            return seq_len
        
        output_size = calc_output_size(seq_length)
        self.flatten_size = 32 * output_size
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 3)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    from train_evaluate import train_and_evaluate
    # 定义文件路径
    data_dir = "csv6"
    train_files = [
        os.path.join(data_dir, '5-rectangle.csv'),
        os.path.join(data_dir, '6-rectangle.csv'),
        os.path.join(data_dir, '7-rectangle.csv')
    ]
    val_file = os.path.join(data_dir, '8-rectangle.csv')
    test_file = os.path.join(data_dir, '9-rectangle.csv')
    
    # 训练和评估模型
    train_and_evaluate(train_files, val_file, test_file,method='CNN')