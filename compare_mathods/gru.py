import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size1=128, hidden_size2=64):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size1, 
                           batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(input_size=hidden_size1, hidden_size=hidden_size2, 
                           batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 3)
        
    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.dropout2(out[:, -1, :])  # Take only the last time step output
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
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
    train_and_evaluate(train_files, val_file, test_file,method='GRU')
