import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])  # Take only the last time step
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    

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
    train_and_evaluate(train_files, val_file, test_file, method='Transformer')