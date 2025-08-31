import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from compare_mathods.calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics,visualize_attention
import matplotlib.pyplot as plt
import psutil

# 定义带有注意力机制的LSTM模型
class LSTMWithAttentionModel(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size1=128, hidden_size2=64):
        super(LSTMWithAttentionModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, 
                           batch_first=True, num_layers=1, bidirectional=True)  # Bidirectional LSTM
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=2*hidden_size1, hidden_size=hidden_size2,  # Input size is doubled because of bidirectional LSTM
                           batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(0.2)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size2, 1)

        self.fc1 = nn.Linear(hidden_size2, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 3)
        
        self.attention_weights = None  # Store attention weights for visualization
        
    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(out).squeeze(-1), dim=-1)
        
        self.attention_weights = attention_weights.detach().cpu().numpy()  # Store weights

        # Apply attention weights to LSTM output
        attended_output = torch.sum(out * attention_weights.unsqueeze(-1), dim=1)
        
        out = self.relu(self.fc1(attended_output))
        out = self.fc2(out)
        return out
    
    def get_attention_weights(self):
        return self.attention_weights


def train_and_evaluate(train_files, val_file, test_file, seq_length=20, epochs=30, batch_size=32, learning_rate=0.001,method='LSTM'):
    
    """
    训练、验证和测试CNN模型。
    
    参数:
        train_files (list): 训练集CSV文件路径列表。
        val_file (str): 验证集CSV文件路径。
        test_file (str): 测试集CSV文件路径。
        seq_length (int): 序列长度。
        epochs (int): 训练轮数。
        batch_size (int): 批大小。
        learning_rate (float): 学习率。
    
    返回值:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 数据准备
    # 1.1 加载和归一化训练数据
    train_data_list = []
    for file in train_files:
        scaled_data, _, _ = preprocess_data(file)
        train_data_list.append(scaled_data)
    train_data = np.concatenate(train_data_list, axis=0)
    
    # 1.2 加载和归一化验证数据
    val_data, _, _ = preprocess_data(val_file)
    
    # 1.3 加载和归一化测试数据
    test_data, feature_scaler, target_scaler = preprocess_data(test_file)
    
    # 1.4 创建序列
    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(val_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # 1.5 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # 1.6 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 模型定义
    input_size = X_train.shape[2]

    model = LSTMWithAttentionModel(seq_length=seq_length, input_size=input_size).to(device)
        
    # 3. 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. 训练循环
    best_val_loss = float('inf')
    train_losses = []  # 存储训练损失
    val_losses = [] # 存储验证损失
        # Initialize an array to store attention weights for all epochs
    all_attention_weights = np.zeros((epochs, seq_length))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
         
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
            # Store attention weights
            attention_weights = model.get_attention_weights()
            # Store attention weights
            if attention_weights is not None:
                batch_size = attention_weights.shape[0]
                all_attention_weights[epoch, :] = np.mean(attention_weights, axis=0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 5. 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('models',f'best_{method}_model.pth'))
            best_model = model
            print("保存最佳模型")

    # Calculate average attention weights across all epochs
    average_attention_weights = np.mean(all_attention_weights, axis=0)
    
    # Visualize average attention weights
    visualize_attention(average_attention_weights, title="Average Attention Weights Across All Epochs")

    visulize_loss(train_losses, val_losses, method)
    save_loss_to_csv(train_losses, val_losses, method)

    # 7. 测试
    model.load_state_dict(torch.load(os.path.join('models',f'best_{method}_model.pth')))
    model.eval()
    
    # 获取模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    # 获取模型占用的显存或内存
    if device.type == 'cuda':
        memory_usage = torch.cuda.memory_allocated(device) / 1024 ** 2  # 以MB为单位
        memory_info = f"Memory allocated on GPU: {memory_usage:.2f} MB"
    else:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 ** 2  # 以MB为单位
        memory_info = f"Memory used by process: {memory_usage:.2f} MB"
    print(memory_info)
    
    # 保存memory info
    results_dir = os.path.join('results', f'{method}')
    os.makedirs(results_dir, exist_ok=True)
    
    memory_df = pd.DataFrame({'Memory Usage': [memory_info]})
    memory_df.to_csv(os.path.join(results_dir, 'memory_usage.csv'), index=False)
    
    
    inference_times = []
    predicted_targets = []
    with torch.no_grad():
        for i in range(len(X_test)):
            start_time = time.time()
            test_output = model(X_test[i].unsqueeze(0))  # 增加一个batch维度
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            predicted_target = test_output.cpu().numpy()
            predicted_targets.append(predicted_target)
            
        # 反归一化
        predicted_targets = target_scaler.inverse_transform(np.concatenate(predicted_targets, axis=0))
        true_targets = target_scaler.inverse_transform(y_test.cpu().numpy())
        
        calculate_and_save_metrics(true_targets, predicted_targets, method)
        
        # 9. 可视化结果
        visualize_results(true_targets, predicted_targets, method)
        
        # 10. 保存结果到CSV
        save_results_to_csv(true_targets, predicted_targets,inference_times, method)
    
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time per sample: {avg_inference_time:.4f} seconds")
    
if __name__ == "__main__":
    # 定义文件路径
    data_dir = "data/csv6"
    train_files = [
        os.path.join(data_dir, '5-rectangle.csv'),
        os.path.join(data_dir, '6-rectangle.csv'),
        os.path.join(data_dir, '7-rectangle.csv')
    ]
    val_file = os.path.join(data_dir, '8-rectangle.csv')
    test_file = os.path.join(data_dir, '9-rectangle.csv')
    
    # 训练和评估模型
    train_and_evaluate(train_files, val_file, test_file,epochs=30,method='AT-LSTM')

    