
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from calTools import create_sequences, preprocess_data,  visualize_results, save_results_to_csv, visulize_loss, save_loss_to_csv, calculate_and_save_metrics
from cnn import CNNModel
from rnn import RNNModel
from lstm import LSTMModel
from kalman import KalmanFilter
from gru import GRUModel
from tifs import TCNModel
from bilstm import BiLSTMModel
from transformer import TransformerModel
# from atlstm import LSTMWithAttentionModel
# from at_wt_tcn import AT_TCNModel
from tcn_kalman import KalmanTCN
import time
import psutil
def train_and_evaluate(train_files, val_file, test_file, seq_length=20, epochs=50, batch_size=32, learning_rate=0.001,method='LSTM',output_path='results'):
    
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
    print(f"Input shape: {X_train.shape}")
    print(f"Input size: {input_size}")
    if method == 'CNN':
        model = CNNModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'RNN':
        model = RNNModel(input_size, hidden_size=10, num_layers=2, output_size=3).to(device)
    elif method == 'Kalman':
        # 卡尔曼滤波器不需要训练
        print("请切换使用卡尔曼滤波器进行预测")
        pass
    elif method == 'LSTM':
        model = LSTMModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'GRU':
        model = GRUModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'Transformer':
        model = TransformerModel( input_size=input_size, output_size=3).to(device)
    elif method == 'TCN':
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16, 32], kernel_size=2,dropout=0.3).to(device)
    elif method == 'BiLSTM':
        model = BiLSTMModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'WT-TCN':
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16,  32,64], kernel_size=2,dropout=0.3).to(device)
    elif method == 'KF-WT-CN':
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16, 32, 64], kernel_size=2,dropout=0.3).to(device)

    # 3. 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. 训练循环
    best_val_loss = float('inf')
    train_losses = []  # 存储训练损失
    val_losses = [] # 存储验证损失
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
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 5. 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
        
        # print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('models',f'best_{method}_model.pth'))
            print("保存最佳模型")
    
    visulize_loss(train_losses, val_losses, method,output_path=output_path)
    save_loss_to_csv(train_losses, val_losses, method,output_path=output_path)

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
    results_dir = os.path.join(output_path, f'{method}')
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
        
        calculate_and_save_metrics(true_targets, predicted_targets, method,output_path=output_path)
        
        # 9. 可视化结果
        visualize_results(true_targets, predicted_targets, method,output_path=output_path)
        
        # 10. 保存结果到CSV
        save_results_to_csv(true_targets, predicted_targets,inference_times, method,output_path=output_path)
    
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time per sample: {avg_inference_time:.4f} seconds")\
        
def just_test(test_file, seq_length=20, method='LSTM'):
    
    """
    测试CNN模型。
    
    参数:
        test_file (str): 测试集CSV文件路径。
        seq_length (int): 序列长度。
        method (str): 使用的模型方法。
    
    返回值:
        None
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    # 1. 数据准备
    # 1.1 加载和归一化测试数据
    test_data, feature_scaler, target_scaler = preprocess_data(test_file)
    
    # 1.2 创建序列
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # 1.3 转换为PyTorch张量
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    # 1.4 创建数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 2. 模型定义
    input_size = X_test.shape[2]
    if method == 'CNN':
        model = CNNModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'RNN':
        model = RNNModel(input_size, hidden_size=10, num_layers=2, output_size=3).to(device)
    elif method == 'Kalman':
        # 卡尔曼滤波器不需要训练
        print("请切换使用卡尔曼滤波器进行预测")
        pass
    elif method == 'LSTM':
        model = LSTMModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'GRU':
        model = GRUModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'Transformer':
        model = TransformerModel( input_size=input_size, output_size=3).to(device)
    elif method == 'TCN':
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16, 32], kernel_size=2,dropout=0.3).to(device)
    elif method == 'BiLSTM':
        model = BiLSTMModel(seq_length=seq_length, input_size=input_size).to(device)
    elif method == 'WT-TCN':
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16, 32, 64], kernel_size=2,dropout=0.3).to(device)
    elif method == 'KF-WT-CN':
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16, 32, 64], kernel_size=2,dropout=0.3).to(device)

    # 3. 加载最佳模型
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
    results_dir = os.path.join('test', f'{method}')
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
        
        calculate_and_save_metrics(true_targets, predicted_targets,output_path='test', method=method)
        
        # 9. 可视化结果
        visualize_results(true_targets, predicted_targets,output_path='test', method=method)
        
        # 10. 保存结果到CSV
        save_results_to_csv(true_targets, predicted_targets,inference_times, output_path='test', method=method)
    
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time per sample: {avg_inference_time:.4f} seconds")\
        
    