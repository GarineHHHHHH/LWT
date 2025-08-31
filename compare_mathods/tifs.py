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
    train = True
    if train == True:
        from train_evaluate import train_and_evaluate
        data_dir = f"data/wave-csv3"
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
        seq_length = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        method = 'WT-TCN'
        
        # 训练和评估模型
        train_and_evaluate(train_files, val_file, test_file, epochs=30, method='WT-TCN')
    
    # 定义文件路径
    else:
        from train_evaluate import train_and_evaluate
        from torch.utils.data import DataLoader, TensorDataset
        import time
        import psutil
        # levels = [1, 2, 3, 4]  # 小波分解的层数
        
        # for level in levels:
        data_dir = f"data/wave-csv3"
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
        seq_length = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        method = 'WT-TCN'
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
        # 训练和评估模型
        # train_and_evaluate(train_files, val_file, test_file,epochs=30,method='WT-TCN')
        # 加载模型models/best_WT-TCN_model.pth 测试
        model = TCNModel(input_size=input_size, output_size=3, num_channels=[16, 32, 64], kernel_size=2,dropout=0.3).to(device)

        # 3. 加载最佳模型
        model.load_state_dict(torch.load(os.path.join('models',f'best_WT-TCN_model.pth')))
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
            
        