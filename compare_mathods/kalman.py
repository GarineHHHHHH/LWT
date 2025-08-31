import numpy as np
import pandas as pd
import os
from calTools import calculate_and_save_metrics, visualize_results, save_results_to_csv
import matplotlib.pyplot as plt
import time
# 1. 定义卡尔曼滤波核心原理和结构
class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, Q, R, H):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # 状态转移矩阵 (假设匀速运动)
        self.F = np.eye(state_dim)
        
        # 观测矩阵
        self.H = H
        
        # 过程噪声协方差矩阵
        self.Q = Q
        
        # 观测噪声协方差矩阵
        self.R = R
        
        # 状态估计协方差矩阵
        self.P = np.eye(state_dim)
        
        # 状态估计
        self.x = np.zeros((state_dim, 1))
    
    def predict(self):
        # 状态预测
        self.x = np.dot(self.F, self.x)
        
        # 协方差预测
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, z):
        # 计算卡尔曼增益
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 更新状态估计
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        
        # 更新协方差矩阵
        I = np.eye(self.state_dim)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
    
    def get_state(self):
        return self.x

def preprocess_data(file_path):
    """
    对单个CSV文件进行预处理，包括填充缺失值、选择特征和目标。
    
    参数:
        file_path (str): CSV文件的路径。
    
    返回值:
        tuple: 包含特征和目标的numpy数组。
    """
    data = pd.read_csv(file_path)
    # 处理缺失值
    data = data.fillna(0)
    # 选择特征和目标
    features = data[['vx_approx_mean','vx_approx_std','vx_detail_mean','vx_detail_std',
                     'vy_approx_mean','vy_approx_std','vy_detail_mean','vy_detail_std',
                     'vz_approx_mean','vz_approx_std','vz_detail_mean','vz_detail_std',
                     ]]
    targets = data[['product_gps_longitude', 'product_gps_latitude', 'altitude']]
    
    return features.values, targets.values

def train_and_evaluate(train_files, val_file, test_file, method='Kalman'):
    """
    训练、验证和测试卡尔曼滤波模型。
    
    参数:
        train_files (list): 训练集CSV文件路径列表 (未使用，卡尔曼滤波无训练)。
        val_file (str): 验证集CSV文件路径 (未使用，卡尔曼滤波无验证)。
        test_file (str): 测试集CSV文件路径。
    
    返回值:
        None
    """
    
    # 1. 数据准备
    # 1.3 加载测试数据
    test_features, test_targets = preprocess_data(test_file)
    
    # 2. 模型定义
    state_dim = 9  # longitude, latitude, altitude + velocities
    measurement_dim = 3  # longitude, latitude, altitude

    # 观测矩阵H (只观测位置)
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # longitude
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # latitude
        [0, 0, 1, 0, 0, 0, 0, 0, 0]   # altitude
    ])

    # 过程噪声和观测噪声 (需要根据实际数据调整)
    Q = np.eye(state_dim) * 0.01
    R = np.eye(measurement_dim) * 1

    kf = KalmanFilter(state_dim, measurement_dim, Q, R, H)

    # 初始化状态 (需要根据实际数据调整)
    initial_state = np.zeros((state_dim, 1))
    initial_state[0:3, 0] = test_targets[0, :]  # 初始位置
    kf.x = initial_state
    
    # 3. 推断
    predicted_targets = []
    # 记录推断时间
    inference_times = []
    # 记录内存使用情况
    memory_usage = []
    
    for i in range(1, len(test_features)):
        
        start_time = time.time()
        # 预测
        kf.predict()
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # 更新
        measurement = test_targets[i, :].reshape(-1, 1)
        kf.update(measurement)
        
        # 记录预测值
        predicted_targets.append(kf.get_state()[0:3].flatten())
    
    # 转换为numpy数组
    predicted_targets = np.array(predicted_targets)
    true_targets = test_targets[1:, :]
    
    # 4. 计算评估指标
    calculate_and_save_metrics(true_targets, predicted_targets,method)
    
    
    # 5. 可视化结果
    visualize_results(true_targets, predicted_targets, method)
    
    # 10. 保存结果到CSV
    save_results_to_csv(true_targets, predicted_targets,inference_times, method)


if __name__ == "__main__":
    # 定义文件路径
    # data_dir = "data/csv6"
    # train_files = [
    #     os.path.join(data_dir, '5-rectangle.csv'),
    #     os.path.join(data_dir, '6-rectangle.csv'),
    #     os.path.join(data_dir, '7-rectangle.csv')
    # ]
    # val_file = os.path.join(data_dir, '8-rectangle.csv')
    # test_file = os.path.join(data_dir, '4-trangle.csv')
    
    # # 训练和评估模型
    # train_and_evaluate(train_files, val_file, test_file, method='Kalman')
    
    data_dir = "data/wave-csv"
    train_files = [
        os.path.join(data_dir, 'rec1.csv'),
        os.path.join(data_dir, 'rec2.csv'),
        os.path.join(data_dir, 'rec3.csv')
    ]
    val_file = os.path.join(data_dir, 'rec4.csv')
    test_file = os.path.join(data_dir, 'rec5.csv')
    
    # 训练和评估模型
    train_and_evaluate(train_files, val_file, test_file,method='Wave-kalman')