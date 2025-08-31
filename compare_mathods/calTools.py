import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def create_sequences(data, seq_length):
    """创建时序序列"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, -3:])  # 目标是最后三个特征
    return np.array(X), np.array(y)

def preprocess_data(file_path):
    """
    对单个CSV文件进行预处理，包括填充缺失值、选择特征和目标，以及归一化。
    
    参数:
        file_path (str): CSV文件的路径。
    
    返回值:
        tuple: 包含归一化后的特征、归一化后的目标、特征的MinMaxScaler对象和目标的MinMaxScaler对象。
    """
    data = pd.read_csv(file_path)
    # 处理缺失值
    data = data.fillna(0)
    # 选择特征和目标
    features = data.drop(['product_gps_longitude', 'product_gps_latitude', 'altitude'], axis=1)
    # 特征列是前12个特征
    features = features.iloc[:, :12]
    targets = data[['product_gps_longitude', 'product_gps_latitude', 'altitude']]
    # 归一化
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)
    
    # 合并特征和目标
    scaled_data = np.concatenate((scaled_features, scaled_targets), axis=1)
    
    return scaled_data, feature_scaler, target_scaler

def calculate_and_save_metrics(true_targets, predicted_targets, method, output_path='results',filename="metrics.csv"):
    """
    计算评估指标并将结果保存到CSV文件中。
    
    参数:
        true_targets (numpy array): 真实的目标值。
        predicted_targets (numpy array): 预测的目标值。
        method (str): 使用的方法名称，用于创建输出目录。
        filename (str): 保存指标的CSV文件名。
    
    返回值:
        None
    """
    # 1. 计算整体路径的指标
    rmse = np.sqrt(mean_squared_error(true_targets, predicted_targets))
    mae = mean_absolute_error(true_targets, predicted_targets)
    
    # MDE (Mean Deviation Error) - 三维欧氏距离
    euclidean_distances = np.sqrt(np.sum((true_targets - predicted_targets)**2, axis=1))
    mde = np.mean(euclidean_distances)
    lat_mde = np.mean(np.abs(true_targets[:, 0] - predicted_targets[:, 0]))  # 经度平均偏差
    lon_mde = np.mean(np.abs(true_targets[:, 1] - predicted_targets[:, 1]))  # 纬度平均偏差
    alt_mde = np.mean(np.abs(true_targets[:, 2] - predicted_targets[:, 2]))  # 高度平均偏差
        
    # MRE (Mean Relative Error)
    relative_errors = np.abs((true_targets - predicted_targets) / true_targets)
    mre = np.mean(relative_errors[~np.isinf(relative_errors) & ~np.isnan(relative_errors)])  # 排除inf和NaN值
    
    # 2. 计算每个目标值的指标
    lat_rmse = np.sqrt(mean_squared_error(true_targets[:, 0], predicted_targets[:, 0]))
    lat_mae = mean_absolute_error(true_targets[:, 0], predicted_targets[:, 0])
    
    lat_mre = np.mean(np.abs((true_targets[:, 0] - predicted_targets[:, 0]) / true_targets[:, 0]))
    
    lon_rmse = np.sqrt(mean_squared_error(true_targets[:, 1], predicted_targets[:, 1]))
    lon_mae = mean_absolute_error(true_targets[:, 1], predicted_targets[:, 1])
    
    lon_mre = np.mean(np.abs((true_targets[:, 1] - predicted_targets[:, 1]) / true_targets[:, 1]))
    
    alt_rmse = np.sqrt(mean_squared_error(true_targets[:, 2], predicted_targets[:, 2]))
    alt_mae = mean_absolute_error(true_targets[:, 2], predicted_targets[:, 2])
   
    alt_mre = np.abs((true_targets[:, 2] - predicted_targets[:, 2])) if np.any(true_targets[:, 2] == 0) else np.mean(np.abs((true_targets[:, 2] - predicted_targets[:, 2]) / true_targets[:, 2]))
    alt_mre = np.mean(alt_mre[~np.isinf(alt_mre) & ~np.isnan(alt_mre)])  # 排除inf和NaN值
    # 3. 创建DataFrame
    metrics_df = pd.DataFrame({
        'rmse': [rmse],
        'mae': [mae],
        'mde': [mde],
        'mre': [mre],
        'lat_rmse': [lat_rmse],
        'lat_mae': [lat_mae],
        'lat_mde': [lat_mde],
        'lat_mre': [lat_mre],
        'lon_rmse': [lon_rmse],
        'lon_mae': [lon_mae],
        'lon_mde': [lon_mde],
        'lon_mre': [lon_mre],
        'alt_rmse': [alt_rmse],
        'alt_mae': [alt_mae],
        'alt_mde': [alt_mde],
        'alt_mre': [alt_mre]
    })
    print(metrics_df)
    # 4. 确保输出目录存在
    output_dir =  os.path.join(output_path,method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 5. 保存到CSV文件
    metrics_df.to_csv(os.path.join(output_dir, filename), index=False)
def save_results_to_csv(true_targets, predicted_targets,inference_times, method,output_path='results', filename="predict_trajectory.csv"):
    """保存测试结果到CSV"""
    results_df = pd.DataFrame({
        'true_longitude': true_targets[:, 0],
        'true_latitude': true_targets[:, 1],
        'true_altitude': true_targets[:, 2],
        'predicted_longitude': predicted_targets[:, 0],
        'predicted_latitude': predicted_targets[:, 1],
        'predicted_altitude': predicted_targets[:, 2],
        'time_cost': inference_times
    })
    
    # 确保输出目录存在
    output_dir =  os.path.join(output_path,method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df.to_csv(os.path.join(output_dir, filename), index=False)
def visualize_results(true_targets, predicted_targets, method, output_path='results',filename="test_results.png"):
    """绘制测试结果"""
    plt.figure(figsize=(12, 6))
    
    # 绘制经纬度
    plt.subplot(1, 2, 1)
    plt.plot(true_targets[:, 0], true_targets[:, 1], label='True Trajectory')
    plt.plot(predicted_targets[:, 0], predicted_targets[:, 1], label='Predicted Trajectory')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('2D Trajectory')
    plt.legend()
    
    # 绘制高度
    plt.subplot(1, 2, 2)
    plt.plot(true_targets[:, 2], label='True Altitude')
    plt.plot(predicted_targets[:, 2], label='Predicted Altitude')
    plt.xlabel('Time Step')
    plt.ylabel('Altitude')
    plt.title('Altitude vs. Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,method,filename))
    plt.close()




def visulize_loss(train_losses, val_losses, method, output_path='results', filename="training_loss.png"):
    """绘制训练损失"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    # 确保输出目录存在
    output_dir =  os.path.join(output_path,method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir,filename))
    plt.close()
    
def save_loss_to_csv(train_losses, val_losses, method, output_path='results',filename="losses.csv"):
    """保存损失到CSV"""
    losses_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    
    # 确保输出目录存在
    output_dir =  os.path.join(output_path,method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    losses_df.to_csv(os.path.join(output_dir,filename), index=False)
    
# Function to visualize attention weights as a heatmap
def visualize_attention(attention_weights, title="Attention Weights"):
    """
    Visualizes attention weights as a heatmap.

    Args:
        attention_weights (numpy.ndarray): Attention weights to visualize.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights.reshape(1, -1), cmap="viridis", annot=True, fmt=".2f", cbar=True)
    plt.title(title)
    plt.xlabel("Sequence Timesteps")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    # plt.show()
    plt.savefig(title + ".png", dpi=300)
    plt.close()
    
    # 将attention_weights保存到CSV文件
    attention_df = pd.DataFrame(attention_weights)
    attention_df.to_csv(title + ".csv", index=False, header=False)
    print(f"Attention weights saved to {title}.csv")