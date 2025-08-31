import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(file_path, output_dir):
    """
    对单个CSV文件进行预处理，包括填充缺失值、选择特征和目标，以及归一化，并将结果保存到新的CSV文件。
    
    参数:
        file_path (str): CSV文件的路径。
        output_dir (str): 输出目录的路径。
    
    返回值:
        None
    """
    data = pd.read_csv(file_path)
    # 处理缺失值
    data = data.fillna(0)
    # 选择特征和目标
    features = data.drop(['product_gps_longitude', 'product_gps_latitude', 'altitude'], axis=1)
    targets = data[['product_gps_longitude', 'product_gps_latitude', 'altitude']]
    # 归一化
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)
    
    # 创建DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_targets_df = pd.DataFrame(scaled_targets, columns=targets.columns)
    
    # 合并特征和目标
    scaled_data = pd.concat([scaled_features_df, scaled_targets_df], axis=1)
    
    # 构建输出路径
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    
    # 保存到新的CSV文件
    scaled_data.to_csv(output_path, index=False)
    
    print(f"已保存到: {output_path}")

# 主程序
if __name__ == "__main__":
    input_dir = "csv6"
    output_dir = "csv7"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录中的所有CSV文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            print(f"处理文件: {filename}")
            
            # 构建完整的文件路径
            file_path = os.path.join(input_dir, filename)
            
            # 预处理数据并保存
            preprocess_data(file_path, output_dir)
            
            print(f"{filename} 处理完成！")
    
    print("所有文件处理完成！")