import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def resample_trajectory(df, interval_ms=100):
    """将轨迹重采样为固定时间间隔"""
    # 确保时间升序排列
    df = df.sort_values(by='time')
    
    # 创建新的统一时间序列
    start_time = df['time'].min()
    end_time = df['time'].max()
    new_times = np.arange(start_time, end_time+1, interval_ms)
    
    # 对每一列进行插值
    resampled_data = {'time': new_times}
    
    for column in df.columns:
        if column != 'time':
            # 创建插值函数
            interpolator = interp1d(df['time'], df[column], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            # 应用插值
            resampled_data[column] = interpolator(new_times)
    
    return pd.DataFrame(resampled_data)

import os
import glob

def process_csv_files(input_dir, output_dir, interval_ms=100):
    """处理input_dir中的所有CSV文件，重采样并保存到output_dir"""
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        # 获取文件名
        file_name = os.path.basename(file_path)
        print(f"处理 {file_name}...")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 重采样
            resampled_df = resample_trajectory(df, interval_ms)
            
            # 保存到输出目录
            output_path = os.path.join(output_dir, file_name)
            resampled_df.to_csv(output_path, index=False)
            
            print(f"成功重采样并保存到 {output_path}")
        except Exception as e:
            print(f"处理 {file_name} 时出错: {e}")
    
    print("所有文件处理完成。")

# 设置输入和输出目录
if __name__ == "__main__":
    base_dir = r"d:\Data\My_Master_Life\科研\论文\gnss轨迹预测\FDR"
    input_dir = os.path.join("data_sup\\csv3")
    output_dir = os.path.join("data_sup\\csv4")

    # 执行处理
    process_csv_files(input_dir, output_dir, interval_ms=100)