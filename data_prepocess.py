import pandas as pd
import numpy as np
import os

# 输入和输出目录
input_dir = "csv5"
output_dir = "csv6"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有CSV文件
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        print(f"处理文件: {filename}")
        
        # 读取CSV文件
        input_path = os.path.join(input_dir, filename)
        data = pd.read_csv(input_path)
        
        # 找到起飞前的最后一条数据
        takeoff_index = data[(data['altitude'] != 0)].index
        if len(takeoff_index) > 0:
            takeoff_index = takeoff_index[0]
            
            # 删除起飞前speed_vx, speed_vy, speed_vz都为0的数据,保留最后一条
            zero_speed_indices = data[
                (data['speed_vx'] == 0) & (data['speed_vy'] == 0) & (data['speed_vz'] == 0)
            ].index
            
            delete_indices = zero_speed_indices[zero_speed_indices < takeoff_index]
            if len(delete_indices) > 0:
                delete_indices = delete_indices[:-1]  # Keep the last one
                data = data.drop(delete_indices)
        
        # 找到着陆后的第一条数据
        landing_indices = data[(data['altitude'] == 0)].index
        if len(landing_indices) > 0:
            landing_index = landing_indices[0]
            
            # 删除着陆后speed_vx, speed_vy, speed_vz都为0的数据
            zero_speed_indices = data[
                (data['speed_vx'] == 0) & (data['speed_vy'] == 0) & (data['speed_vz'] == 0)
            ].index
            
            delete_indices = zero_speed_indices[zero_speed_indices > landing_index]
            data = data.drop(delete_indices)
        
        # 重置索引
        data = data.reset_index(drop=True)
        
        # 构建输出路径
        output_path = os.path.join(output_dir, filename)
        
        # 保存到新的CSV文件
        data.to_csv(output_path, index=False)
        
        print(f"已保存到: {output_path}")

print("处理完成！")