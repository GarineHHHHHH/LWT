import pandas as pd
import numpy as np
import os

# 输入和输出目录
input_dir = "data_sup\\csv5"
output_dir = "data_sup\\csv6"  # 直接修改源数据，所以输入输出目录一致

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
        
        # 创建新的time列
        data['time'] = range(1, len(data) + 1)
        
        # 构建输出路径
        output_path = os.path.join(output_dir, filename)
        
        # 保存到CSV文件，覆盖原有文件
        data.to_csv(output_path, index=False)
        
        print(f"已保存到: {output_path}")

print("处理完成！")