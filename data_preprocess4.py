import pandas as pd
import numpy as np
import os

# 地球半径 (单位: 米)
R = 6371000

def lon_lat_to_xy(lon, lat, origin_lon, origin_lat):
    """
    将经纬度坐标转换为笛卡尔坐标
    
    参数:
        lon: 经度 (角度)
        lat: 纬度 (角度)
        origin_lon: 原点经度 (角度)
        origin_lat: 原点纬度 (角度)
    
    返回值:
        (x, y): 笛卡尔坐标 (米)
    """
    
    # 转换为弧度
    lon, lat, origin_lon, origin_lat = map(np.radians, [lon, lat, origin_lon, origin_lat])
    
    # 差值
    dlon = lon - origin_lon
    dlat = lat - origin_lat
    
    # 计算x和y
    x = R * dlon * np.cos(origin_lat)
    y = R * dlat
    
    return x, y

# 输入和输出目录
input_dir = "data_sup\\csv4"
output_dir = "data_sup\\csv5"

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
        
        # 获取第一条数据的经纬度作为原点
        origin_lon = data['product_gps_longitude'].iloc[0]
        origin_lat = data['product_gps_latitude'].iloc[0]
        
        # 转换为笛卡尔坐标
        x, y = lon_lat_to_xy(data['product_gps_longitude'], data['product_gps_latitude'], origin_lon, origin_lat)
        
        # 更新DataFrame
        data['product_gps_longitude'] = x
        data['product_gps_latitude'] = y
        
        # 构建输出路径
        output_path = os.path.join(output_dir, filename)
        
        # 保存到新的CSV文件
        data.to_csv(output_path, index=False)
        
        print(f"已保存到: {output_path}")

print("处理完成！")