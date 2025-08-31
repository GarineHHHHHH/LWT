import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'  # 确保数学公式也使用Times New Roman的风格

def plot_3d_trajectory_from_csv(csv_directory):
    """
    遍历指定目录下的所有CSV文件，并根据经纬度和高度数据绘制3D点线轨迹图。

    Args:
        csv_directory (str): 包含CSV文件的目录路径。
    """

    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_directory, filename)
            print(f"正在处理文件: {filepath}")

            try:
                # 读取CSV数据
                longitude = []
                latitude = []
                altitude = []

                with open(filepath, 'r', encoding='utf-8') as csvfile:  # 确保使用正确的编码
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            longitude.append(float(row['product_gps_longitude']))
                            latitude.append(float(row['product_gps_latitude']))
                            altitude.append(float(row['altitude']))
                        except (KeyError, ValueError) as e:
                            print(f"警告: 文件 {filename} 中存在缺失或无效数据行。跳过该行。错误信息: {e}")
                            continue # 跳过当前行

                # 转换为NumPy数组
                longitude = np.array(longitude)
                latitude = np.array(latitude)
                altitude = np.array(altitude)

                # 创建3D图
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # 绘制轨迹
                ax.plot(longitude, latitude, altitude, marker='o', linestyle='-', markersize=3, alpha=0.6)
                # 标注第一个点为绿色
                ax.plot(longitude[0], latitude[0], altitude[0], marker='o', markersize=7, color='green', label='Start', alpha=0.6)

                # 标注最后一个点为红色
                ax.plot(longitude[-1], latitude[-1], altitude[-1], marker='o', markersize=7, color='red', label='End', alpha=0.6)

                # 设置坐标轴标签
                ax.set_xlabel('Longitude', fontsize=12)  # 设置X轴标签字体大小
                ax.set_ylabel('Latitude', fontsize=12)   # 设置Y轴标签字体大小
                ax.set_zlabel('Altitude', fontsize=12)   # 设置Z轴标签字体大小

                # 设置标题
                ax.set_title(f'3D Trajectory from {os.path.splitext(filename)[0]}', fontsize=14) # 设置标题字体大小

                # 设置刻度字体大小和方向
                ax.tick_params(axis='x', labelsize=12, direction='in')
                ax.tick_params(axis='y', labelsize=12, direction='in')
                ax.tick_params(axis='z', labelsize=12, direction='in')

               # 保存图片
                plt.savefig(f"3D_Trajectory_{os.path.splitext(filename)[0]}.png", dpi=300, bbox_inches='tight')
                # 显示图形
                # plt.show()

            except FileNotFoundError:
                print(f"错误: 文件 {filepath} 未找到。")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")


#  使用示例
if __name__ == "__main__":
    csv_directory = "csv5"  # 替换为你的CSV文件所在的目录
    plot_3d_trajectory_from_csv(csv_directory)