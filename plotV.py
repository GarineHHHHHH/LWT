import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # 确保数学公式也使用Times New Roman的风格


def plot_velocity_components(csv_directory):
    """
    遍历指定目录下的所有CSV文件，并根据speed_vx, speed_vy, speed_vz数据绘制三个方向速度的点线变化图在同一张图中，使用不同颜色区分。

    Args:
        csv_directory (str): 包含CSV文件的目录路径。
    """

    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_directory, filename)
            print(f"正在处理文件: {filepath}")

            try:
                # 读取CSV数据
                time = []
                speed_vx = []
                speed_vy = []
                speed_vz = []

                with open(filepath, 'r', encoding='utf-8') as csvfile:  # 确保使用正确的编码
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            time.append(float(row['time']))
                            speed_vx.append(float(row['speed_vx']))
                            speed_vy.append(float(row['speed_vy']))
                            speed_vz.append(float(row['speed_vz']))
                        except (KeyError, ValueError) as e:
                            print(f"警告: 文件 {filename} 中存在缺失或无效数据行。跳过该行。错误信息: {e}")
                            continue  # 跳过当前行

                # 转换为NumPy数组
                time = np.array(time)
                speed_vx = np.array(speed_vx)
                speed_vy = np.array(speed_vy)
                speed_vz = np.array(speed_vz)

                # 创建图
                fig, ax = plt.subplots(figsize=(10, 8))  # 创建一个图和一个坐标轴对象
                fig.suptitle(f'Velocity Components from {os.path.splitext(filename)[0]}', fontsize=16)  # 设置总标题

                # 绘制 speed_vx, 使用蓝色
                ax.plot(time, speed_vx, marker='o', linestyle='-', markersize=2, label='speed_vx', color='blue')

                # 绘制 speed_vy, 使用绿色
                ax.plot(time, speed_vy, marker='s', linestyle='--', markersize=2, label='speed_vy', color='green')

                # 绘制 speed_vz, 使用红色
                ax.plot(time, speed_vz, marker='^', linestyle=':', markersize=2, label='speed_vz', color='red')

                # 设置坐标轴标签
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Speed', fontsize=12)  # 统一y轴标签

                # 设置刻度字体大小
                ax.tick_params(axis='x', labelsize=10, direction='in')
                ax.tick_params(axis='y', labelsize=10, direction='in')

                # 显示图例
                ax.legend()

                # 调整子图之间的间距
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整tight_layout以适应suptitle

                # 显示图形
                plt.show()
                # plt.savefig(os.path.join(csv_directory, f'{os.path.splitext(filename)[0]}_velocity_components.png'), dpi=300, bbox_inches='tight')

            except FileNotFoundError:
                print(f"错误: 文件 {filepath} 未找到。")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")


#  使用示例
if __name__ == "__main__":
    csv_directory = "csv4"  # 替换为你的CSV文件所在的目录
    plot_velocity_components(csv_directory)