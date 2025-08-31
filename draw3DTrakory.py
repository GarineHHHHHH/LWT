import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import PolyCollection
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import art3d
# Configuration for Times New Roman font
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 24


def plot_trajectory_with_error_bands(csv_filepath, attacker_pos, attacker_hijack_pos, drone_takeoff_pos, drone_target_pos,save_path="trajectory_with_error_bands.png"):
    """
    绘制带有误差带的三维轨迹图，并标注特殊点。
    
    参数:
    csv_filepath (str): 包含轨迹数据的CSV文件路径。
    attacker_pos (list): 攻击者位置坐标 [x, y, z]。
    attacker_hijack_pos (list): 攻击者劫持位置坐标 [x, y, z]。
    drone_takeoff_pos (list): 无人机起飞位置坐标 [x, y, z]。
    save_path (str, optional): 保存图像的文件路径。默认为 "trajectory_with_error_bands.png"。
    """
    
    x_coords, y_coords, z_coords, x_sd, y_sd, z_sd = [], [], [], [], [], []
    
    with open(csv_filepath, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            x_coords.append(float(row['x']))
            y_coords.append(float(row['y']))
            z_coords.append(float(row['z']))
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(x_coords, y_coords, z_coords, label='Trajectory', linewidth=2)
      # 绘制在XY平面的投影
    ax.plot(x_coords, y_coords, np.zeros_like(z_coords), zdir='z', color='lightcoral', linestyle='--', linewidth=1, label='XY Projection')
      
    # 绘制以attacker_hijack_pos为圆心，半径为1000的圆
    circle = patches.Circle((attacker_pos[0], attacker_pos[1]), radius=1000, facecolor='none', edgecolor='black', linestyle='-', label='1000m Radius')
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir="z")  # 将2D图形转换为3D图形
    



    # 标注特殊点
    ax.scatter(attacker_pos[0], attacker_pos[1], attacker_pos[2], color='red', marker='x', s=100, label='Attacker Position')
    ax.scatter(attacker_hijack_pos[0], attacker_hijack_pos[1], attacker_hijack_pos[2], color='purple', marker='x', s=100, label='Hijack Position')
    ax.scatter(drone_takeoff_pos[0], drone_takeoff_pos[1], drone_takeoff_pos[2], color='green', marker='o', s=100, label='Takeoff Position')
    ax.scatter(drone_target_pos[0], drone_target_pos[1], drone_target_pos[2], color='green', marker='o', s=100, label='Takeoff Position')
    
    # Plot every 10th data point
    ax.scatter(x_coords[::200], y_coords[::200], z_coords[::200], color='blue', marker='o', s=50, label='Data Points')
    
    ax.set_xlabel('X (m)', labelpad=15)
    ax.set_ylabel('Y (m)', labelpad=15)
    ax.set_zlabel('Z (m)', labelpad=15)
    ax.set_title('Drone Trajectory with Error Bands', fontsize=24)
    # ax.legend(loc='upper left', fontsize=20)
    
    # 设置刻度朝内
    ax.tick_params(axis='x', direction='in', pad=10)
    ax.tick_params(axis='y', direction='in', pad=10)
    ax.tick_params(axis='z', direction='in', pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 示例用法
csv_filepath = os.path.join("upLab", "flight_records_stats.csv")
attacker_pos = [1800.0, 2400.0, 0.0]
attacker_hijack_pos = [1800, 3400, 0]
drone_takeoff_pos = [0, 0, 80]
drone_target_pos = [3000, 4000, 0]

plot_trajectory_with_error_bands(csv_filepath, attacker_pos, attacker_hijack_pos, drone_takeoff_pos, drone_target_pos)