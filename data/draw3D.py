import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
df = pd.read_csv('data/wave-csv2/tra8.csv')

# 提取三维坐标
x = df['product_gps_longitude'].values
y = df['product_gps_latitude'].values
z = df['altitude'].values

# 创建三维图
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制轨迹
ax.plot(x, y, z, label='3D Trajectory', color='b')

# 设置比例
ax.set_box_aspect([16, 9, 9])  # 长宽高比例

# 设置标签
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')
ax.set_title('3D Trajectory (16:9:9)')

plt.legend()
plt.tight_layout()
plt.savefig('3d_trajectory.png', dpi=300)  # 保存图像到文件