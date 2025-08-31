import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
# 全局字体：优先使用 Times New Roman，其次使用常见衬线字体作为回退
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示
# 放大坐标轴刻度字体
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
def load_data(base_dir, methods):
    base = Path(base_dir)
    true_lon = true_lat = true_alt = None
    preds = {}
    present_methods = []
    for m in methods:
        csv_path = base / m / "predict_trajectory.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        required = [
            'true_longitude','true_latitude','true_altitude',
            'predicted_longitude','predicted_latitude','predicted_altitude'
        ]
        if not all(c in df.columns for c in required):
            print(f"[WARN] Columns missing in {csv_path}")
            continue

        if true_lon is None:
            true_lon = df['true_longitude'].to_numpy()
            true_lat = df['true_latitude'].to_numpy()
            true_alt = df['true_altitude'].to_numpy()

        preds[m] = (
            df['predicted_longitude'].to_numpy(),
            df['predicted_latitude'].to_numpy(),
            df['predicted_altitude'].to_numpy()
        )
        present_methods.append(m)

    if true_lon is None:
        raise RuntimeError("No valid predict_trajectory.csv found.")
    return (true_lon, true_lat, true_alt), preds, present_methods
def plot_comparison(base_dir, methods):
    true_xyz, preds, present = load_data(base_dir, methods)
    true_lon, true_lat, true_alt = true_xyz
    n = len(true_lon)
    idx = range(n)
    # 颜色映射：M1 紫色，M2 深蓝，M3 红色，M4 绿色， M5 青色，M6 灰色，M7 暗黄
    colors = {
        # 通用 M 系列映射
        'TCN': '#65A9D6',       # （突出）
        'M1': '#800080',  # 紫色
        'M2': '#00008B',  # 深蓝
        'M3': '#FF0000',  # 红色
        'M4': '#008000',  # 绿色
        'M5': '#00CED1',  # 青色（更偏暗的青，便于区分）
        'M6': '#808080',  # 灰色
        'M7': '#B8860B',  # 暗黄（深金色）
        # 若方法名为具体模型名，则按顺序映射为 M1~M7 的颜色
        # 'RNN': '#800080',       # M1 紫色
        # 'GRU': '#00008B',       # M2 深蓝
        # 'CNN': '#FF0000',       # M3 红色
        # 'LSTM': '#008000',      # M4 绿色
        # 'Transformer': '#00CED1', # M5 青色
        # 'BiLSTM': '#808080',    # M6 灰色
        # 'WT-TCN': '#B8860B',    # M7 暗黄
        # TCN 单独指定一个突出色

    }

    # 压缩整体高度，使子图（尤其是 2D xyz 对比图）呈矩形
    fig = plt.figure(figsize=(24, 4))
    ax3d = fig.add_subplot(1, 4, 1, projection='3d')
    ax_x = fig.add_subplot(1, 4, 2)
    ax_y = fig.add_subplot(1, 4, 3)
    ax_z = fig.add_subplot(1, 4, 4)
    # 调整视角，使 3D 图中 z 轴刻度标记出现在左侧视觉位置
    ax3d.view_init(elev=20, azim=120)
    # 坐标轴刻度字号统一放大（包含 3D 的 z 轴）
    for ax in (ax_x, ax_y, ax_z):
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        # 限制横轴主刻度数量为约 5 个，并关闭次刻度，避免过密
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.minorticks_off()
    for axis in ('x', 'y', 'z'):
        ax3d.tick_params(axis=axis, which='major', labelsize=14)
        ax3d.tick_params(axis=axis, which='minor', labelsize=12)

    # True trajectory (3D)
    true_line, = ax3d.plot(true_lon, true_lat, true_alt, color='black', lw=2.8, label='True', zorder=4)

    # True projections (仅绘制一次，避免重复)
    ax_x.plot(idx, true_lon, color='black', lw=2.4, alpha=0.85, label='True', zorder=4)
    ax_y.plot(idx, true_lat, color='black', lw=2.4, alpha=0.85, label='True', zorder=4)
    ax_z.plot(idx, true_alt, color='black', lw=2.4, alpha=0.85, label='True', zorder=4)

    # Predicted trajectories
    handles = [true_line]
    labels = ['True']
    # 先绘制非 TCN，再绘制 TCN，使 TCN 处于最上层；同时设置更高 zorder
    ordered = [m for m in methods if m in present and m != 'TCN'] + [m for m in methods if m in present and m == 'TCN']
    for m in ordered:
        is_tcn = (m == 'TCN')
        ls = '-' if is_tcn else '--'
        lw3 = 2.0 if is_tcn else 1.2
        lw2 = 1.6 if is_tcn else 1.0
        z = 5 if is_tcn else 2  # 更高的 zorder 让 TCN 叠放在上层
        plon, plat, palt = preds[m]
        line3d, = ax3d.plot(plon, plat, palt, lw=lw3, color=colors.get(m, None), linestyle=ls, label=m, alpha=0.95 if is_tcn else 0.9, zorder=z)
        # 2D projections
        ax_x.plot(idx, plon, lw=lw2, color=colors.get(m, None), linestyle=ls, alpha=0.95 if is_tcn else 0.9, zorder=z)
        ax_y.plot(idx, plat, lw=lw2, color=colors.get(m, None), linestyle=ls, alpha=0.95 if is_tcn else 0.9, zorder=z)
        ax_z.plot(idx, palt, lw=lw2, color=colors.get(m, None), linestyle=ls, alpha=0.95 if is_tcn else 0.9, zorder=z)

        handles.append(line3d)
        labels.append(m)

    # 让3D三个轴数据范围等比，避免失真
    x_all = np.concatenate([true_lon] + [preds[m][0] for m in present])
    y_all = np.concatenate([true_lat] + [preds[m][1] for m in present])
    z_all = np.concatenate([true_alt] + [preds[m][2] for m in present])
    def set_axes_equal_3d(ax, x, y, z):
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        x_mid, y_mid, z_mid = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
        max_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
        r = max_range / 2
        ax.set_xlim(x_mid - r, x_mid + r)
        ax.set_ylim(y_mid - r, y_mid + r)
        ax.set_zlim(z_mid - r, z_mid + r)
    set_axes_equal_3d(ax3d, x_all, y_all, z_all)

    # Titles and labels
    # ax3d.set_title('Comparison of Predicted and True Trajectories', fontsize=12)
    ax3d.set_xlabel('x (m)')
    ax3d.set_ylabel('y (m)')
    ax3d.set_zlabel('z (m)')

    ax_x.set_title('x (Longitude)', fontsize=16)
    ax_x.set_xlabel('Time step')
    ax_x.set_ylabel('x (m)')

    ax_y.set_title('y (Latitude)', fontsize=16)
    ax_y.set_xlabel('Time step')
    ax_y.set_ylabel('y (m)  ')

    ax_z.set_title('z (Altitude)', fontsize=16)
    ax_z.set_xlabel('Time step')
    ax_z.set_ylabel('z (m)')

    for ax in [ax_x, ax_y, ax_z]:
        ax.grid(alpha=0.3)

    # 设置子图外观：3D 维持等比例立方体，2D 设置为扁平的矩形（高度/宽度 < 1）
    ax3d.set_box_aspect((1, 1, 1))
    # 限制 y 轴主刻度数量为约 3-4 个
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax_x.set_box_aspect(0.6)
    ax_y.set_box_aspect(0.6)
    ax_z.set_box_aspect(0.6)

    # Global legend
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    fig.suptitle('Comparison of Predicted and True Trajectories', fontsize=14, y=0.98)
    # 调整底部留白，避免压缩高度后图例遮挡
    fig.subplots_adjust(bottom=0.22, wspace=0.3)

    out_path = Path(base_dir) / "predict_vs_true_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    base_dir = r"d:\\Projects\\CodeProjects\\FDR\\test_straight_spark_shuffle"
    methods = ['RNN', 'GRU', 'CNN', 'LSTM', 'Transformer', 'TCN', 'BiLSTM', 'WT-TCN']
    plot_comparison(base_dir, methods)