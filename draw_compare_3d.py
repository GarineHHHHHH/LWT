import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global font settings: prefer Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False


def load_data(base_dir: Path, methods):
    """Load true and predicted trajectories from method subfolders.

    Returns (true_lon, true_lat, true_alt), preds(dict method->(lon,lat,alt)), present_methods
    """
    true_lon = true_lat = true_alt = None
    preds = {}
    present = []

    for m in methods:
        csv_path = base_dir / m / 'predict_trajectory.csv'
        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        req = ['true_longitude','true_latitude','true_altitude',
               'predicted_longitude','predicted_latitude','predicted_altitude']
        if not all(c in df.columns for c in req):
            print(f"[WARN] Required columns missing in: {csv_path}")
            continue

        if true_lon is None:
            true_lon = df['true_longitude'].to_numpy()
            true_lat = df['true_latitude'].to_numpy()
            true_alt = df['true_altitude'].to_numpy()

        preds[m] = (
            df['predicted_longitude'].to_numpy(),
            df['predicted_latitude'].to_numpy(),
            df['predicted_altitude'].to_numpy(),
        )
        present.append(m)

    if true_lon is None:
        raise RuntimeError('No valid predict_trajectory.csv found in given methods.')
    return (true_lon, true_lat, true_alt), preds, present


def set_axes_equal_3d(ax, x, y, z):
    """Make 3D axes have equal scale for x/y/z."""
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    x_mid, y_mid, z_mid = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    r = max_range / 2
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)


def plot_3d_comparison(base_dir: Path, methods):
    (true_lon, true_lat, true_alt), preds, present = load_data(base_dir, methods)

    # Colors (consistent with earlier requests):
    colors = {
        'RNN': '#800080',       # purple
        'GRU': '#00008B',       # dark blue
        'CNN': '#FF0000',       # red
        'LSTM': '#008000',      # green
        'Transformer': '#00CED1',  # dark cyan
        'BiLSTM': '#808080',    # gray
        'WT-TCN': '#B8860B',    # dark goldenrod
        'TCN': '#65A9D6',       # highlighted blue
    }

    fig = plt.figure(figsize=(8.5, 7.5))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # Make z-axis ticks appear on left visually by adjusting view
    ax3d.view_init(elev=20, azim=120)

    # Plot True (thicker)
    ax3d.plot(true_lon, true_lat, true_alt, color='black', lw=2.8, label='True')

    # Plot predictions: TCN solid + thicker; others dashed
    for m in methods:
        if m not in present:
            continue
        plon, plat, palt = preds[m]
        is_tcn = (m == 'TCN')
        ls = '-' if is_tcn else '--'
        lw = 2.0 if is_tcn else 1.2
        ax3d.plot(plon, plat, palt, lw=lw, linestyle=ls, color=colors.get(m, None), label=m, alpha=0.95 if is_tcn else 0.9)

    # Equalize axes scale
    x_all = np.concatenate([true_lon] + [preds[m][0] for m in present])
    y_all = np.concatenate([true_lat] + [preds[m][1] for m in present])
    z_all = np.concatenate([true_alt] + [preds[m][2] for m in present])
    set_axes_equal_3d(ax3d, x_all, y_all, z_all)

    # Labels and legend
    ax3d.set_xlabel('x (m)')
    ax3d.set_ylabel('y (m)')
    ax3d.set_zlabel('z (m)')
    ax3d.set_box_aspect((1, 1, 1))
    # ax3d.legend(loc='best', frameon=False)

    out_path = base_dir / 'predict_vs_true_3d.png'
    fig.tight_layout()
    fig.show()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot 3D trajectory comparison for multiple methods')
    parser.add_argument('--base', type=str, default='test_rectangular_up_up', help='Base directory containing method subfolders')
    parser.add_argument('--methods', type=str, nargs='*', default=['RNN','GRU','CNN','LSTM','Transformer','TCN','BiLSTM','WT-TCN'], help='Method folder names to include')
    args = parser.parse_args()

    base_dir = Path(args.base)
    plot_3d_comparison(base_dir, args.methods)


if __name__ == '__main__':
    main()
