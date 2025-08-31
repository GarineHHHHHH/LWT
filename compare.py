import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from compare_mathods.kalman import kalman_filter_prediction
from compare_mathods.lstm import lstm_prediction
from compare_mathods.cnn import cnn_prediction
from compare_mathods.rnn import rnn_prediction
# 加载所有CSV文件
data_dir = "csv5"
csv_files = [
    '2-spark.csv', '3-spark.csv', '3-triangle.csv', '4-trangle.csv', 
    '5-rectangle.csv', '6-rectangle.csv', '7-rectangle.csv', '8-rectangle.csv', '9-rectangle.csv'
]

# 结果存储
results = {
    'kalman': {'rmse': [], 'mae': []},
    'lstm': {'rmse': [], 'mae': []},
    'cnn': {'rmse': [], 'mae': []},
    'rnn': {'rmse': [], 'mae': []}
}

for file in csv_files:
    print(f"处理文件: {file}")
    data = pd.read_csv(os.path.join(data_dir, file))
    
    # 1. Kalman滤波
    print("运行Kalman滤波...")
    pred_kalman, true_kalman = kalman_filter_prediction(data)
    rmse_kalman = np.sqrt(mean_squared_error(true_kalman.reshape(-1, 3), pred_kalman.reshape(-1, 3)))
    mae_kalman = mean_absolute_error(true_kalman.reshape(-1, 3), pred_kalman.reshape(-1, 3))
    results['kalman']['rmse'].append(rmse_kalman)
    results['kalman']['mae'].append(mae_kalman)
    
    # 2. LSTM
    print("运行LSTM...")
    _, _, true_lstm, pred_lstm = lstm_prediction(data)
    rmse_lstm = np.sqrt(mean_squared_error(true_lstm, pred_lstm))
    mae_lstm = mean_absolute_error(true_lstm, pred_lstm)
    results['lstm']['rmse'].append(rmse_lstm)
    results['lstm']['mae'].append(mae_lstm)
    
    # 3. CNN
    print("运行CNN...")
    _, _, true_cnn, pred_cnn = cnn_prediction(data)
    rmse_cnn = np.sqrt(mean_squared_error(true_cnn, pred_cnn))
    mae_cnn = mean_absolute_error(true_cnn, pred_cnn)
    results['cnn']['rmse'].append(rmse_cnn)
    results['cnn']['mae'].append(mae_cnn)
    
    # 4. RNN
    print("运行RNN...")
    _, _, true_rnn, pred_rnn = rnn_prediction(data)
    rmse_rnn = np.sqrt(mean_squared_error(true_rnn, pred_rnn))
    mae_rnn = mean_absolute_error(true_rnn, pred_rnn)
    results['rnn']['rmse'].append(rmse_rnn)
    results['rnn']['mae'].append(mae_rnn)
    
    # 可视化轨迹预测结果
    plt.figure(figsize=(12, 8))
    
   # 可视化轨迹预测结果
    plt.figure(figsize=(12, 8))
    
    # 经度-纬度图
    plt.subplot(2, 2, 1)
    plt.plot(true_kalman[:, 0], true_kalman[:, 1], 'b-', label='Truth')
    plt.plot(pred_kalman[:, 0], pred_kalman[:, 1], 'r--', label='Kalman')
    plt.plot(pred_lstm[:, 0], pred_lstm[:, 1], 'g--', label='LSTM')
    plt.legend()
    plt.title('Trajectory prediction')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    
    # 高度图
    plt.subplot(2, 2, 2)
    plt.plot(true_kalman[:, 2], 'b-', label='Truth')
    plt.plot(pred_kalman[:, 2], 'r--', label='Kalman')
    plt.plot(pred_lstm[:, 2], 'g--', label='LSTM')
    plt.legend()
    plt.title('Altitude prediction')
    plt.xlabel('Time step')
    plt.ylabel('Altitude(m)')
    
    # 误差比较
    plt.subplot(2, 2, 3)
    methods = ['Kalman', 'LSTM', 'CNN', 'RNN']
    rmse_values = [rmse_kalman, rmse_lstm, rmse_cnn, rmse_rnn]
    plt.bar(methods, rmse_values)
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    
    plt.subplot(2, 2, 4)
    mae_values = [mae_kalman, mae_lstm, mae_cnn, mae_rnn]
    plt.bar(methods, mae_values)
    plt.title('MAE Comparison')
    plt.ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig(f'prediction_results_{file.split(".")[0]}.png')
    plt.close()

# 打印平均结果
print("\n平均性能指标:")
for method in results:
    avg_rmse = np.mean(results[method]['rmse'])
    avg_mae = np.mean(results[method]['mae'])
    print(f"{method}: RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")