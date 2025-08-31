from train_evaluate import train_and_evaluate,just_test
import os


data_dir = "data_sup/csv6"
train_files = [
    os.path.join(data_dir, '2-spark.csv'),
    os.path.join(data_dir, '3-spark.csv'),
    os.path.join(data_dir, '3-triangle.csv'),
    os.path.join(data_dir, '4-triangle.csv'),
    os.path.join(data_dir, '5-rectangle.csv'),
    os.path.join(data_dir, '6-rectangle.csv')
]
# val_files = os.path.join(data_dir, '7-rectangle.csv')
val_file = os.path.join(data_dir, '7-rectangle.csv')
test_file = os.path.join(data_dir, 'straight_spark_shuffle.csv')
methods = ['RNN', 'GRU', 'CNN', 'LSTM','Transformer','TCN','BiLSTM','TCN','WT-TCN']
# methods = ['TCN', 'WT-TCN']
# methods = ['BiLSTM']
# methods = ['AT-LSTM']
for method in methods:
    # 训练和评估模型
    # train_and_evaluate(train_files, val_file, test_file,epochs=30, method=method)
    just_test(test_file, method=method)
