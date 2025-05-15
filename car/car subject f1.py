import pandas as pd

# 读取真实数据和预测数据的CSV文件
true_data = pd.read_csv(r"D:\py project\qg\汽车\验证集.csv")
predicted_data = pd.read_csv(r"D:\py project\qg\汽车\有效验证集.csv")

# 将数据按照id分组，并将subject列转换为列表
true_data_grouped = true_data.groupby('content_id')['subject'].apply(list).reset_index(name='true_subjects')
predicted_data_grouped = predicted_data.groupby('content_id')['subject'].apply(list).reset_index(name='predicted_subjects')

# 合并真实数据和预测数据，通过id进行左连接，以确保所有真实数据都被包含
merged_data = pd.merge(true_data_grouped, predicted_data_grouped, on='content_id', how='left')

# 初始化Tp, Fp, Fn
Tp = 0
Fp = 0
Fn = 0

# 遍历合并后的数据，计算Tp, Fp, Fn
for index, row in merged_data.iterrows():
    true_subjects = set(row['true_subjects'])
    # 修改这里，先检查是否为NaN
    predicted_subjects = set(pd.Series(row['predicted_subjects']).dropna()) if row['predicted_subjects'] is not None else set()

    # 计算Tp, Fp, Fn
    Tp += len(true_subjects.intersection(predicted_subjects))
    Fp += len(predicted_subjects.difference(true_subjects))
    Fn += len(true_subjects.difference(predicted_subjects))

# 计算精确率和召回率
precision = Tp / (Tp + Fp) if (Tp + Fp) != 0 else 0
recall = Tp / (Tp + Fn) if (Tp + Fn) != 0 else 0

# 计算F1-Score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"精确率: {precision}")
print(f"召回率: {recall}")
print(f"F1-Score: {f1}")
