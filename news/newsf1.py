import pandas as pd
from sklearn.metrics import f1_score

# 读取包含真实标签的CSV文件
true_labels_df = pd.read_csv(r"D:\py project\qg\新闻\测试集.csv")

# 读取包含预测标签的CSV文件
predicted_labels_df = pd.read_csv(r"D:\py project\qg\新闻\新闻测试集结果1.csv")

# 确保两个数据集的id列对齐
true_labels_df.set_index('id', inplace=True)
predicted_labels_df.set_index('id', inplace=True)

# 找出两个数据集共有的id
common_ids = true_labels_df.index.intersection(predicted_labels_df.index)

# 提取真实标签和预测标签
true_labels = true_labels_df.loc[common_ids, 'emotion_label']
predicted_labels = predicted_labels_df.loc[common_ids, 'emotion_label']

# 计算F1分数
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"F1分数: {f1}")

