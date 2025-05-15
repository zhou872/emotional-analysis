import pandas as pd
from sklearn.metrics import f1_score

# 读取包含真实标签的CSV文件
true_labels_df = pd.read_csv(r"文件路径")

# 读取包含预测标签的CSV文件
predicted_labels_df = pd.read_csv(r"文件路径")

# 确保两个数据集的id列对齐，使用content_id和subject共同作为索引
true_labels_df.set_index(['content_id', 'subject'], inplace=True)
predicted_labels_df.set_index(['content_id', 'subject'], inplace=True)

# 找出两个数据集共有的索引
common_indices = true_labels_df.index.intersection(predicted_labels_df.index)

# 提取真实标签和预测标签
true_labels = true_labels_df.loc[common_indices, 'sentiment_value']
predicted_labels = predicted_labels_df.loc[common_indices, 'sentiment_value']

# 计算F1分数
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"F1分数: {f1}")
