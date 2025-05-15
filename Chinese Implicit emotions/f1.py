import pandas as pd
from sklearn.metrics import f1_score

# 读取包含真实标签的CSV文件
true_labels_df = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\实验集.csv")

# 读取包含预测标签的CSV文件
predicted_labels_df = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\实验集结果.csv")

# 确保两个数据集的id列对齐，使用ID和ID2共同作为索引
true_labels_df.set_index(['ID', 'ID2'], inplace=True)
predicted_labels_df.set_index(['ID', 'ID2'], inplace=True)

# 找出两个数据集共有的索引
common_indices = true_labels_df.index.intersection(predicted_labels_df.index)

# 提取真实标签和预测标签
true_labels = true_labels_df.loc[common_indices, 'label']
predicted_labels = predicted_labels_df.loc[common_indices, 'label']

# 过滤掉真实标签为-1的样本
filtered_indices = true_labels[true_labels != -1].index
true_labels = true_labels.loc[filtered_indices]
predicted_labels = predicted_labels.loc[filtered_indices]

# 计算F1分数
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"F1分数: {f1}")
