import pandas as pd
from sklearn.metrics import cohen_kappa_score

# 从CSV文件中导入真实标签和预测标签
true_labels_df = pd.read_csv(r"D:\qinggan\2021 CCF BDCI产品评论观点提取\train_data_public.csv")
predicted_labels_df = pd.read_csv(r"D:\py project\qg\产品\测试集结果.csv")

# 确保两个数据集的id列对齐
true_labels_df.set_index('id', inplace=True)
predicted_labels_df.set_index('id', inplace=True)

# 找出两个数据集共有的id
common_ids = true_labels_df.index.intersection(predicted_labels_df.index)

# 提取真实标签和预测标签
true_labels = true_labels_df.loc[common_ids, 'class']
predicted_labels = predicted_labels_df.loc[common_ids, 'class']

# 计算Kappa系数
kappa = cohen_kappa_score(true_labels, predicted_labels)

# S2即为Kappa系数
S2 = kappa

print(f"S2 (Kappa系数): {S2}")
