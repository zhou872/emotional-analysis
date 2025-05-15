import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import cohen_kappa_score

# 读取CSV文件
real_df = pd.read_csv(r"C:\Users\z'z'j\Desktop\情感分类\2021 CCF BDCI产品评论观点提取\train_data_public.csv")
predicted_df = pd.read_csv(r"D:\py project\qg\产品\测试集产品主题分类结果.csv")

# 确保数据对齐，只保留预测文件中存在的ID
aligned_real_df = real_df[real_df['id'].isin(predicted_df['id'])]
aligned_predicted_df = predicted_df[predicted_df['id'].isin(real_df['id'])]

# 重置索引
aligned_real_df.reset_index(drop=True, inplace=True)
aligned_predicted_df.reset_index(drop=True, inplace=True)


# 提取实体标注
real_tags = aligned_real_df['tag'].tolist()
predicted_tags = aligned_predicted_df['tag'].tolist()

# 将字符串标签列表转换为适合计算的格式
# 这里假设标签是用空格分隔的字符串
real_tags = [tags.split() for tags in real_tags]
predicted_tags = [tags.split() for tags in predicted_tags]

# 使用MultiLabelBinarizer将列表转换为二进制形式
mlb = MultiLabelBinarizer()
real_tags_bin = mlb.fit_transform(real_tags)
predicted_tags_bin = mlb.transform(predicted_tags)

# 计算精确率、召回率和F1值
precision = f1_score(real_tags_bin, predicted_tags_bin, average='micro')
recall = f1_score(real_tags_bin, predicted_tags_bin, average='micro')
f1 = f1_score(real_tags_bin, predicted_tags_bin, average='micro')


print(f"精确率: {precision}")
print(f"召回率: {recall}")
print(f"F1值: {f1}")


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

S1 = f1

# 计算综合评分 S
S = 0.5 * S1 + 0.5 * S2

print(f"综合评分 S: {S}")

