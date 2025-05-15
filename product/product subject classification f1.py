import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# 读取CSV文件
real_df = pd.read_csv(r"C:\Users\z'z'j\Desktop\情感分类\2021 CCF BDCI产品评论观点提取\train_data_public.csv")
predicted_df = pd.read_csv(r"D:\py project\qg\产品\产品实验集主题分类结果.csv")

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
