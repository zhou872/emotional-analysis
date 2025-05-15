import pandas as pd

# 读取两个csv文件
df1 = pd.read_csv(r"C:\Users\z'z'j\Desktop\情感分类\2021 CCF BDCI产品评论观点提取\train_data_public.csv")
df2 = pd.read_csv(r"D:\py project\qg\产品\产品实验集主题分类结果.csv")

# 合并两个数据框，使用id作为键，并指示我们想要一个外连接，这样即使某些id只出现在一个数据框中，它们也会被保留下来
merged_df = pd.merge(df1, df2, on='id', how='inner', suffixes=('_df1', '_df2'))

# 找出id相同但tag不同的行
diff_tag_df = merged_df[merged_df['tag_df1'] != merged_df['tag_df2']]

# 选择你想要的列进行输出
result_df = diff_tag_df[['id', 'text_df1', 'tag_df1', 'tag_df2']]

# 将结果重命名，便于理解
result_df.columns = ['id', 'text', 'tag_real', 'tag_predict']

# 将结果保存到一个新的csv文件中
result_df.to_csv('主题分类对比.csv', index=False)

print("完成对比，结果已保存到 different_tags.csv 文件中。")
