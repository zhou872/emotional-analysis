import pandas as pd

# 读取CSV文件
df1 = pd.read_csv(r"D:\py project\qg\汽车\验证集结果.csv")
df2 = pd.read_csv(r"D:\py project\qg\汽车\有效验证集.csv")

# 根据id列进行合并，并使用df2的subject列覆盖df1的subject列
merged_df = pd.merge(df1, df2[['content_id', 'subject']], on='content_id', how='inner', suffixes=('', '_new'))

# 用df2的subject列替换df1的subject列
merged_df['subject'] = merged_df['subject_new']

# 删除辅助列
merged_df.drop(columns=['subject_new'], inplace=True)

# 保存结果到新的CSV文件
merged_df.to_csv('验证集综合结果.csv', index=False)
