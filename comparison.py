import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\测试集.csv")
df2 = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\测试集结果.csv")

# 根据id列对齐两个数据框
merged_df = pd.merge(df1, df2, on='id', suffixes=('_答案', '_模型'))

# 找出emotion_label列值不同的行
diff_df = merged_df[merged_df['label'] != merged_df['emotion']]

# 提取需要的列，并添加到结果数据框中id,class,text
result_df = diff_df[['id','label','emotion','review_答案']].copy()





# 将结果保存到新的CSV文件中
result_df.to_csv(r'2对比结果.csv', index=False)

print("对比完成，结果已保存到'对比结果.csv'文件中。")
