import pandas as pd

# 读取原始的CSV文件
df = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\实验集结果.csv")

# 创建一个ID2列，用于区分ID列相同的数据
df['ID2'] = df.groupby('ID').cumcount() + 1

# 将修改后的DataFrame保存到新的CSV文件
df.to_csv(r"D:\py project\qg\中文隐式情感分析\实验集结果.csv", index=False)
