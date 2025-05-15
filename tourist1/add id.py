import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"D:\py project\qg\旅游2\旅游2测试集.csv")

# 添加ID列
df['id'] = range(1, len(df) + 1)

# 将ID列移到第一列
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# 保存修改后的CSV文件
df.to_csv(r"D:\py project\qg\旅游2\旅游2测试集.csv", index=False)
