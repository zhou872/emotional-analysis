import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\SMP2019_ECISA_Train.csv")

# 将label列的空值替换为-1
df['label'] = df['label'].fillna(-1)

# 将label列的数据类型转换为整数
df['label'] = df['label'].astype(int)


# 保存修改后的CSV文件
df.to_csv(r"D:\py project\qg\中文隐式情感分析\SMP2019_ECISA_Train.csv", index=False)
