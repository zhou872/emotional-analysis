import pandas as pd

# 读取CSV文件
data = pd.read_csv(r"D:\py project\qg\产品\产品训练集.csv")

# 确保数据集的行数大于200
assert len(data) > 200, "数据集的行数必须大于200"

# 确保emotion_label列存在
assert 'class' in data.columns, "数据集中缺少emotion_label列"

# 计算每层需要的样本数量
n_samples_per_group = 100 // len(data['class'].unique())

# 分层抽样：根据emotion_label列的值进行抽样
first_sample = data.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples_per_group)))

# 计算剩余数据集
remaining_data = data.drop(first_sample.index)

# 重新计算每层需要的样本数量，以确保总共200条
n_samples_per_group_second = 100 // len(remaining_data['class'].unique())

# 对剩余的数据集进行二次分层抽样
second_sample = remaining_data.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples_per_group_second)))

# 保存样本数据集为CSV文件
first_sample.to_csv('产品试验集.csv', index=False)
second_sample.to_csv('产品测试集.csv', index=False)

print("结果已保存")
