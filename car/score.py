import pandas as pd

# 读取CSV文件
predictions = pd.read_csv(r"D:\py project\qg\汽车\验证集综合结果.csv")
truth = pd.read_csv(r"C:\Users\z'z'j\Desktop\情感分类\CCF-BDCI-2018-Car Reviews Sentiment Competition (汽车行业用户观点主题及情感识别)\train_2.csv")

# 初始化计数器
Tp = 0
Fp = 0
Fn = 0

# 创建一个字典来存储真实标签
truth_dict = {}
for _, row in truth.iterrows():
    key = (row['content_id'], row['subject'])
    if key not in truth_dict:
        truth_dict[key] = []
    truth_dict[key].append(row['sentiment_value'])

# 创建一个集合来存储预测结果的键
prediction_keys = set()
for _, row in predictions.iterrows():
    key = (row['content_id'], row['subject'])
    prediction_keys.add(key)

# 找到两个文件中共有的键
common_keys = set(truth_dict.keys()).intersection(prediction_keys)

# 遍历共有的键
for key in common_keys:
    # 获取真实标签中的情感值
    true_sentiments = truth_dict[key]
    # 获取预测结果中的情感值
    pred_sentiments = predictions[(predictions['content_id'] == key[0]) & (predictions['subject'] == key[1])][
        'sentiment_value'].tolist()

    # 比较真实标签和预测结果
    for sentiment in pred_sentiments:
        if sentiment in true_sentiments:
            Tp += 1
            true_sentiments.remove(sentiment)
        else:
            Fp += 1

    # 计算漏判的数量
    Fn += len(true_sentiments)

# 计算准确率（P）和召回率（R）
P = Tp / (Tp + Fp) if (Tp + Fp) != 0 else 0
R = Tp / (Tp + Fn) if (Tp + Fn) != 0 else 0

# 计算F1-Score
F1 = (2 * P * R) / (P + R) if (P + R) != 0 else 0

print(f'Tp: {Tp}, Fp: {Fp}, Fn: {Fn}')
print(f'准确率 (P): {P}')
print(f'召回率 (R): {R}')
print(f'F1-Score: {F1}')