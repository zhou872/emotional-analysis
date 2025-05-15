import pandas as pd
import re  # 导入正则表达式库

# 读取CSV文件
data = pd.read_csv(r"D:\qinggan\2019互联网新闻情感分析比赛\Train_DataSet.csv")

# 预处理新闻内容
def preprocess_news(news):
    # 将新闻内容转化为字符串
    news = str(news)
    # 去除HTML标签
    news = re.sub(r'<.*?>', '', news)
    # 去除URL链接
    news = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$,]|%[0-9a-fA-F][0-9a-fA-F])+', '', news)
    # 去除广告信息
    news = re.sub(r'-\s*end\s*-\s*下面的内容你可能会喜欢↓↓↓.*', '', news, flags=re.DOTALL)
    # 去除多余符号
    news = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：“”‘’]', '', news)
    return news

# 应用预处理函数到content列
data['content'] = data['content'].apply(preprocess_news)

# 保存预处理后的数据集到新的CSV文件，只保留id、title和预处理后的content
data[['id', 'title', 'content']].to_csv('preprocessed_data.csv', index=False)

print("预处理后的测试数据已保存到preprocessed_data.csv")
