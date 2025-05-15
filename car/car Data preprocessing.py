import pandas as pd
import re


# 读取CSV文件
data = pd.read_csv(r"D:\qinggan\CCF-BDCI-2018-Car Reviews Sentiment Competition (汽车行业用户观点主题及情感识别)\train_2.csv")

# 预处理新闻内容
def clean(content):
    # 将新闻内容转化为字符串
    content = str(content)
    # 去除HTML标签
    content = re.sub(r'<.*?>', '', content)
    # 去除URL链接
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$,]|%[0-9a-fA-F][0-9a-fA-F])+', '', content)
    # 去除广告信息
    content = re.sub(r'-\s*end\s*-\s*下面的内容你可能会喜欢↓↓↓.*', '', content, flags=re.DOTALL)
    # 去除多余符号
    content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：“”‘’]', '', content)
    return content

# 应用预处理函数到content列
data['content'] = data['content'].apply(clean)

# 保存预处理后的数据集到新的CSV文件，只保留id、title和预处理后的content
data[['content_id','content','subject','sentiment_value','sentiment_word']].to_csv('car训练集.csv', index=False)

print("预处理后的测试数据已保存到car训练集.csv")
