import pandas as pd
from openai import OpenAI

data = pd.read_csv(r"D:\py project\qg\旅游2\旅游2测试集.csv")
data2 = pd.read_csv(r"D:\py project\qg\旅游2\2对比结果.csv")
# 读取之前提供的提示词文档
with open(r"D:\py project\qg\旅游2\旅游2新提示词.txt", "r", encoding="utf-8") as file:
    previous_prompt = file.read()

# 将DataFrame转换为字符串，这里假设你希望将整个CSV内容作为一条消息传递
content = data.to_string(index=False)
content2 = data2.to_string(index=False)

client = OpenAI(api_key="Your API Key", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手，需要帮助修改生成调用deepseek的api对一个旅游评论数据集进行情感分析时，所需要提供给api的提示模板"},
        {"role": "user", "content": f"""我有一个有关于旅游评论的数据集，一会想调用deepseek的api对它进行情感分类。现在需要你根据这个数据集，和你之前提供给我的提示词，根据分类错误的数据，帮助我生成一会我需要提供给api的新的提示词，目的是使我一会调用api时，得到的结果的f1值更高。
         提示词的基本要求：一会调用api进行情感分类输出结果用1代表积极情感，0代表消极情感。返回答案只要单个数字即可。
         数据集：{content}。
         以下是之前的提示词：{previous_prompt}，你需要对其进行修改。
         以下是之前分类错误的数据，这个数据中，第一列是id，用于区分不同数据；第二列是真实情感值；第三列是之前分类错误的情感值；第四列是评论内容。：{content2}"""},
    ],
    stream=False
)

print(response.choices[0].message.content)
result = response.choices[0].message.content

# 将结果保存到文件中
with open(r"D:\py project\qg\旅游2\旅游2新提示词.txt", "w", encoding="utf-8") as file:
    file.write(result)

print(f"结果已保存到")