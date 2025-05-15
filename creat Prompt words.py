import pandas as pd
from openai import OpenAI

data = pd.read_csv(r"D:\py project\qg\旅游2\旅游2测试集.csv")

# 将DataFrame转换为字符串，这里假设你希望将整个CSV内容作为一条消息传递
content = data.to_string(index=False)

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手，需要帮助生成调用deepseek的api对一个旅游评论数据集进行情感分析时，所需要提供给api的提示模板"},
        {"role": "user", "content": f"""我有一个有关于旅游评论的数据集，一会想调用deepseek的api对它进行情感分类。现在需要你根据这个数据集，帮助我生成一会我需要提供给api的新的提示词，目的是使我一会调用api时，得到的结果的f1值更高。
         提示词的基本要求：一会调用api进行情感分类输出结果用1代表积极情感，0代表消极情感。返回答案只要单个数字即可。
         数据集：{content}。"""},
    ],
    stream=False
)

print(response.choices[0].message.content)
result = response.choices[0].message.content

# 将结果保存到文件中
with open(r"D:\py project\qg\旅游2\旅游2提示词.txt", "w") as file:
    file.write(result)

print(f"结果已保存到")