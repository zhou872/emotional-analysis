import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件，确保包含title和content两列
data = pd.read_csv(r"D:\py project\qg\基于社交媒体的海南旅游景区评价数据集\测试集.csv")

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-093359907c794153ba1cd1424f7f6705", base_url="https://api.deepseek.com")


# 定义一个函数来处理每条新闻的情感分类
def classify_emotion(id,descrtion):
    try:
        for attempt in range(3):
            response = client.chat.completions.create(
                model='deepseek-reasoner',
                messages=[
                    {"role": "system", "content":
                        "你是一个乐于助人的助手，我有一个有关于旅游评论的数据集。你要对这个数据集进行情感分类。用1代表积极情感，0代表中性情感，-1代表消极情感。一步一步思考,回答下面的问题。最后在响应末尾用分隔符####返回答案,答案只要单个数字即可，不要别的东西。"
                        "再读一遍问题:{我有一个有关于旅游评论的数据集。你要对这个数据集进行情感分类。用1代表积极情感，0代表中性情感，-1代表消极情感。一步一步思考,回答下面的问题。最后在响应末尾用分隔符####返回答案,答案只要单个数字即可，不要别的东西。}"

                        "示例："
                        "潜水艇价格也还可以"
                        "####0"
                     
                        "酒店位置很好，住的非常舒适"
                        "####1"

                        "完全就是一个商店嘛！都是卖东西的！黎族卖布，苗族卖银，东西还特别贵，太坑了"
                        "####-1"
                     },
                    {"role": "user",
                     "content": f"再读一遍问题:我有一个有关于旅游评论的数据集。你要对这个数据集进行情感分类。用1代表积极情感，0代表中性情感，-1代表消极情感。一步一步思考,回答下面的问题。最后在响应末尾用分隔符####返回答案,答案只要单个数字即可，不要别的东西。评论内容：{descrtion}"},
                ],
                stream=False
            )
            # 获取模型返回的情感分类结果
            response_content = response.choices[0].message.content.strip()
            # 提取####后的值
            emotion = response_content.split('####')[-1].strip()
            # 判断返回值是否为0, 1, 2
            if emotion in ['0', '1', '-1']:
                return id, descrtion, emotion
            else:
                print(f"新闻ID: {id} 异常返回值: {emotion}，尝试重新发送请求 (第 {attempt + 1} 次)")

        print(f"新闻ID: {id} 经过三次尝试后仍然返回异常值，返回id: (ID: {id})")
        return id, descrtion, "2"
    except Exception as exc:
        print(f'{id} 生成异常: {exc}')
        return id, descrtion, "2"  # 统一返回一个代表失败的值


# 使用ThreadPoolExecutor进行多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_news = {executor.submit(classify_emotion, id,descrtion): (id,descrtion)
                      for id, descrtion in data[['id', 'descrtion']].values}
    for future in tqdm(as_completed(future_to_news), total=len(data['descrtion']), desc="处理评论", unit="条"):
        id,descrtion = future_to_news[future]
        try:
            result = future.result()
            results.append(result)

        except Exception as exc:
            print(f'新闻ID: {id} 生成异常: {exc}')
            results.append((id, descrtion, "2"))

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['id', 'descrtion', 'emotion'])

# 分离成功和失败的数据
successful_news = results_df[results_df['emotion'] != '2']
failed_news = results_df[results_df['emotion'] == '2']

# 保存成功的情感分类结果到新的CSV文件，只保留id和label
successful_news[['id', 'emotion', 'descrtion']].to_csv('测试集结果.csv', index=False)

# 保存分类失败的新闻到另一个CSV文件
if not failed_news.empty:
    failed_news[['id',  'emotion', 'descrtion']].to_csv('失败.csv', index=False)

print("测试数据的情感分类完成")
if not failed_news.empty:
    print("分类失败已保存")
