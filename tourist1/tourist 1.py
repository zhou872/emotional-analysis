import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件，确保包含title和content两列
data = pd.read_csv(r"D:\py project\qg\旅游1\旅游1测试集.csv")

# 初始化OpenAI客户端
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")


# 定义一个函数来处理每条新闻的情感分类
def classify_emotion(id, content):
    try:
        for attempt in range(3):
            response = client.chat.completions.create(
                model='deepseek-reasoner',
                messages=[
                    {"role": "system", "content":
                        """你是一个专业的情感分析模型，一步一步思考,回答下面的问题。你现在需要对旅游评论进行二分类情感判断。请严格遵循以下规则：
1. 输出仅允许为单个数字：积极情感=1，消极情感=0
2. 积极情感特征：
   - 出现"值得""方便""满意""推荐""不错""历史感""愉快""美景"等正向词汇
   - 描述文化体验/便利服务/性价比/推荐意愿
   - 表达收获感/愉悦感/超出预期
3. 消极情感特征：
   - 出现"霸王条款""坑""后悔""人太多""排队久""性价比低"等负面词汇
   - 描述强制消费/管理混乱/体验受损/价格不满
   - 表达失望/劝退意向/设施抱怨
4. 特别注意：
   - "需要导游"若暗示额外消费则为0，纯建议则为1
   - "人多"若导致体验下降则为0，仅客观描述保持1
   - 价格评价需结合语境："贵但值得"=1，"不值高价"=0

示例判断：
[感受三国历史文化] → 1
[门票绑定看戏霸王条款] → 0
[人多但值得推荐] → 1     """
                     },
                    {"role": "user",
                     "content": f"评论内容：{content}"},
                ],
                stream=False
            )
            # 获取模型返回的情感分类结果
            emotion = response.choices[0].message.content.strip()
            # 判断返回值是否为0, 1, 2
            if emotion in ['0', '1']:
                return id, content, emotion
            else:
                print(f"新闻ID: {id} 异常返回值: {emotion}，尝试重新发送请求 (第 {attempt + 1} 次)")

        print(f"ID: {id} 经过三次尝试后仍然返回异常值，返回id: (ID: {id})")
        return id, content, "2"
    except Exception as exc:
        print(f'{id} 生成异常: {exc}')
        return id, content, "2"  # 统一返回一个代表失败的值


# 使用ThreadPoolExecutor进行多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_news = {executor.submit(classify_emotion, id, content): (id, content)
                      for id, content in data[['id', 'content']].values}
    for future in tqdm(as_completed(future_to_news), total=len(data['content']), desc="处理评论", unit="条"):
        id, content = future_to_news[future]
        try:
            result = future.result()
            results.append(result)

        except Exception as exc:
            print(f'新闻ID: {id} 生成异常: {exc}')
            results.append((id, content, "2"))

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['id', 'content', 'emotion'])

# 分离成功和失败的数据
successful_news = results_df[results_df['emotion'] != '2']
failed_news = results_df[results_df['emotion'] == '2']

# 保存成功的情感分类结果到新的CSV文件，只保留id和label
successful_news[['id', 'emotion', 'content']].to_csv('测试集结果.csv', index=False)

# 保存分类失败的新闻到另一个CSV文件
if not failed_news.empty:
    failed_news[['id', 'emotion', 'content']].to_csv('失败.csv', index=False)

print("测试数据的情感分类完成")
if not failed_news.empty:
    print("分类失败已保存")
