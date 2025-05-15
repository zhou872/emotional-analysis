import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件，确保包含title和content两列
data = pd.read_csv(r"D:\py project\qg\旅游2\旅游2测试集.csv")

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-093359907c794153ba1cd1424f7f6705", base_url="https://api.deepseek.com")


# 定义一个函数来处理每条新闻的情感分类
def classify_emotion(id, review):
    try:
        for attempt in range(3):
            response = client.chat.completions.create(
                model='deepseek-reasoner',
                messages=[
                    {"role": "system", "content":
                        """
你作为酒店评论情感分析专家，请严格按以下规则分类：

【核心规则】
输出0/1的判断必须同时满足：
1. 优先识别重复消费信号（"下次还住"/"推荐"/"依然选择"）
2. 酒店补偿措施（升级/退款/赠送）可覆盖负面描述
3. 卫生安全 > 服务质量 > 硬件设施 > 价格因素

【分类标准强化】
积极（1）新增条件：
✓ 有明确推荐行为或复购意愿
✓ 出现"免费升级"/"贴心服务"/"解决问题"等关键词
✓ 酒店方有主动改进措施
✓ 负面描述后接转折性肯定

消极（0）新增条件：
✓ 出现"欺骗"/"安全隐患"/"虚假宣传"等原则问题
✓ 服务态度类负面词重复出现≥2次
✓ 存在持续困扰的硬件问题（隔音）

【优先级更新】
1. 时间维度：最新体验 > 初始体验
2. 补偿措施有效性：实物补偿 > 口头道歉
3. 复合型评价处理：
   - 积极维度数量 > 消极维度 → 1
   - 消极维度含安全/卫生问题 → 0

【新增特征处理】
积极信号增强：
✓ 性价比表述（人均60元）
✓ 跨时段复购
✓ 特色优势（透明浴室）

消极信号增强：
✓ 位置缺陷（偏远的）
✓ 隐性对比（对比其他酒店）
✓ 价格欺诈

【特殊句式强化】
1. "虽然A问题，但是B优点" → 当B含复购意愿时判1
2. "即使A缺点，也B肯定" → 重点分析B的强度
3. 补充点评时间戳 → 后续内容权重x1.5
最终只返回单个数字，不要返回分析过程
"""
                     },
                    {"role": "user",
                     "content": f"评论内容：{review}"},
                ],
                stream=False
            )
            # 获取模型返回的情感分类结果
            emotion = response.choices[0].message.content.strip()
            # 判断返回值是否为0, 1, 2
            if emotion in ['0', '1']:
                return id, review, emotion
            else:
                print(f"新闻ID: {id} 异常返回值: {emotion}，尝试重新发送请求 (第 {attempt + 1} 次)")

        print(f"ID: {id} 经过三次尝试后仍然返回异常值，返回id: (ID: {id})")
        return id, review, "2"
    except Exception as exc:
        print(f'{id} 生成异常: {exc}')
        return id, review, "2"  # 统一返回一个代表失败的值


# 使用ThreadPoolExecutor进行多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_news = {executor.submit(classify_emotion, id, review): (id, review)
                      for id, review in data[['id', 'review']].values}
    for future in tqdm(as_completed(future_to_news), total=len(data['review']), desc="处理评论", unit="条"):
        id, review = future_to_news[future]
        try:
            result = future.result()
            results.append(result)

        except Exception as exc:
            print(f'新闻ID: {id} 生成异常: {exc}')
            results.append((id, review, "2"))

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['id', 'review', 'emotion'])

# 分离成功和失败的数据
successful_news = results_df[results_df['emotion'] != '2']
failed_news = results_df[results_df['emotion'] == '2']

# 保存成功的情感分类结果到新的CSV文件，只保留id和label
successful_news[['id', 'emotion', 'review']].to_csv('旅游2测试集结果.csv', index=False)

# 保存分类失败的新闻到另一个CSV文件
if not failed_news.empty:
    failed_news[['id', 'emotion', 'review']].to_csv('失败.csv', index=False)

print("测试数据的情感分类完成")
if not failed_news.empty:
    print("分类失败已保存")
