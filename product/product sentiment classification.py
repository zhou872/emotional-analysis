import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件，确保包含title和content两列
data = pd.read_csv(r"D:\py project\qg\产品\产品测试集.csv")

# 初始化OpenAI客户端
client = OpenAI(api_key="Your API Key", base_url="https://api.deepseek.com")

# 定义一个函数来处理每条新闻的情感分类
def classify_emotion(id,text):
        try:

            response = client.chat.completions.create(
                model='deepseek-reasoner',
                messages=[
                    {"role": "system", "content":
                        "你是一个情感大师，现在帮我进行一个数据集的情感分类，判断出数据集中用户对银行产品评论的情感极性，其中正面情绪输出1，中性情绪输出2以及负面情绪输出0，最后只要输出单个数字即可，不要别的东西。"
                        "新闻示例："
                        "建设银行提额很慢的""回答：0"
                        "这几天也接到了建行的分期电话，让我16分12期，跟他说5000，3期，意思意思。快两年了还没首提，心塞""回答：0"
                        "我是15500，他说今天不还就起诉我，网贷坑人啊""回答：0"
                        "我4星加常年10几个小砖，他拒我76次，最近一年流水大概700多万吧，没见过那个预审批额度出来，我还是白户呢，应该不存在卡多问题，也没有贷款逾期，申什么拒什么，估计没人""回答：0"
                        "七八年了农行还是6千，别的行7万多的都有了""回答：0"
                        "都是褥羊毛而已，生不了钱，日常薅羊毛还是很开心的哦""回答：1"
                        "好啥好 最后坑不还得我来填么？""回答：1"
                        "当然是拿来买基金啊，白嫖多香。""回答：1"
                        "今天用了，秒批，平安的哦""回答：1"
                        "看了建行是你的真爱了！不过建行确实好用！""回答：1"
                        "我感觉这样才是合理的，花呗白条没要那么多信息，照样可以给额度。有征信威慑，没那么多人敢借了不还。与其眉毛胡子一把抓，还不如按额度区间对客户进行不同程度的调查，免""回答：2"
                        "我是建行贷款，别的银行信用卡没还，账单大概4w""回答：2"
                        "晒晒“断”卡，我的招行卡猫咪卡""回答：2"
                        "版主辛苦！为什么有的人可以全额现金分期，我只能分一半，是卡种不同，还是怎么着，望指教，感谢！""回答：2"
                        "我这里的建行想办大额，看重的最容易的就是房产证""回答：2"},
                        {"role": "user",
                     "content": f"帮我进行一个数据集的情感分类，判断出数据集中用户对银行产品评论的情感极性，其中正面情绪输出1，中性情绪输出2以及负面情绪输出0，最后只要输出单个数字即可，不要别的东西。新闻内容：{text}"},
                ],
                stream=False
            )
            # 获取模型返回的情感分类结果
            emotion = response.choices[0].message.content.strip()
            # 判断返回值是否为0, 1, 2
            if emotion in ['0', '1', '2']:
                return id,text,emotion
            else:
                print(f"异常返回值: {emotion}。")
        except Exception as exc:
            print(f'{id} 生成异常: {exc}')
        return id,text,"-1" # 统一返回一个代表失败的值

# 使用ThreadPoolExecutor进行多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_news = {executor.submit(classify_emotion, id,text): (id,text) for id,text in data[['id','text']].values}
    for future in tqdm(as_completed(future_to_news), total=len(data['text']), desc="处理评论", unit="条"):
        id,text = future_to_news[future]
        try:
            result = future.result()
            results.append(result)

        except Exception as exc:
            print(f'新闻ID: {id} 生成异常: {exc}')
            results.append((id,text,"-1"))

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['id','text','class'])

# 分离成功和失败的数据
successful_news = results_df[results_df['class'] != '-1']
failed_news = results_df[results_df['class'] == '-1']

# 保存成功的情感分类结果到新的CSV文件，只保留id和label
successful_news[['id','class','text']].to_csv('测试集结果.csv', index=False)

# 保存分类失败的新闻到另一个CSV文件
if not failed_news.empty:
    failed_news[['id','class','text']].to_csv('failed.csv', index=False)

print("测试数据的情感分类完成，成功结果已保存到测试集结果.csv")
if not failed_news.empty:
    print("分类失败的新闻已保存到failed.csv")

