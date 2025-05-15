import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库
import time  # 导入time库以支持重试间隔
import re  # 导入正则表达式库以支持验证

# 读取CSV文件
df = pd.read_csv(r"D:\py project\qg\产品\产品测试集.csv")

# 初始化OpenAI客户端
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

# 定义一个函数来验证BIO格式的标注结果
def is_valid_bio_format(text):
    # 定义一个正则表达式来匹配BIO格式的标注
    pattern = re.compile(r'^(B-(?:BANK|PRODUCT|COMMENTS_N|COMMENTS_ADJ)|I-(?:BANK|PRODUCT|COMMENTS_N|COMMENTS_ADJ)|O)( (B-(?:BANK|PRODUCT|COMMENTS_N|COMMENTS_ADJ)|I-(?:BANK|PRODUCT|COMMENTS_N|COMMENTS_ADJ)|O))*$')
    return bool(pattern.match(text))

# 定义一个函数来调用API进行实体标注，并增加重试机制
def annotate_text(id, text, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，我有一个用户有关于银行产品评论的csv文件，任务是对该文本进行BIO格式的实体标注，实体标注采用BIO格式，即Begin, In, Out格式。具体说明：B-BANK 代表银行实体的开始，I-BANK 代表银行实体的内部，B-PRODUCT 代表产品实体的开始，I-PRODUCT 代表产品实体的内部，O 代表不属于标注的范围，B-COMMENTS_N 代表用户评论（名词），I-COMMENTS_N 代表用户评论（名词）实体的内部，B-COMMENTS_ADJ 代表用户评论（形容词），I-COMMENTS_ADJ 代表用户评论（形容词）实体的内部。每个字和标点符号都要有对应的分类，每个分类之间以一个空格进行分隔。最后只给出标注结果就行，不要返回原文，不要解释和返回其他信息。谢谢！"
                    "示例："
                    "交行14年用过，半年准备提额，却直接被降到1Ｋ，半年期间只T过一次三千，其它全部真实消费，第六个月的时候为了增加评分提额，还特意分期两万，但降额后电话投诉，申请提...""B-BANK I-BANK O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O O O O O O O O O O O O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O B-COMMENTS_N I-COMMENTS_N O O O O B-PRODUCT I-PRODUCT O O O O B-COMMENTS_ADJ O O O O O O O O O O O O O"
                    "建设银行提额很慢的……""B-BANK I-BANK I-BANK I-BANK B-COMMENTS_N I-COMMENTS_N B-COMMENTS_ADJ I-COMMENTS_ADJ O O O"
                    "我的怎么显示0.25费率，而且不管分多少期都一样费率，可惜只有69k""O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O O O O B-COMMENTS_N I-COMMENTS_N O B-COMMENTS_ADJ I-COMMENTS_ADJ B-COMMENTS_ADJ I-COMMENTS_ADJ O O O"
                    "利率不错，可以撸""B-COMMENTS_N I-COMMENTS_N B-COMMENTS_ADJ I-COMMENTS_ADJ O O O O"
                    "不能??好像房贷跟信用卡是分开审核的反正我的不得""O O O O O O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O O O O"
                    "我感觉这样才是合理的，花呗白条没要那么多信息，照样可以给额度。有征信威慑，没那么多人敢借了不还。与其眉毛胡子一把抓，还不如按额度区间对客户进行不同程度的调查，免...""O O O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O O B-PRODUCT I-PRODUCT B-PRODUCT I-PRODUCT O O O O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O B-PRODUCT I-PRODUCT B-COMMENTS_ADJ I-COMMENTS_ADJ O O O O O O O O O O O O O O O O O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O O O O O O O O O O O O O O O O"
                    "羡慕，可能上个月申请多了。上月连续下了浦发广发华夏交通。这个月申请建行，农业??邮储各种秒拒，买了点建行理财，3个月后在看...""O O O O O O O O O O O O O O O O O O O B-BANK I-BANK B-BANK I-BANK B-BANK I-BANK O O O O O O O O O O O O O O O O O O O O O O O O O O O B-PRODUCT I-PRODUCT O O O O O O O O O O"
                    "这个短债只是用来提升信誉刷建行预审批的，又不是什么赚钱的基金。也只有建行有卖，我买过，不但不赚钱还会亏本的。...""O O B-PRODUCT I-PRODUCT O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O O O B-BANK I-BANK B-COMMENTS_N I-COMMENTS_N I-COMMENTS_N O O O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ O B-PRODUCT I-PRODUCT O O B-COMMENTS_ADJ I-COMMENTS_ADJ B-BANK I-BANK O O O O O O O O O B-COMMENTS_ADJ I-COMMENTS_ADJ I-COMMENTS_ADJ O O B-COMMENTS_ADJ I-COMMENTS_ADJ O O O O O"
                    "打电话问中信信用卡为啥没批呗""O O O O B-BANK I-BANK B-PRODUCT I-PRODUCT I-PRODUCT O O O O O"
                    "这几天也接到了建行的分期电话，让我1.6分12期，跟他说5000，3期，意思意思。快两年了还没首提，心塞""O O O O O O O B-BANK I-BANK O B-PRODUCT I-PRODUCT O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O B-COMMENTS_ADJ I-COMMENTS_ADJ"
                    "我是建行贷款，别的银行信用卡没还，账单大概4w""O O B-BANK I-BANK B-PRODUCT I-PRODUCT O O O O O B-PRODUCT I-PRODUCT I-PRODUCT O O O B-COMMENTS_N I-COMMENTS_N O O O O"
                    "微粒贷，借呗，这些也是多头的因素。因为这些都上征信。我两行卡，都是多头。因为我也是有使用微粒贷，花呗，借呗，白条。查征信显示这些全部上征信了。...""B-PRODUCT I-PRODUCT I-PRODUCT O B-PRODUCT I-PRODUCT O O O O O O O O O O O O O O O O O B-PRODUCT I-PRODUCT O O O O O O O O O O O O O O O O O O O B-PRODUCT I-PRODUCT I-PRODUCT O B-PRODUCT I-PRODUCT O B-PRODUCT I-PRODUCT O B-PRODUCT I-PRODUCT O O B-PRODUCT I-PRODUCT O O O O O O O B-PRODUCT I-PRODUCT O O O O O"
                    "别急。说不准的。第一次过的时候也审核了十几天。不过最后全额度通过。利息高就没用""B-COMMENTS_ADJ I-COMMENTS_ADJ O O B-COMMENTS_ADJ I-COMMENTS_ADJ O O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O O O O O O B-COMMENTS_N I-COMMENTS_N O O O O B-COMMENTS_N I-COMMENTS_N B-COMMENTS_ADJ O O O"
                     },
                    {"role": "user", "content": f"评论内容：{text}"}
                ],
                stream=False
            )
            # 假设API返回的是标注好的文本，以BIO格式
            bio_result = response.choices[0].message.content
            if is_valid_bio_format(bio_result):
                return bio_result
            else:
                print(f'ID: {id} 返回的结果格式无效: {bio_result}')
                with open('error_logs.txt', 'a') as f:
                    f.write(f'ID: {id}, 评论内容: {text}\n返回的结果格式无效: {bio_result}\n')
                raise ValueError("返回的结果格式无效")
        except Exception as exc:
            if attempt < retries - 1:  # 如果不是最后一次尝试
                time.sleep(2 ** attempt)  # 重试间隔时间指数增长
            else:
                print(f'ID: {id} 生成异常: {exc}')
                with open('error_logs.txt', 'a') as f:
                    f.write(f'ID: {id}, 评论内容: {text}\n错误信息: {exc}\n')
                return "-1"

# 使用ThreadPoolExecutor进行多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_news = {executor.submit(annotate_text, id, text): id for id, text in df[['id', 'text']].values}
    for future in tqdm(as_completed(future_to_news), total=len(df['text']), desc="处理评论", unit="条"):
        id = future_to_news[future]
        try:
            result = future.result()
        except Exception as exc:
            result = "-1"
            print(f'ID: {id} 获取结果异常: {exc}')
            with open('error_logs.txt', 'a') as f:
                f.write(f'ID: {id}, 评论内容: {df[df["id"] == id]["text"].values[0]}\n获取结果错误信息: {exc}\n')
        results.append((id, result))

# 创建一个DataFrame来保存结果，然后合并到原始df中
results_df = pd.DataFrame(results, columns=['id', 'annotated_text'])
df = pd.merge(df, results_df, on='id', how='left')

# 将标注结果保存到新的CSV文件
df.to_csv('测试集产品主题分类结果.csv', index=False)

print("标注完成，结果已保存到产品主题分类结果.csv")
