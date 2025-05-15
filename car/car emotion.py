import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件，确保包含title和content两列
data = pd.read_csv(r"D:\py project\qg\汽车\测试集.csv")


# 初始化OpenAI客户端
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

# 定义一个函数来处理每条新闻的情感分类
def classify_emotion(content_id,content,subject,sentiment_value,sentiment_word):
        try:
            content_id = str(content_id)
            content = str(content)
            subject = str(subject)
            sentiment_value = str(sentiment_value)
            sentiment_word = str(sentiment_word)

            combined_content = f'ID：{content_id}，内容：{content},主题：{subject},情感词：{sentiment_word}'

            if pd.isna(content) :
                print(f"新闻内容为空，返回id: (ID: {content_id})")
                return  content_id,content,subject,sentiment_value,sentiment_word,"2"
            # 拼接title和content作为输入，限制最大长度为4000个字符
            max_length = 4000

            if len(content_id) + len(content) + len(subject) + len(sentiment_word)  > max_length:
                content = combined_content[:max_length -  len('...')] + '...'
            response = client.chat.completions.create(
                model='deepseek-reasoner',
                messages=[
                    {"role": "system", "content":
                        "你是一个情感大师，现在帮我进行一个数据集的情感分类，根据我提供的内容，主题，情感词，判断出数据集中用户对汽车评论的情感极性，其中正面情绪输出1，中性情绪输出0以及负面情绪输出-1，最后只要输出单个数字即可，不要别的东西。"
                        "示例："
                        "13149,因为森林人即将换代，这套系统没必要装在一款即将换代的车型上，因为肯定会影响价格。,价格,影响""回答:0"
                        "8865,这玩意都是给有钱任性又不懂车的土豪用的，这价格换一次我妹夫EP020可以换三锅了,价格,有钱任性""回答:-1"
                        "13315,优惠可以了！购进吧！买了不会后悔的！时间可鉴！   ,价格,不会后悔""回答:1"
                        "7366,来湖北，我上月订的2.0蓝色时尚，优惠1万3送6次保养，本月底提车,价格,优惠""回答:0"
                        "7907,垃圾导航 都不晓得怎么退出  比如听收音，退到主界面还是收音在响，换成USB就能切换过来，到了USB又不知道怎么退出 先天不足啊,配置,垃圾""回答:-1"
                        "12821,说实话，基本上用不上车上导航，用手机更方便！音响效果不用纠结，毕竟不是想成为移动音乐厅。,配置""回答:0"
                        "1262,石桥这款轮胎是进口的，我记得途虎网上面有，但是有可能要提前预定的。我觉得森林人这款车有条件可以配二套轮胎，日常使用就用石桥的原配胎，走长途自驾马牌ccc系列不错。,配置,""回答:0"
                        "4737,这玩意我从来没看过，真鸡肋，有倒车影像这个基本没用,配置,鸡肋""回答:-1"
                        "9624,我车25的，刚换完中缸。平时用车是非常满意的，动力足，加速快，安静省油。真的好用。   ,油耗,省油""回答:1"
                        "9938,主要看上森的水平对置发动机和空间，看这价格真比不了杠五。    ,动力,""回答:1"
                        "9938,主要看上森的水平对置发动机和空间，看这价格真比不了杠五。    ,空间,""回答:1"
                        "9938,主要看上森的水平对置发动机和空间，看这价格真比不了杠五。    ,价格,""回答:0"
                        "148,我也是这样，瘦小的人坐在森的座椅上毫无包裹感，我开一个小时以上就腰疼，所以我买个一个记忆腰靠,舒适性,腰疼""回答:-1"
                        "9311,空间倒无所谓大多数都是我一个人，我对空间没什么要求。没有全景天窗很可惜，xv有种不上不下的感觉。外观我觉得很一般。屁股很不喜欢，可能是因为短    ,空间,""回答:0"
                        "9311,空间倒无所谓大多数都是我一个人，我对空间没什么要求。没有全景天窗很可惜，xv有种不上不下的感觉。外观我觉得很一般。屁股很不喜欢，可能是因为短    ,外观,屁股很不喜欢""回答:-1"
                        "3069,你的刹车片是用什么牌子的，多少钱。那里买。原厂比较贵要2500元，是真的吗？   ,价格,""回答:0"},
                    {"role": "user",
                     "content": f"帮我进行一个数据集的情感分类，根据我提供的内容，主题，情感词，判断出数据集中用户对汽车评论的情感极性，其中正面情绪输出1，中性情绪输出0以及负面情绪输出-1，最后只要输出单个数字即可，不要别的东西。内容：{combined_content}"},
                ],
                stream=False
            )
            # 获取模型返回的情感分类结果
            emotion = response.choices[0].message.content.strip()
            # 判断返回值是否为0, 1, 2
            if emotion in ['-1', '0', '1']:
                return content_id,content,subject,emotion,sentiment_word
            else:
                print(f"异常返回值: {emotion}。")
        except Exception as exc:
            print(f'{content_id} 生成异常: {exc}')
        return content_id,content,subject,sentiment_word,"2" # 统一返回一个代表失败的值

# 使用ThreadPoolExecutor进行多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_news = {executor.submit(classify_emotion, content_id,content,subject,sentiment_value,sentiment_word): (content_id,content,subject,sentiment_value,sentiment_word) for content_id,content,subject,sentiment_value,sentiment_word in data[['content_id','content','subject','sentiment_value','sentiment_word']].values}
    for future in tqdm(as_completed(future_to_news), total=len(future_to_news), desc="处理评论", unit="条"):
        content_id,content,subject,sentiment_value,sentiment_word = future_to_news[future]
        try:
            result = future.result()
            results.append(result)

        except Exception as exc:
            print(f'ID: {content_id} 生成异常: {exc}')
            results.append((content_id,content,subject,sentiment_word,"2"))

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['content_id','content','subject','sentiment_value','sentiment_word'])

# 分离成功和失败的数据
successful_news = results_df[results_df['sentiment_value'] != '2']
failed_news = results_df[results_df['sentiment_value'] == '2']

# 保存成功的情感分类结果到新的CSV文件，只保留id和label
successful_news[['content_id','sentiment_value','content','subject','sentiment_word']].to_csv('测试集结果.csv', index=False)

# 保存分类失败的新闻到另一个CSV文件
if not failed_news.empty:
    failed_news[['content_id','content','subject','sentiment_value','sentiment_word']].to_csv('failed.csv', index=False)

print("测试数据的情感分类完成，成功结果已保存到测试集结果.csv")
if not failed_news.empty:
    print("分类失败的新闻已保存到failed.csv")
