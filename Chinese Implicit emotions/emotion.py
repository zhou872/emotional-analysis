import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件，确保包含title和content两列
data = pd.read_csv(r"D:\py project\qg\中文隐式情感分析\实验集.csv")

# 确保每一组内的句子按照ID2排序
sorted_data = data.sort_values(by=['ID', 'ID2'])

# 初始化OpenAI客户端
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

# 定义一个函数来处理每组新闻的情感分类
def classify_emotion_group(group_id, group_data):
    try:
        # 按照ID2排序确保句子顺序正确
        group_data = group_data.sort_values(by='ID2')

        # 将ID2和Sentence组合成combined_content
        combined_content = f'ID：{group_id}，内容：\n' + '\n'.join(f'{row["ID2"]},{row["Sentence"]}' for index, row in group_data.iterrows())
        # 检查内容是否为空
        if pd.isna(combined_content):
            print(f"内容为空，返回id: {group_id}")
            return group_id, ['-2'] * len(group_data), group_data['Sentence'].values

        response = client.chat.completions.create(
            model='deepseek-reasoner',
            messages=[
                {"role": "system", "content":
                    "你是一个乐于助人的助手，现在帮我进行一个中文对话数据集的情感分类，判断我给出的每组对话中每句话的隐式情感，其中第一个数字是组内每句话的前后顺序（从1开始），其中0为中性情感，1褒义隐式情感，2贬义隐式情感.要根据上下文对话结合语境判断。一步一步思考,回答下面的问题。用分隔符####返回答案,答案只要单个数字即可，不要别的东西。每组对话情感值返回的顺序与句子顺序相同。"
                    "再读一遍问题：{帮我进行一个中文对话数据集的情感分类，判断我给出的每组对话中每句话的隐式情感，其中第一个数字是组内每句话的前后顺序（从1开始），其中0为中性情感，1褒义隐式情感，2贬义隐式情感.要根据上下文对话结合语境判断一步一步思考,回答下面的问题。用分隔符####返回答案,答案只要单个数字即可，不要别的东西。每组对话情感值返回的顺序与句子顺序相同。}"
                    "我们将隐式情感定义为：“不含有显式情感词，但表达了主观情感的语言片段”，并将其划分为事实型隐式情感和修辞型隐式情感。其中，修辞型隐式情感又可细分为隐喻/比喻型、反问型以及反讽型。"

                    "对话示例（示例中只给出部分话的情感分类和思维链，但你在进行情感分类时要对每句话都进行分类）："
                    "1,1,讨厌谁我就给对方买蒙牛！"
                    "1,2,别以为政治与你无关，有人送我蒙牛的产品我会以为对方要害我！"
                    "提取句子中的关键词和语境:句子核心动作是将蒙牛产品与害我建立关联"
                    "分析这些元素如何暗示褒义或贬义情感:通过 我会以为对方要害我 的强烈负面表述，暗示蒙牛产品具有危害性. 别以为政治与你无关 的否定句式增强语境严肃性，暗示产品质量问题可能具有社会性影响"
                    "整体通过建立产品接收与人身危害的隐喻关系，隐晦表达对品牌的不信任和负面评价"
                    "####2"
                    
                    "4,1,下回烧香可叫我，我每月去一次的，阿弥陀佛~"
                    "核心动作是邀请参与烧香和定期参与，显示积极态度"
                    "烧香可叫我 通过主动邀约行为，隐含对宗教活动的积极态度"
                    "每月去一次 的频率强调，暗示规律性参与带来的精神认同感"
                    "阿弥陀佛 的宗教用语配合波浪号，营造轻松虔诚的语境氛围"
                    "未直接评价活动价值，但通过参与意愿和仪式感传递隐性正面判断"
                    "####1"
                    "4,2,好的，香友"
                    
                    "10,1,期待奥运会夺冠  @大河报"                    
                    "10,2,【刘翔抢跑被罚出 罗伯斯7秒66夺冠】。"                    
                    "10,3,北京时间24日凌晨，2012年斯德哥尔摩室内田径赛正在进行。"
                    "句子主干为“斯德哥尔摩室内田径赛正在进行”的客观时态陈述"
                    "“北京时间24日凌晨”采用标准新闻时间格式，强化信息传递功能"
                    "缺少修饰性形容词或副词，未触发对赛事质量的价值判断"
                    "整体符合新闻报道导语特征，仅履行信息同步职能而无情感诱导"
                    "####0"  
                    "10,4,在备受关注的男子60米栏比赛中，“中国飞人”刘翔意外抢跑，被罚出场外，无缘赛季两连冠；"
                    "10,5,古巴名将罗伯斯以7秒66折桂。"
                    
                    
                    "17,1,【酷派集团子公司被平安银行起诉提前还贷，大股东乐视担保】平安银行深圳分行诉称，其经调查发现，作为担保人之一的酷派集团一家附属公司已出现财务状况恶化的情况，将严重影响借款人的经营及履约能力，故向广东省深圳市中级人民法院提起诉讼。"                    
                    "17,2,（21世纪经济报道）酷派集团子公司被平安银行起诉提前还贷，大股东乐视担保?"
                    "核心事件“被起诉提前还贷”暗含财务风险，通过司法程序强化负面联想"
                    "特殊句式“大股东乐视担保”将经营风险与声名狼藉的关联方捆绑，触发负面记忆锚点"
                    "媒体署名“21世纪经济报道”增强事件可信度，暗示披露信息的权威性"
                    "问号结尾制造悬疑效果，引导读者对担保有效性的质疑"
                    "通过“起诉-担保”的因果链暗示公司治理缺陷，形成对品牌信誉的隐性贬损"
                    "该句虽无显性负面词，但通过三个维度传递隐式贬义：①司法纠纷与财务危机的强关联 ②问题担保方的连带效应 ③经济类媒体的风险警示属性"
                    "####2"      
                    
                    "19,1,转发了东方财富网的微博:【A股创业板估值真相：不但接近纳斯达克还接近历史底部】创业板估值进一步探底，与此同时，同样以新兴产业为主的美国纳斯达克综合指数估值却屡创新高。"
                    "核心行为“转发权威财经媒体内容”隐含对分析结论的认同，通过第三方信源增强说服力"
                    "“探底”与“屡创新高”的对比修辞，暗示创业板估值修复空间，触发市场预期正向联想"
                    "数据引用（纳斯达克指数）建立国际化参照系，赋予本土市场价值重估的合理性"
                    "####1"                    
                    "19,2,（证券时报网）A股创业板估值真相：不但接近纳斯达克还接近历史底部??"                    
                    "19,3,原文评论[3]转发理由:今天乐视网停牌100天，对标特斯拉吗？"
                    "核心行为对标特斯拉通过反向类比制造讽刺语境，利用特斯拉的商业成功反衬乐视困境"
                    "停牌100天的量化描述暗示企业经营异常，与对标形成价值背离的荒诞感"
                    "疑问句式表面寻求解释，实则强化停牌时长-对标合理性的逻辑矛盾"
                    "通过三重否定实现隐性贬损——否定时间合理性（100天停牌）、否定对标可行性（特斯拉参照系）、否定企业信息披露透明度（用问号质疑公告逻辑）"
                    "####2"           
                    
                    "24,1,【乐视在国内被讨债贾跃亭在美或建法乐第未来新总部】21日消息，获得贾跃亭投资的电动汽车公司法乐第未来已聘请中国MAD建筑事务所，帮助该公司在加州北部MareIsland设计未来风格的总部。"
                    "核心结构通过「国内被讨债」与「在美建总部」的地理对比，暗示债务逃避与战略重心转移的关联"
                    "建筑事务所信息作为中性事实穿插，反衬「被讨债」的负面焦点，形成明褒实贬的叙事张力"
                    "用「法乐第」谐音双关（法拉第未来原型品牌），暗指商业信誉的符号化消解与重组"
                    "通过「国内困境-海外扩张」的镜像对照，构建企业家责任缺失的隐喻框架"
                    "####2"                      
                    "24,2,对法乐第未来而言，在近期遭遇一系列困难之后，这处总部代表了新的未来。"
                    "「一系列困难之后」的让步状语，铺垫逆境背景以强化「新未来」的积极转折意义"
                    "动词「代表」赋予建筑实体象征价值，将物理空间转化为企业复兴的精神符号"
                    "「未来风格」的双关运用，既指建筑设计又隐喻企业发展方向的创新性"
                    "通过「困难（过去时）-总部（现在时）」的时态对比，构建触底反弹的叙事逻辑"
                    "####1" 
                    
                    "15,1,最精彩的就是，郭美美领着小伊伊上春晚了。。。"
                    "15,2,上海全攻略 :两个娘娘腔，两只小人妖。。"
                    "「娘娘腔」「人妖」借转述名义使用歧视性称谓，"
                    "生物隐喻「小」+「人妖」构成双重贬抑，将人格特征异化为非正常生物形态"
                    "并列结构消解个体差异，通过群体标签化实现隐性人格侮辱"
                    "####2"
                    "15,3,上海全攻略 据说，刘谦和傅琰东将同时亮相2012年CCTV龙年春晚的舞台。"
                    "「据说」标明消息未经验证，严格区分事实陈述与观点表达"
                    "标准化时间锚点「2012年CCTV龙年春晚」确保事件描述的客观性"
                    "整体句式符合媒体通稿特征，履行信息传递而非评论功能"
                    "####0"
                    "15,4,烦不烦呀?"
                    
                    "30,1,三条线，三段历史，中轴线，中华民国。"
                    "识别关键词：历史名词罗列（中轴线/中华民国）"
                    "无情感修饰词或评价性表述"
                    "####0"
                    "30,2,东线太平天国，西线煦园。"
                    "地理方位+历史地点并列描述"
                    "太平天国作为专有名词无褒贬指向,煦园作为园林名称本身无情感属性"
                    "####0"
                    "30,3,既可阅览江南园林，又可对民国文化有诸多了解"
                    "阅览和了解为中性动词"
                    "整体为功能性描述而非价值判断"
                    "####0"
                },
                {"role": "user",
                 "content": f"帮我进行一个中文对话数据集的情感分类，判断我给出的每组对话中每句话的隐式情感，其中第一个数字是组内每句话的前后顺序（从1开始），其中0为中性情感，1褒义隐式情感，2贬义隐式情感.要根据上下文对话结合语境判断。一步一步思考,回答下面的问题。最后在响应末尾用分隔符####返回答案,答案只要单个数字即可，不要别的东西。每组对话情感值返回的顺序与句子顺序相同。内容：{combined_content}"},
            ],
            stream=False
        )
        # 获取模型返回的情感分类结果
        label = response.choices[0].message.content.strip()
        # 提取实际的标签值
        actual_label = [l.strip() for l in label.split('####') if l.strip().isdigit()]
        # 将所有的标签值组合成一个字符串
        actual_labels = ''.join(actual_label)
        # 判断返回值是否为0, 1, 2
        if all(l in ['0', '1', '2'] for l in actual_labels):
            return group_id, actual_labels, group_data['Sentence'].values
        else:
            print(f"异常返回值: {actual_labels}。")
            return group_id, ['-2'] * len(group_data), group_data['Sentence'].values
    except Exception as exc:
        print(f'ID: {group_id} 生成异常: {exc}')
        return group_id, ['-2'] * len(group_data), group_data['Sentence'].values

# 使用ThreadPoolExecutor进行多线程处理
results = []
grouped_data = data.groupby('ID')

with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers可以根据你的CPU核心数和网络条件调整
    future_to_group = {executor.submit(classify_emotion_group, group_id, group_data): group_id for group_id, group_data in grouped_data}
    for future in tqdm(as_completed(future_to_group), total=len(grouped_data), desc="处理组", unit="组"):
        group_id = future_to_group[future]
        try:
            result = future.result()
            results.extend(zip([group_id] * len(result[1]), result[2], result[1]))
        except Exception as exc:
            print(f'ID: {group_id} 生成异常: {exc}')

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['ID', 'Sentence', 'label'])

# 分离成功和失败的数据
successful_news = results_df[results_df['label'] != '-2']
failed_news = results_df[results_df['label'] == '-2']

# 保存成功的情感分类结果到新的CSV文件，只保留id和label
successful_news[['ID', 'Sentence', 'label']].to_csv('实验集结果.csv', index=False)

# 保存分类失败的新闻到另一个CSV文件
if not failed_news.empty:
    failed_news[['ID', 'Sentence', 'label']].to_csv('failed.csv', index=False)

print("测试数据的情感分类完成，成功结果已保存到测试集结果.csv")
if not failed_news.empty:
    print("分类失败的对话已保存到failed.csv")
