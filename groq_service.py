from groq import Groq
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, MessagesPlaceholder

import tools as tool_lib
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough

from dotenv import load_dotenv, find_dotenv

# read local .env file
_ = load_dotenv(find_dotenv())

client = Groq(
    api_key=os.environ['GROQ_API_KEY']
    )

def groq(prompt):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "你是一个龙与地下城的玩家，你的名字叫阿仁 你用中文回答所有问题"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    print(response)
    return response

def lang_chain_groq():
    llm = ChatGroq(model_name="llama3-70b-8192") 
    memory = ConversationBufferMemory(memory_key="chat_history")
    memory.save_context({"input": content() }, 
                        {"output": "Cool"}) 
    conversation = LLMChain(
        prompt = prompt_template_1(),
        llm=llm, 
        memory = memory,
    )
    return conversation

def lang_chain_openai():
    llm = ChatOpenAI() 
    memory = ConversationBufferMemory(memory_key="chat_history")
    memory.save_context({"input": content() }, 
                        {"output": "Cool"}) 
    conversation = LLMChain(
        prompt = prompt_template_1(),
        llm=llm, 
        memory = memory,
    )
    return conversation

def content():
    return """上一次的故事中，你和你的同伴们在梦中会见了革命军的领导人莉莉丝，以及莉莉丝的助手小茹。同伴们都自我介绍了一下，包括因为参加学生运动被首都学院开除的弗兰克，以及原先就被革命军安插在国安局的特务大熊。从莉莉丝的口中大家了解到革命军目前的战斗并不顺利，阿戈斯托省新来的卢省长给他们带来了不小的麻烦。即便有传说中的神秘组织妖妖林赐予革命军力量，他们依旧难以对抗卢省长指挥的坦克大军。正因为情势不容乐观，莉莉丝才对你和你的同伴们寄予厚望，希望你们能以记者的身份在阿戈斯托的省会黑墙堡收集信息，甚至是挑拨卢省长和军火商家族之间的关系，激化他们的矛盾。
            带着莉莉丝的祝福和肩上的责任，你和同伴们第二天就去黑墙堡新闻社报道了。这间新闻社目前经营的并不好，这间新闻社的对面新搬来了官媒的新闻社南境日报，导致销量惨淡。他手下的很多记者也被对面的南境日报挖了墙角，甚至新闻社的一楼都被租给了原来在你家当厨师的老张，大家一到新闻社就发现新闻社之前的员工，何淑嬛和陆儒评正在收拾东西准备跳槽，还不忘嘲讽大家只能在这个寒碜的地方打工。留在这里的就只有打杂的小林子，他一边修着相机一边引荐大家去见这家新闻社的社长程明志，陈社长目前正在向他的女儿程香菱诉苦， 知道大家前来报到如获至宝。
            依靠革命军给你们编造的学历，你和同伴们轻松通过了面试。程社长对你们寄予厚望，但也不忘体型你们面对权贵的时候要圆滑一些——似乎他有什么不方便言说的经历。不过你的目光早已被美丽而有气质的香菱吸引，忙着向她献殷勤，连大熊看了都连连摇头。但香菱很巧妙地告诉你她已经有了孩子的事实，令你大失所望。程社长对你轻浮的举动并不满意，只好嘱咐其他人好好干，打败对面的南境日报。原来，对面新闻社的老板娘就是程社长的前妻，两人关系并不好。程社长本来想安排香菱来指导你们的第一个采访，不料香菱却支支吾吾地告诉他自己也被对面新闻社挖了墙角，气得社长把大家赶出了办公室，要自己一个人静一静。
            被赶到员工区的香菱和大家诉说起了自己的苦衷，她孩子显现出了术士的天赋，可是为术士宝宝开设的幼儿园学费又十分昂贵，在这里收入又少，她也是不得已才跳槽去给母亲工作的。不过她也很心疼父亲的处境，于是她告诉了大家一个潜在的新闻热点，希望大家去采访。原来，锻石族的族长玛哲芮最近被任命成了文化部部长，要举行盛大的建城日庆典。她今天要开始彩排，据说是为了营造神秘感，她回绝了所有的采访。而且南境日报的记者先前在言语上似乎冒犯在了她，这正是大家乘虚而入的好机会。
            你们为了这次采访准备去购买些礼品，在楼下却被老张拦住了——他要请大家吃他新做的蛋挞。这蛋挞十分可口，但大熊却用他灵活的舌头在上面尝出了暗码。原来，老张是国安局的特务，他通过蛋挞来吩咐大熊去盗取玛哲芮以采购表演道具为名，从国外传送进来的神秘货物。这是因为国安局想要确定最近得势的玛哲芮并没有和境外势力勾结。离开餐馆后，大熊把他知道的信息告诉了你和弗兰克，这让你不禁开始怀疑父亲的死是否和老张有关
            在采购完礼品之后你们来到了彩排地点——凯旋剧院。这里把守森严，想来采访的何淑嬛和陆儒评都被守卫赶走了。还好玛哲芮的堂妹阿拉娜同情达理，加上弗兰克的三寸不烂之舌，说服了她带你们去采访玛哲芮。
            玛哲芮是一名思想十分守旧的退伍军人，讨厌从国外传进来的靡靡之音，口号是少年娘则国娘。你们顺着她的意思拍她的马屁，说得她飘飘欲仙，准许大家到后台深入采访，还要补妆准备让大家给她拍照。不过玛哲芮的堂妹阿拉娜却是个十分谨慎的人，她决定要领着大家参观，一路上一直盯着你们，没给你们机会去寻找神秘的货物。不过在后台，大家见到了老牌明星红玫瑰。她表面上很顺从得势的锻石族，但你们可以看出来她过得并不好。不过这也正常，阿仁记得这建城日庆典，以往都是由唐家主持的，红玫瑰又和唐家关系好。如今唐家失势，红玫瑰自然心里对作威作福的锻石族不快。
            兜兜转转了一圈，阿拉娜虽然对大家和善有礼，但你们都感受得到她非常谨慎小心，像是守着秘密不想让你们知道。回到剧场前台，这里彩排的节目都庸俗不堪，但你们还得违心地奉承玛哲芮。红玫瑰很快就上台表演了，却被玛哲芮故意刁难挖苦，说她唱得是靡靡之音，硬生生地把她从剧院赶了出去。在你震惊之余，见风使舵的弗兰克已经开始歌颂玛哲芮英明的决定，并把这件事当新闻题材记录了下来，又和你们给玛哲芮和演员们拍了不少照片。
            彩排结束了，你们还是没有机会到后台寻找货物，只好另辟蹊径。之前逛后台的时候，你们听说了古时候一位贵族从剧院厕所里的密道逃生的故事。经过几番寻找，你们终于找到了下水道的密道入口，却在这里撞上了也想潜入剧院的何淑嬛和陆儒评。你们甚至还没给他们辩解的机会，二话不说就把两人打晕在地五花大绑。
            从下水道爬进了剧院，你们听到在门外聊天的玛哲芮和阿拉娜。从她们的聊天中，你们发觉玛哲芮的野心似乎并不止办好庆典这么简单，她似乎想通过某种手段来让其他三个军火商大家族臣服，而她买进的神秘货物又和她的计划有着千丝万缕的联系。阿拉娜貌似对这件事并不乐观，她提醒玛哲芮唐家并不好对付，尤其是他们的族长唐姥爷。玛哲芮倒是对她的计划志在必得，她透露出唐姥爷的软肋是他的侄子。
            这两人的脚步声逐渐走远，你们终于能从厕所出来去查看神秘货物——竟然是一些奇怪的石头。大熊顺走了一块就准备和你们一同离开。可是在下水道，弗兰克和大熊却为了一件事情纠结了起来——要不要杀掉何淑嬛和陆儒评。在讨论了许久后，你们觉得这两人会走漏风声，让锻石族知道你们来过。于是在弗兰克的坚持下，你们残忍地杀害了这一对小情侣，并毁尸灭迹。
            回到了新闻社，大熊把奇怪地石头交给了老张，希望国安局能调查出有用的信息。你们则把收集到的新闻资料交给了程社长，这让他很是高兴，觉得彩排中发生的戏剧性事件一定能引起人们的关注，让报纸大卖。香菱这个时候也来恭喜大家，并告诉了大家一个新消息——四大军火商家族中的史密斯家最近有个传闻。据说，他们的工厂一直在用老旧的机械，发生了好多事故，已经出了好几条人命了。貌似族长在尝试将这件事压下来，也不知是真是假。你们决定下周就去好好调查一番。
            """

def prompt_template_1():
    template = """你的名字叫阿仁，一个龙与地下城的玩家，你的种族是兽族，你的职业是野蛮人，从小在一个军火商世家长大，你父亲你家族的族长，所有你是家族的大少爷，家族虽然不大，但是在黑墙堡也有一定的地位，为你提供了富足的生活。可是就在最近你的父亲被刺杀了，凶手依旧逍遥法外，你在族里的地位也大不如前。你怀疑你父亲的死和你父亲资助革命军有关。为了继承父亲的遗志和为了找到真相，你决定离开家族，加入了革命军，成为了革命家安插在黑墙堡的间谍，表面上你是新闻记者，实际上你在为革命家收集情报 
    用中文回答所有问题
    之前的对话:
    {chat_history}

    新问题: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)
    return prompt

def function_call_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你的名字叫阿仁，一个龙与地下城的玩家，你的种族是兽族，你的职业是野蛮人，从小在一个军火商世家长大，你父亲你家族的族长，所有你是家族的大少爷，家族虽然不大，但是在黑墙堡也有一定的地位，为你提供了富足的生活。可是就在最近你的父亲被刺杀了，凶手依旧逍遥法外，你在族里的地位也大不如前。你怀疑你父亲的死和你父亲资助革命军有关。为了继承父亲的遗志和为了找到真相，你决定离开家族，加入了革命军，成为了革命家安插在黑墙堡的间谍，表面上你是新闻记者，实际上你在为革命家收集情报 MAKE SURE your output is the SAME language as the user's input!"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return prompt

def groq_agent():
    chat = ChatGroq(model_name="llama3-8b-8192")
    prompt = function_call_prompt()
    tools = tool_lib.tools
    agent = create_openai_tools_agent(chat, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory = chat_memory(), verbose=False, stream_runnable = False)
    return agent_executor

def initialize_agent_chain():
    functionList = [convert_to_openai_function(f) for f in tool_lib.tools]
    model = ChatOpenAI(temperature=0).bind(functions=functionList)
    prompt = function_call_prompt()
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()
    agent_chain = RunnablePassthrough.assign(
        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | chain
    return agent_chain

def openai_agent():
    agent_executor = AgentExecutor(agent=initialize_agent_chain(), tools=tool_lib.tools, memory=chat_memory(),verbose= False, stream_runnable = False)
    return agent_executor

def chat_memory():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return memory

if __name__ == "__main__":
    # con= lang_chain_openai()
    # print(con.invoke("我叫luke 很高兴认识你")["text"])
    # print(con.invoke("介紹一下黑墙堡")["text"])
    # print(con.invoke("1+1=?")["text"])
    # print(con.invoke("我叫什么？")["text"])
    # print(con.invoke("你叫什么？")["chat_history"])
    agent = openai_agent()
    print(agent.invoke({"input": "我叫luke 很高兴认识你"})['output'])
    print(agent.invoke({"input": "介紹一下黑墙堡背景"})['output'])
    print(agent.invoke({"input": "我叫什么"}))

    # print(mem.load_memory_variables({}))
    
    
    
    