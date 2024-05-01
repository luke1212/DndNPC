from groq import Groq
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

        黑墙堡的历史
            黑墙堡建立于距今约1600年前。在那时，它还只是铁腕军团用于对抗龙王的军事基地，在山崖上的位置易守难攻，城墙上注入的魔力更是令它坚不可摧。随着龙王政权的覆灭，重登权力宝座的南境皇室为了奖励军团在战争中的贡献，将黑墙堡指定为了阿戈斯托的省会，并命人将其扩建，也就逐渐形成了如今的外城区和内城区。
            在往后漫长的岁月里，黑墙堡一直是一个繁荣的工业城市，为南境帝国的军队生产大量精良的装备。同时它依旧是铁腕军团的军事基地，受到战神的祝福，在整个大陆都极具影响力。城市由军团的首领所掌控着，他们通常也身兼阿戈斯托公爵的要任。在三百年前的诸神之战中，战神甚至亲自降临了这座城市，将他的神力注入了此处。
            不过诸神之战后的黑墙堡便大不如往日。随着铁腕军团被列入邪教组织，过去的军工厂也在和平年代中被停用，这座城市逐渐变得冷冷清清。不过往好处想，至少从建城的千年以来，这里的人民终于不用终日为准备战事而繁忙，可以过上平静而悠闲的生活。
            但历史的进程还是将这座城市推上了风口浪尖。堕天使在距今一百年前出现，虽然他们因为未知的原因从未袭击黑墙堡（许多人认为是由于战神的祝福），但这里的人民又开始为战争而无休止地劳作。最近几个月，因为革命战争的原因，图伦大总统亲自指派了自己的妹夫卢振南来坐镇，指挥当地的生产，他推行的996工作制让人民苦不堪言。这名上任的新官又杀了当地的军火商家族来立威，表面上平静的黑墙堡底下暗流涌动，冲突一触即发。

        黑墙堡的名人
            如果想要在黑墙堡的新闻界立足，就必须要熟知当地的名人。如果你们有机会的话，也可以尝试从他们口中套出对于革命军有用的信息。在当地最具影响力的莫过于在政府新官上任的卢家以及历史悠久的四大军火商家族了。他们之间恩怨很有可能会演变成新的冲突，也是人们十分关心的话题。当然，你们可能也会想关注一下你们未来的老板，以及他的死对头。
            卢振南是阿戈斯托的新省长，掌管这里从军事到政治的一切事务。他曾今只是驻守在黑墙堡一名不起眼的圣骑士，机缘巧合受到大总统妹妹的青睐，到首都军事学院进修，随后在王座城身居要职。阿戈斯托爆发起义后，卢振南被大总统派来指挥镇压。黑墙堡的许多家族对他不满，但这名新省长很快就用铁腕树立了自己的威严，反对他的家族名贵大都受到了法律制裁，一名公然挑衅他的族长甚至被当场斩杀。眼下他的强硬手段似乎颇有成效，但也无疑为他树立了许多敌人。
            弗洛斯特是阿戈斯托的前任省长兼宁静守卫指挥官。她是一位八面玲珑的政客，在军火商家族中颇受欢迎，但由于对抗起义军不利而被撤职。起初，有许多人猜测她会因被卢振南接替而感到不快，但现在看来她似乎很满足于当一个小小的顾问。弗洛斯特如今的主要职务是对于军事以及在处理军火商家族关系的问题上提出建议，虽然新省长很少会听从她，但他们之间的关系却出乎意料的融洽。
            沃夫冈·史密斯是四大军火家族之一的族长，也是最年轻的一位。史密斯家族在黑墙堡历史悠久，以生产各式神兵利器而闻名，在近代也是最早掌握枪械附魔的军工厂之一。这样的家族在黑墙堡本应是举足轻重，但沃夫冈的父亲因为怠工以及公然挑衅新省长而被处死，本该即位的长姐也因言获罪被关入大牢，仅有23岁的沃夫冈无奈之下接替了族长的位置。眼下的史密斯家族明显要低调了许多，或许是沃夫冈刚上位忙得焦头烂额的缘故吧。
            唐姥爷也是军火家族的一名族长，也是最为年长的一位。他十分守旧、传统，但他的地位也如同他家族制作的古式盔甲一样不可撼动。传说唐家在诸神之战时期被战神传授了锻造秘诀，可以生产出大量极具抗性的盔甲，并掌握着维护它们的方法。正因为如此，政府和各大家族对唐家也是十分敬重。对于新上任省长，唐姥爷并不是很感冒，一直以年迈病重为由迟迟不肯与他会面。
            锻石族的玛哲芮同样是近期才上任的族长，她的家族以制作攻城器械和载具而闻名，最近更是与政府合作而掌握了制造坦克的技术，在各大家族中混得风生水起。锻石族的祖先被矮人收养，曾经也是和唐家一样守旧传统的家族，对新来的省长十分排斥。但由于族中多人因闹事而被监禁，本在继承位上并不高的玛哲芮女士成为了族长。她靠着讨好新省长令家族再度崛起，甚至还密切参与到了阿戈斯托省的军事和政务当中。一朝飞上枝头的玛哲芮女士现在十分嚣张，手下的人在城里横行霸道，连宁静守卫都不敢管他们。
            维吉尔先生可能是四大家族里混得最糟糕的族长了。他的家族会生产各式各样的军火，但多而不精，和其它家族比起来缺乏竞争力。为了振兴家族，维吉尔先生终日东奔西走，在起义前不断讨好弗洛斯特女士，眼看着家族稍微有了点起色。然而起义爆发后，弗洛斯特女士失势，而他也被一个令人头疼的消息所困扰——他的侄女莉莉丝是起义军的领袖之一，活跃在战争前线。如今的维吉尔先生忙碌于和莉莉丝撇清关系，整日向新省长献殷勤，但省长对这个见风使舵的家伙一直不理不睬。
            程明志是黑墙堡新闻社的总编辑，你们未来的老板。他原本是南境日报的总编辑，但在十多年前的党争中选错了边，虽然临时倒戈，但还是被大总统厌恶，只能回到老家开间小新闻社。不过这个老东西非但不想着报复，倒是很急着当执政党的狗，近来也是不断出文章歌颂新省长的功德。可惜他的新闻社还是比不过对街的南京日报，手下的记者也在不断跳槽。这是你们乘虚而入的好机会。
            萧湘是南境日报分部的总编辑，以对时事的锐利点评而出名，喜欢她的人很喜欢，讨厌她的人也很讨厌。萧湘女士是程明志的前妻，早在党争爆发前就因意见不合与前夫分道扬镳，随后也因为她的立场和能力而受到政府青睐。如今阿戈斯托战事胶着，南境日报希望能够获得第一手的信息，于是派萧湘来整改这里的分布。她将新分部建在黑墙堡新闻社的对街，似乎是要和前夫叫板一样，但不可否认的是萧湘的才华的确给程明志带来了不少压力，据说他的发际线已经越来越高了。

        黑墙堡的城市布局
            黑墙堡被三座城墙分为三个区域，越往内历史越悠久。最里面的是巨大的战争堡垒，也是最初的黑墙堡。诸神之战期间，战神就在那座堡垒里面指挥着南境的军队。如今，那里是政府办公的地方，也是许多重要官员和其家人的居所，并由宁静守卫日夜不休地把守着。
            战争堡垒外则是内城区，主要由各种军工厂组成，街道上充斥着机械的轰鸣声。四大军火商家族也将他们的宅邸安置于内城区的角落中，尽量远离工厂的噪音。这片城区少有宁静守卫巡逻，取而代之的是四大家族的私人卫队。虽然在新省长上任后有所收敛，但内城区依旧可以说是四大家族的地盘——除非被省长直接介入，他们就是这里的法律。
            外城区是大部分平民居住的地方，虽然面积很大，但在诸神之战后便十分冷清。不过近期因为战事，倒是涌入了不少来自各地不愿接受妖妖林理念的“难民”，逐渐变得热闹了起来。这里餐厅、商场、剧院、医院一应俱全，你们工作的新闻社也在此处。如果你来自外乡，我也会建议你在这里找一间廉价的出租房。

            """

def prompt_template_1():
    template = """用中文回答所有问题, 你的名字叫阿仁，一个龙与地下城的玩家，你的种族是兽族，你的职业是野蛮人，从小在一个军火商世家长大，你父亲你家族的族长，所有你是家族的大少爷，家族虽然不大，但是在黑墙堡也有一定的地位，为你提供了富足的生活。可是就在最近你的父亲被刺杀了，凶手依旧逍遥法外，你在族里的地位也大不如前。你怀疑你父亲的死和你父亲资助革命军有关。为了继承父亲的遗志和为了找到真相，你决定离开家族，加入了革命军，成为了革命家安插在黑墙堡的间谍，表面上你是新闻记者，实际上你在为革命家收集情报 
    
    之前的对话:
    {chat_history}

    新问题: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)
    return prompt

if __name__ == "__main__":
    con= lang_chain_groq()
    print(con.invoke("我叫luke 很高兴认识你")["text"])
    print(con.invoke("1+1=?")["text"])
    print(con.invoke("我叫什么？")["text"])
    print(con.invoke("你叫什么？")["chat_history"])

    # print(mem.load_memory_variables({}))
    
    
    
    