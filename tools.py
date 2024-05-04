from langchain.agents import tool
from langchain.agents import initialize_agent
import data_retrieval

@tool
def background_story(text: str) -> str:
    """return 黑墙堡的历史，黑墙堡的名人(卢振南、弗洛斯特、沃夫冈·史密斯、唐姥爷、玛哲芮、维吉尔先生、程明志，萧湘, 四大军火家族)，黑墙堡的城市布局的信息。\ 
    , use this for any question related to knowing 黑墙堡的历史，黑墙堡的名人(卢振南、弗洛斯特、沃夫冈·史密斯、唐姥爷、玛哲芮、维吉尔先生、程明志，萧湘,四大军火家族)，黑墙堡的城市布局.\
    the input should always be the question related to the 黑墙堡的历史，黑墙堡的名人(卢振南、弗洛斯特、沃夫冈·史密斯、唐姥爷、玛哲芮、维吉尔先生、程明志，萧湘,四大军火家族)，黑墙堡的城市布局的信息.\
    and the function should always return 黑墙堡的历史，黑墙堡的名人(卢振南、弗洛斯特、沃夫冈·史密斯、唐姥爷、玛哲芮、维吉尔先生、程明志，萧湘,四大军火家族)，黑墙堡的城市布局的信息。\ 
    -- any question that is not 黑墙堡的历史，黑墙堡的名人(卢振南、弗洛斯特、沃夫冈·史密斯、唐姥爷、玛哲芮、维吉尔先生、程明志，萧湘,四大军火家族)，黑墙堡的城市布局的信息 \
    related should be handled by the main agent."""
    response = []
    for doc in data_retrieval.similarity_search(text):
        response.append(doc.page_content)
    return response

tools = [background_story]
