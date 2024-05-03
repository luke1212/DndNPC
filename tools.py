from langchain.agents import tool
from langchain.agents import initialize_agent
import data_retrieval

@tool
def background_story(text: str) -> str:
    """Returns background story, use this for any question related to knowing background stroy.\
    the input should always be the question related to the background story.\
    and the function should always return background story.\ 
    -- any question that is not background story \
    related should be handled by the main agent."""
    return data_retrieval.similarity_search(text)[0].page_content

tools = [background_story]
