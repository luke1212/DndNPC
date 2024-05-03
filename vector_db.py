import openai
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv, find_dotenv

# read local .env file
_ = load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

_loader = DirectoryLoader('docs')
_docs = _loader.load()

def chroma_vector_db():
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    documents = text_splitter.split_documents(_docs)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    return db


if __name__ == "__main__":
    db = chroma_vector_db() 
    query = "老张"
    docs = db.similarity_search(query)
    print(docs[0].page_content)
    