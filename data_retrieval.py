import vector_db
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

_chroma_vector_db = vector_db.chroma_vector_db()
_url_vector_db = vector_db.url_vector_db("https://legacy.reactjs.org/tutorial/tutorial.html", 2)

def similarity_search(query):
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    docs = _chroma_vector_db.similarity_search(query)
    return docs

def url_similarity_search(query):
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    docs = _url_vector_db.similarity_search(query)
    return docs

def data_contextual_compression_retriever(query):
    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=_chroma_vector_db.as_retriever()
    )

    compressed_docs = compression_retriever.invoke(query)
    return compressed_docs

if __name__ == "__main__":
    docs = url_similarity_search("Dictionary Merge & Update Operators")
    for doc in docs:
        print(doc.metadata)
