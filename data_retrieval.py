import vector_db

_chroma_vector_db = vector_db.chroma_vector_db()

def similarity_search(query):
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    docs = _chroma_vector_db.similarity_search(query)
    return docs

if __name__ == "__main__":
    docs = similarity_search("黑墙堡")
    print(docs[0].page_content)