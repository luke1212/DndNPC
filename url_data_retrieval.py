from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

def recursive_url_loader(url: str, max_depth: int):
    loader = RecursiveUrlLoader(url=url, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser").text)
    docs = loader.load()
    return docs

def create_txt_files(docs, path):
    for doc in docs:
        if doc.metadata.get('title') is None:
            continue
        with open(f"{path}/{doc.metadata['title'].replace('/', '_')}.txt", "w", encoding='utf-8') as f:
            f.write(doc.page_content)
            
def create_markdown_files(docs, path):
    for doc in docs:
        if doc.metadata.get('title') is None:
            continue
        with open(f"{path}/{doc.metadata['title'].replace('/', '_')}.md", "w", encoding='utf-8') as f:
            f.write(f"# {doc.metadata['title']}\n\n{doc.page_content}")

if __name__ == "__main__":
    docs = recursive_url_loader("https://react.dev/learn", 2)
    create_markdown_files(docs, "python_docs")
    # for doc in docs:
    #     create_txt_files(docs, "")
    #     print(doc.metadata)
