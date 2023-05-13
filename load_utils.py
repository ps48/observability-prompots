from typing import Any, List
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter


def loadDocuments() -> List[Document]:
    # load opensearch observability website
    documents = []

    # load all documentation from opensearch docs/blogs
    loader = DirectoryLoader("./static_data", glob="**/*.md")
    for doc in loader.load():
        documents.append(doc)

    # load ppl samples queries
    loader = DirectoryLoader("./static_data", glob="**/*.txt")
    for doc in loader.load():
        documents.append(doc)

    # split content
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)

