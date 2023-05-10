from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


def loadDocuments():
    documents = []

    # load all documentation from opensearch docs/blogs
    loader = DirectoryLoader("./engineer_data", glob="**/*.txt")
    for doc in loader.load():
        documents.append(doc)

    # split content
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=3000, chunk_overlap=0, separators=[" ", ",", "\n"]
    # )
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    return text_splitter.split_documents(documents)
