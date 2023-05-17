import os
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from base_llm import BaseModel


current_directory = os.path.dirname(os.path.realpath(__file__))
vectorstore_directory = current_directory + "/chroma_vector_store"


def loadDocuments() -> List[Document]:
    documents = []

    loader = DirectoryLoader(current_directory + "/static_data")
    for doc in loader.load():
        documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def create_vector_store(embeddings=BaseModel().get_embeddings()):
    if os.path.exists(vectorstore_directory):
        print("reading persisted vectorstore")
        vectorstore = Chroma(
            persist_directory=vectorstore_directory,
            collection_name="otel-demo-knowledge",
            embedding_function=embeddings,
        )
    else:
        print("loading documents to vector store")
        documents = loadDocuments()
        if len(documents) == 0:
            print("no documents loaded, exiting")
            exit(1)

        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=vectorstore_directory,
            collection_name="otel-demo-knowledge",
        )
        vectorstore.persist()

    return vectorstore
