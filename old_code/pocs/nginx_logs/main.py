from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from load_utils import loadDocuments
from langchain.docstore.document import Document
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

#   chat prompt
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Chroma


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can customize this to allow specific origins only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# chat history list
chat_history = []

# default prompt
system_template_prompt = """
Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Your role is to help an engineer to solve system issues by looking at logs, traces and metrics, then further relating them to supplied Question and answers.
Find issue in logs above, summarize the issue and provide next steps for the engineer to follow.

Nginx logs JSON format

['time': '2023-05-02T13:15:22Z', 'request': 'POST /api/v1/order', 'status': 201, 'response_time': 0.800, 'bytes_sent': 100, 'upstream_response_time': 0.600, 'upstream_addr': '10.0.0.6:8080']
['time': '2023-05-02T13:26:23Z', 'request': 'POST /api/v1/order', 'status': 201, 'response_time': 1.100, 'bytes_sent': 100, 'upstream_response_time': 0.900, 'upstream_addr': '10.0.0.6:8080']
['time': '2023-05-02T13:31:24Z', 'request': 'POST /api/v1/order', 'status': 201, 'response_time': 1.800, 'bytes_sent': 100, 'upstream_response_time': 1.200, 'upstream_addr': '10.0.0.6:8080']
['time': '2023-05-02T13:36:25Z', 'request': 'POST /api/v1/order', 'status': 201, 'response_time': 3.000, 'bytes_sent': 100, 'upstream_response_time': 2.500, 'upstream_addr': '10.0.0.6:8080']
['time': '2023-05-02T13:15:26Z', 'request': 'POST /api/v1/order', 'status': 201, 'response_time': 4.500, 'bytes_sent': 100, 'upstream_response_time': 3.800, 'upstream_addr': '10.0.0.6:8080']
----------------
{context}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template_prompt),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
# Construct a ChatVectorDBChain with a streaming llm for combine docs
# and a separate, non-streaming llm for question generation
documents = loadDocuments()
llm = OpenAI(temperature=1e-10)

# # vector store generation using embedding
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
streaming_llm = ChatOpenAI(
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    temperature=0.5,
)

# lang chain `staff` chain type
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=prompt)

# chat vector chain
qa = ChatVectorDBChain(
    vectorstore=vectorstore,
    combine_docs_chain=doc_chain,
    question_generator=question_generator,
)


@app.post("/question")
async def question(query: str = Body(...)):
    global chat_history  # Declare chat_history as a global variable
    result = qa(
        {"question": query, "chat_history": chat_history}, return_only_outputs=True
    )

    chat_history.append((query, result["answer"]))
    vectorstore.add_texts(texts=[query, result["answer"]])
    return result
