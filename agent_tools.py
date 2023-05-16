import os
from langchain import LLMChain, PromptTemplate
import requests
from base_llm import BaseModel
from load_utils import loadDocuments
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.agents.tools import Tool
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# Load env variables
load_dotenv()

# Base model used by all tools
basemodel = BaseModel()
llm = basemodel.get_model()


#################################
# OTEL Knowledge tool
#################################
documents = loadDocuments()
# vector store generation using embedding
vectorstore = Chroma.from_documents(
    documents, OpenAIEmbeddings(), collection_name="otel-demo-knowledge"
)
# create a retriver chain to query index with Q/A
otel_knowledge = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)


#################################
# Generate PPL Query Tool
# NOTE: To be replaced by llm
#################################
def generate_ppl_query(input_text: str):
    return (
        "source=jaeger-span* "
        + "| where process.serviceName = 'productcatalogservice'"
        + "| stats avg(duration) as response_time"
    )


#################################
# Execute PPL Query tool
#################################
def execute_ppl_query(ppl_query: str):
    print("PPL Query input given by the agent: ", ppl_query)
    response = requests.post(
        "https://dashboards.otel.lijshu.people.aws.dev/api/ppl/search",
        data={"query": ppl_query, "format": "jdbc"},
        auth=(os.getenv("OSD_USER"), os.getenv("OSD_PASS")),
        headers={"osd-xsrf": "true"},
    )
    return str(response.text)


#################################
# Summarization Tool
#################################
prompt_template = "Write a concise summary of the following: {text}?"
summarize_text_chain = LLMChain(
    llm=llm, prompt=PromptTemplate.from_template(prompt_template)
)


def create_tools():
    tools = [
        Tool(
            name="OTEL Demo knowledge",
            func=otel_knowledge.run,
            description="useful for when you need to answer question about the OTEL Demo Architecture",
        ),
        Tool(
            name="PPL Query Generator",
            func=lambda input_text: generate_ppl_query(input_text),
            description="useful for when you need to generate OpenSearch PPL Query",
        ),
        Tool(
            name="Execute PPL Query",
            func=lambda ppl_query: execute_ppl_query(ppl_query),
            description="useful for when you need to execute the OpenSearch PPL Query. This tool takes PPL Query as the input",
        ),
        Tool(
            name="Summarization",
            func=summarize_text_chain,
            description="useful for when you need to frame a final answer as response, input should be in format text: input",
        ),
    ]

    return tools
