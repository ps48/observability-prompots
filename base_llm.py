from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


class BaseModel:
    """
    Base Model class to init the llm model to be used by tools, agents, planners and executor
    NOTE: In future this can have multiple options to easily switch between LLM models/params
    """

    llm = None
    embeddings = None

    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()

    def get_model(self):
        return self.llm

    def get_embeddings(self):
        return self.embeddings
