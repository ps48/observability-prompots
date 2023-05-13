from langchain import OpenAI


class BaseModel:
    """
    Base Model class to init the llm model to be used by tools, agents, planners and executor
    NOTE: In future this can have multiple options to easily switch between LLM models/params
    """

    llm = None

    def __init__(self):
        self.llm = OpenAI(temperature=0)

    def get_model(self):
        return self.llm
