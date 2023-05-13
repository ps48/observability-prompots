from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from agent_tools import create_tools
from base_llm import BaseModel

tools = create_tools()
llm = BaseModel().get_model()

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
)

agent.run("What is the current response time of the product catalog service?")

