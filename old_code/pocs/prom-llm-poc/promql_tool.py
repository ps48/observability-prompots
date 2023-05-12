from fastapi import FastAPI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from opensearchpy import OpenSearch
from pydantic import BaseModel
import requests
from datetime import datetime

app = FastAPI()
# auth = ("USER", "PASS")
prometheus_url = "http://localhost:9090"


def get_all_metrics():
    response = requests.get(f"{prometheus_url}/api/v1/label/__name__/values")

    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        return f"Query failed with status code {response.status_code}: {response.text}"


def query_prometheus(metric_name):
    query_string = metric_name + "[5m]"
    current_time = datetime.now().strftime("%s")

    response = requests.post(
        f"{prometheus_url}/api/v1/query",
        params={"query": query_string, "time": current_time},
    )

    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        return f"Query failed with status code {response.status_code}: {response.text}"


tools = [
    Tool(
        name="Get all Prometheus metric names",
        func=lambda string: get_all_metrics(),
        description="use when you want to fetch names of all Prometheus metrics",
    ),
    Tool(
        name="Query Prometheus",
        func=lambda string: query_prometheus(string),
        description="use when you want to fetch last 5 mins of a specific Prometheus metric",
    ),
]

local_llm = OpenAI(temperature=0, verbose=True)
memory = ConversationBufferMemory(memory_key="chat_history")

system_prompt = """
Assistant should follow the below steps to check if a there is an anomaly in a metric:

1. Use the metric name and validate if it exists in the list of prometheus metrics.
2. Query prometheus with the metric name and check values of last 5 mins.
3. Check the trend in values and find an anomaly. If found report it. 
"""


agent = initialize_agent(
    tools,
    local_llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True,
)

# # print(agent.agent.from_llm_and_tools(local_llm, tools).llm_chain.prompt)
agent.agent.llm_chain.prompt.template = """
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant always follows the steps below to check if a there is an anomaly in a metric:

1. Assistant checks validity of the prometheus metric name
2. Query prometheus metric and checks values of last 5 mins.
3. Finally, it checks the trend in values and find an anomaly. If found report it. 

TOOLS:
------

Assistant has access to the following tools:

> Get all Prometheus metric names: use when you want to fetch names of all Prometheus metrics
> Query Prometheus: use when you want to fetch last 5 mins of a specific Prometheus metric

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [Get all Prometheus metric names, Query Prometheus]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""


# print(agent.agent.llm_chain.prompt.template)

# agent.agent.llm_chain.prompt = new_prompt


print(agent.run("Is there an anomaly in go_memstats_alloc_bytes_total metric?"))
# print(agent.run("Is go_memstats_alloc_bytes_total a valid prometheus metric name?"))
# print(agent.run("Show all metric names from prometheus?"))

