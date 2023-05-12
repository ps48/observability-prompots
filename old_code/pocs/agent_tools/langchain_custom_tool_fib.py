from fastapi import FastAPI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from opensearchpy import OpenSearch
from pydantic import BaseModel

app = FastAPI()
host = "localhost"
port = 9200
auth = ("admin", "admin")

client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_compress=True,  # enables gzip compression for request bodies
    http_auth=auth,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)


def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


def sort_string(string):
    return "".join(sorted(string))


def query_os(string):
    response = client.transport.perform_request(
        "POST", "/_plugins/_ppl", body={"query": string}
    )
    return str(response)


tools = [
    Tool(
        name="Fibonacci",
        func=lambda n: str(fib(int(n))),
        description="use when you want to calculate the nth fibonacci number",
    ),
    Tool(
        name="Sort String",
        func=lambda string: sort_string(string),
        description="use when you want to sort a string alphabetically",
    ),
    Tool(
        name="Query OpenSearch",
        func=lambda string: query_os(string),
        description="use when you want to Query OpenSearch with PPL",
    ),
]

local_llm = OpenAI(temperature=0, verbose=True)
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    local_llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True,
)


class Data(BaseModel):
    question: str


@app.post("/predict")
async def predict(data: Data):
    return agent.run(data.question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
