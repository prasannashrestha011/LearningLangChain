from langchain.agents import create_agent 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool
load_dotenv()

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
model = init_chat_model(
    model="nvidia/nemotron-nano-9b-v2:free",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1"
)
duck_tool=DuckDuckGoSearchAPIWrapper()
web_search_tool=Tool(
        name="duckduckgo_search",
        func=duck_tool.run, 
        description="Useful for searching the web for relevant information."

        )

model = create_agent(
        model=model,
        tools=[web_search_tool],
    system_prompt="You are NgGrok AI, you answer users query consisely"
)
while True:
    query=input("H: ")
    if query.lower()=="exit":
        break
    response=model.invoke({"messages":[{"role":"user","content":query}]})
    for msg in response["messages"]:
        if msg.type == "ai":  # or isinstance(msg, AIMessage)
            print("AI: ",msg.content)
