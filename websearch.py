from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are NgGrok AI, you answer users' queries concisely. "
            "If you use web search results, summarize them and only return the relevant info.",
        ),
        MessagesPlaceholder("messages"),
    ]
)

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

duck_tool_wrapper = DuckDuckGoSearchAPIWrapper()


# Define tool using @tool decorator for better compatibility
@tool
def duckduckgo_search(query: str) -> str:
    """Search the web for relevant information using DuckDuckGo.

    Args:
        query: The search query string

    Returns:
        Search results as a string
    """
    print(f"🟢 Web search used for query: {query}")
    return duck_tool_wrapper.run(query)


# Bind tools to LLM
llm_with_tools = llm.bind_tools([duckduckgo_search])
agent = prompt | llm_with_tools

# Creating tool node
tool_node = ToolNode([duckduckgo_search])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_agent(state: AgentState):
    response = agent.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "end"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "end": END}  # Use string "end"
)
workflow.add_edge("tools", "agent")
# REMOVE: workflow.add_edge("agent", END)  # This conflicts with conditional_edges!

app = workflow.compile()

messages = []
while True:
    query = input("H: ")
    if query.lower() == "exit":
        break

    messages.append(HumanMessage(content=query))
    response = app.invoke({"messages": messages})

    # Update messages with the full response
    messages = response["messages"]

    ai_msg = messages[-1]
    print("AI: ", ai_msg.content)
