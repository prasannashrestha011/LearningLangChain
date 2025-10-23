from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from typing import List, TypedDict
from langchain_core.documents import Document
from db import retriever
from dotenv import load_dotenv
import os, getpass

# Load API key
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialize LLM
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.environ["GOOGLE_API_KEY"]
)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Read the context from PDF and answer the query. If unknown, say so."),
    ("user", "Context: {context}\n\nQuestion: {query}")
])

# Simple in-memory store
store = InMemoryStore()
namespace = ("user-123", "chat")

# Message counter
message_counter = 0

# Chat state
class ChatState(TypedDict):
    user_input: str
    context: List[Document]
    last_response: str

# Nodes
def add_context(state: ChatState):
    context = retriever.invoke(state["user_input"])
    return {"context": context}

def answer_node(state: ChatState):
    global message_counter
    
    # Format document context
    doc_context = "\n\n".join([doc.page_content for doc in state["context"]])
    
    # previous conversations
    try:
        memories = store.search(namespace, query=state["user_input"], limit=5)
        memory_text = "\n".join([
            f"Previous Q: {m.value.get('content', '')}\nA: {m.value.get('response', '')}" 
            for m in memories
        ]) if memories else ""
    except:
        memory_text = ""
    
    full_context = f"{doc_context}\n\nPrevious conversation:\n{memory_text}" if memory_text else doc_context
    
    message = prompt_template.invoke({
        "context": full_context,
        "query": state["user_input"]
    })
    response = llm.invoke(message)
    
    try:
        message_counter += 1
        store.put(
            namespace,
            f"msg-{message_counter}",
            {"content": state["user_input"], "response": str(response.content)}
        )
    except Exception as e:
        print(f"Memory save error: {e}")
    
    return {"last_response": str(response.content)}

graph = (
    StateGraph(ChatState)
    .add_node("add_context", add_context)
    .add_node("answer_node", answer_node)
    .add_edge(START, "add_context")
    .add_edge("add_context", "answer_node")
    .add_edge("answer_node", END)
    .compile()
)

print("Chat started! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    try:
        result = graph.invoke({
            "user_input": query,
            "context": [],
            "last_response": ""
        })
        print(f"AI: {result['last_response']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
