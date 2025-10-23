from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from typing import List ,TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate 
from db import retriever
from dotenv import load_dotenv
import os
import getpass
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


llm = init_chat_model("gemini-2.5-flash",
model_provider="google_genai", api_key=os.environ["GOOGLE_API_KEY"])

"""Defining prompt template"""
prompt_template=ChatPromptTemplate.from_messages([
    ("system","You are an AI assistant. Your Job is to read the context from PDF doc and answer based on user query. If you dont know the answer, sincerely say you dont know."),
    ("user","Given :{context}, answer my follwing query:{query}")
    ])
class ChatState(TypedDict):
    user_input:str
    context:List[Document]
    last_response:str

"""Defining nodes"""

def add_context(state:ChatState):
    """Retrieving relevant documents from vector store"""
    context=retriever.invoke(state["user_input"])
    state["context"]=context
    return {"context":state["context"]}


def answer_node(state:ChatState):
    context=state['context']
    message=prompt_template.invoke({"context":context,"query":state["user_input"]})
    response=llm.invoke(message)
    return {"last_response":str(response.content)}
    
graph=(StateGraph(ChatState).add_node("add_context",add_context)
       .add_node("answer_node",answer_node)
       .add_edge(START,"add_context")
       .add_edge("add_context","answer_node")
       .add_edge("answer_node",END).compile()
       ) 
while True:
    query=input("Enter your query")
    if query.lower=="exit":
        break
    result=graph.invoke({"user_input":query,"context":[],"last_response":""})
    print(result["last_response"])
