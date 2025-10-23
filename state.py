from typing import List, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph,START 
from pydantic import BaseModel
"""Defining a state """
class ChatState(BaseModel):
    user_input:str 
    context:List[Document]
    last_response:Optional[str]

def add_context(state:ChatState):
    state.context.append(new_context)
    return state.model_dump()

def answer_node(state:ChatState):
    state.last_response=state.last_response
    return state.model_dump()

graph=(
        StateGraph(ChatState).add_node("add_context",add_context).add_node("answer",answer_node)
        )
