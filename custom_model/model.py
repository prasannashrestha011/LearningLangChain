from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader 
from bs4.filter import SoupStrainer

from typing import List,TypedDict
from langchain_core.documents import Document

"""Work flow imports"""
from langchain_classic import hub
from langgraph.graph import StateGraph,START
load_dotenv()


model=init_chat_model("gemini-2.5-flash",model_provider="google_genai")

""" Initializing gemini embeddings"""
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

"""Initializing vector store"""
vector_store=InMemoryVectorStore(embedding=embeddings)

"""scraping the blog posts and splitting the text to store in vector db"""
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)

strainer=SoupStrainer(class_={"post-title","post-header","post-content"})
loader=WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
                  bs_kwargs={"parse_only":strainer})
doc=loader.load()

all_splits=text_splitter.split_documents(doc)
vector_ids=vector_store.add_documents(all_splits)
"""Preparaing prompt template"""
prompt=hub.pull("rlm/rag-prompt")

"""Defining the work flow"""
class State(TypedDict):
    question:str 
    context:List[Document]
    answer:str 

def retriver(state:State):
    context=vector_store.similarity_search(state['question'])
    return {"context":context}

def generator(state:State):
    doc_content="\n\n".join(doc.page_content for doc in state['context'])
    messages=prompt.invoke({"question":state['question'],"context":doc_content})
    response=model.invoke(messages)
    return {"answer":response.content}

graph_builder=StateGraph(State).add_sequence([retriver,generator])
graph_builder.add_edge(START,"retriver")
graph=graph_builder.compile()


for step in graph.stream(
    {"question": "What is Self Reflection?"}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")
