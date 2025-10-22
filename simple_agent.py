from bs4.filter import SoupStrainer 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.vectorstores import InMemoryVectorStore
from embeddings import embeddings
from langchain_classic import hub
from langchain_core.documents import Document 
from typing import List,TypedDict
from model_setup import model
#work flow imports
from langgraph.graph import START,StateGraph
strainer=SoupStrainer(class_=("post-title","post-header","post-content"))
loader=WebBaseLoader(
web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
bs_kwargs={"parse_only":strainer}
        )

docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter( 
                                            chunk_size=1000,
                                             chunk_overlap=200,
                                             add_start_index=True)
all_splits=text_splitter.split_documents(docs)
print("Total splits ",len(all_splits))

"""Initialize the In memory vector DB"""
vector_store=InMemoryVectorStore(embedding=embeddings)
doc_ids=vector_store.add_documents(all_splits)

prompt=hub.pull("rlm/rag-prompt")
example_message=prompt.invoke({"context":"(context goes here)","question":"(question goes here)"})

"""State"""
class State(TypedDict):
    question:str 
    context:List[Document]
    answer:str

"""Nodes"""
def retrive(state:State):
    retrived_docs=vector_store.similarity_search(state['question'])
    return {"context":retrived_docs}
def generate(state:State):
    docs_content="\n\n".join(doc.page_content for doc in state['context'])
    messages=prompt.invoke({"question":state['question'],"context":docs_content})
    response=model.invoke(messages)
    return {"answer":response} 
"""Control work flow""" 
graph_builder=StateGraph(State).add_sequence([retrive,generate])
graph_builder.add_edge(START,"retrive")
graph=graph_builder.compile()


for step in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")
