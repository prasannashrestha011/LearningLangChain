from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from state import PDFQAState
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END

load_dotenv()
def load_split_pdfs(state:PDFQAState)->PDFQAState: 
    loader=PyPDFLoader("../assets/UNIT-1.pdf")
    doc=loader.load()

    text_splitters=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    all_splits=text_splitters.split_documents(doc)

    state["pdf_chunks"]=all_splits
    return state


def embed_doc(state:PDFQAState)->PDFQAState:
    embeddings=HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-small")
    store=Chroma(embedding_function=embeddings,collection_name="workflow_db",persist_directory="./workflow_db")
    if state["pdf_chunks"]:
        state["vector_db"]=store.from_documents(state["pdf_chunks"],embeddings)
    return state


def retrieve_chunks(state:PDFQAState)->PDFQAState:
    store=state["vector_db"]
    if store:
        similar_chunks=store.similarity_search(state["questions"])
    state["relevant_chunks"]=similar_chunks
    return state

def prepare_answer(state:PDFQAState)->PDFQAState:
    llm=init_chat_model(model="llama-3.3-70B-versatile",model_provider="groq")
    if state["relevant_chunks"]:
        context="\n\n".join(doc.page_content for doc in state["relevant_chunks"])
    prompt=f"Answer the question using the context:{context},{state["questions"]}"
    response=llm.invoke([HumanMessage(content=prompt)])
    state["answer"]=response
    return state

graphbuilder=StateGraph(PDFQAState).add_sequence([load_split_pdfs,embed_doc,retrieve_chunks,prepare_answer])
graphbuilder.add_edge(START,"load_split_pdfs")
graphbuilder.add_edge("prepare_answer",END)
graph=graphbuilder.compile()

while True:
    input:PDFQAState={
            "answer":None,
            "relevant_chunks":None,
            "vector_db":None,
            "pdf_chunks":None,
            "questions":str(__builtins__.input("Human: ")),
            }
    if input["questions"].lower()=="exit":
        break
    response=graph.invoke(input)
    print("AI: ",response["answer"].content)
