from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
load_dotenv()

model=init_chat_model(model="gemini-2.5-flash",model_provider="google_genai")
embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

store=Chroma(
        collection_name="attention_vectors",
        embedding_function=embedding,
        persist_directory="./chroma_db")

"""Loading multiple pdf files"""
file_path=os.path.abspath("../assets/attention.pdf")
loader=PyPDFLoader(file_path)
doc=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=400,add_start_index=True)
all_splits=text_splitter.split_documents(doc)

vector_ids=store.add_documents(all_splits)
retriver=store.as_retriever()

while True:
    query=input("Enter your query")
    if query=="exit":
        break       
    retrived_doc=retriver.(query)
    context="\n\n".join([doc.page_content for doc in retrived_doc])

    prompt=ChatPromptTemplate.from_messages([
        ("system","Use the following context to answer the user queries"),
        ("user","{context},{query}")
        ])
    messages=prompt.invoke({"context":context,"query":query})
    response=model.invoke(messages)
    print(response.content)
