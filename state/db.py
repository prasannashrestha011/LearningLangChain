from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from doc_preprocessing import all_splits
from dotenv import load_dotenv 
load_dotenv()
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
store=Chroma(collection_name="state_db",embedding_function=embeddings,persist_directory="./memory_db")
store.add_documents(all_splits)
retriever=store.as_retriever()


