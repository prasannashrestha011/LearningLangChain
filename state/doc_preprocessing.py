from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
import os

file_path=os.path.abspath("./assets/attention.pdf")
loader=PyPDFLoader(file_path)
doc=loader.load()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=400,add_start_index=True)
all_splits=text_splitter.split_documents(doc)
