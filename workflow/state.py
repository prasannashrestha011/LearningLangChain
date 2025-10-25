from typing import TypedDict

from langchain_core.documents import Document 
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
class PDFQAState(TypedDict):
    questions:str 
    pdf_chunks:list[Document] | None 
    vector_db:Chroma | None
    relevant_chunks:list[Document] | None
    answer:AIMessage | None 

