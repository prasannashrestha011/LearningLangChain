from langchain_core.runnables import chain
from langchain_core.documents import Document 
from vector_store import vector_store

@chain 
def retriver(query:str):
    return vector_store.similarity_search_with_score(query,k=1)

response=retriver.batch(
        ["Advantage and Disadvantage of C programming language",
         "Why C programming is used",
         "Difference between Top down and Bottom up approach"]
        ) 
print("="*20)
print(response)
print("="*20)
