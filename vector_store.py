"""
Using In memory store for light weight work load

-In this code, we first stored the chunks into In memory vector DB. 
-The vector DB provide several operations , and one of them is similarity_search.
-We then query a document to find similar documents in the store.
"""
from langchain_core.vectorstores import InMemoryVectorStore 
from embeddings import embeddings
from semantic_search import all_splits
import asyncio
vector_store=InMemoryVectorStore(embeddings)
ids=vector_store.add_documents(documents=all_splits)

result=vector_store.similarity_search_with_score(
        "Explain different programming approach"
        )
doc,score=result[0]
print("Score: ",score)
print(doc)

