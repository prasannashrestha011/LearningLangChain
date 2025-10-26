    """
Embeddings pdf content into vectors which can be used for semantic search when using vector store

The characters are transformed into array of numeric values

When performing search operation, similarity metrics such as cosine similarity can be used to semantically match documents from vector DB.

"""
from dotenv import load_dotenv 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from semantic_search import all_splits

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_1=embeddings.embed_query(all_splits[0].page_content)
vector_2=embeddings.embed_query(all_splits[1].page_content)

print(vector_1[:0])
