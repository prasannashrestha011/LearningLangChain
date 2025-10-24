from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
model=init_chat_model(model="llama-3.3-70B-versatile",model_provider="groq")

while True:
    query=input("Write your query")
    if query.lower()=="exit":
        break
    response=model.invoke([HumanMessage(content=query)])
    print("AI: ",response.content)
