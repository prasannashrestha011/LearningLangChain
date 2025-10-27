import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
print(os.getenv("DEEPSEEK_API_KEY"))

llm = init_chat_model(
    model="deepseek/deepseek-chat-v3.1:free",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
response = llm.invoke("Who model are you?")
print(response.content)
