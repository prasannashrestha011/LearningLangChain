from dotenv import load_dotenv
import os
import getpass

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage,SystemMessage
model=init_chat_model("gemini-2.5-flash",model_provider="google_genai")

messages=[
        SystemMessage(content="You are a poem writer, user will ask you to write poem"),
        HumanMessage(content="Write a poem about a bird")
        ]
for token in model.stream(messages):
    print(token.content,end='|')
