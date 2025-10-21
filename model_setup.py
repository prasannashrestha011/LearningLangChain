
# model_setup.py
from dotenv import load_dotenv
import os
import getpass
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialize model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
