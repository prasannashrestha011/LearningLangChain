from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase 
from langchain_community.agent_toolkits import SQLDatabaseToolkit
load_dotenv()
model = init_chat_model(
    model="nvidia/nemotron-nano-9b-v2:free",
    model_provider="openai",       
    base_url="https://openrouter.ai/api/v1"  
)

db=SQLDatabase.from_uri("sqlite:///agent.db")

tool_kit=SQLDatabaseToolkit(db=db,llm=model)
tools=tool_kit.get_tools()


list_tools=next(tool for tool in tools if tool.name=="sql_db_schema")
response=list_tools.run("employees")
print(response)
