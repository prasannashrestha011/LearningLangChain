from langchain.chat_models import init_chat_model
from langchain.tools.tool_node import _ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
load_dotenv()
db=SQLDatabase.from_uri("sqlite:///Chinook.db")
llm=init_chat_model(model="gemini-2.5-flash",model_provider="google_genai")
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
tools=toolkit.get_tools()

get_schema_tool=next(tool for tool in tools if tool.name=="sql_db_schema")
get_schema_node=_ToolNode([get_schema_tool],name="get_schema")

run_query_tool=next(tool for tool in tools if tool.name=="sql_db_query")
run_schema_node=_ToolNode([run_query_tool],name="run_query")

tool_call={
        "name":"sql_db_list_tables",
        "args":{},
        "id":"abc123",
        "type":"tool_call"
    }
tool_call_message=AIMessage(content="",tool_call=[tool_call])
list_table_tool=next(tool for tool in tools if tool.name=="sql_db_list_tables")
print(list_table_tool.invoke(tool_call))
