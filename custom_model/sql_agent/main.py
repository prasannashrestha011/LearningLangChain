"""
SQL Agent using LangChain, LangGraph, and Gemini

This agent can:
- List available database tables
- Get table schemas
- Generate SQL queries from natural language
- Validate and check queries for common mistakes
- Execute queries and return results
"""

from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.tools.tool_node import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState
from langgraph.graph.state import END, StateGraph
from langgraph.prebuilt import ToolNode as _ToolNode

# Load environment variables
load_dotenv()

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize Gemini LLM
llm = init_chat_model(model="gemini-2.0-flash-exp", model_provider="google_genai")

# Create SQL toolkit and extract tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Extract specific tools
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = _ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = _ToolNode([run_query_tool], name="run_query")


def list_tables(state: MessagesState):
    """List all available tables in the database"""
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "list_tables_call",
        "type": "tool_call"
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(content=f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}


def call_get_schema(state: MessagesState):
    """Call the schema retrieval tool"""
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# System prompt for query generation
generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)


def generate_query(state: MessagesState):
    """Generate SQL query from natural language question"""
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt
    }
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


# System prompt for query validation
check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)


def check_query(state: MessagesState):
    """Validate and potentially correct the generated SQL query"""
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Extract the query from the last message's tool call
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    
    # Preserve the message ID for continuity
    response.id = state["messages"][-1].id

    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    """Decide whether to continue to query validation or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


# Build the agent graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("list_tables", list_tables)
builder.add_node("call_get_schema", call_get_schema)
builder.add_node("get_schema", get_schema_node)
builder.add_node("generate_query", generate_query)
builder.add_node("check_query", check_query)
builder.add_node("run_query", run_query_node)

# Add edges
builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

# Compile the agent
agent = builder.compile()


# Helper function to run the agent
def run_agent(question: str):
    """Run the SQL agent with a natural language question"""
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"{'='*70}\n")
    
    # Invoke the agent with the question
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
    # Print the final response
    for message in result["messages"]:
        if hasattr(message, 'content') and message.content and not hasattr(message, 'tool_calls'):
            print(f"Answer: {message.content}\n")
    
    return result


# Example usage
if __name__ == "__main__":
    # Make sure you have:
    # 1. .env file with GOOGLE_API_KEY
    # 2. Chinook.db in the same directory
    
    example_questions = [
        "How many customers are there?",
        "What are the top 5 albums by number of tracks?",
        "Which artist has the most albums?",
        "Show me all employees who work in Sales",
        "What is the total revenue from invoices in 2010?",
    ]
    
    for question in example_questions:
        try:
            run_agent(question)
        except Exception as e:
            print(f"Error: {str(e)}\n")
