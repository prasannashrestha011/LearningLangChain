from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()
db=SQLDatabase.from_uri("sqlite:///Chinook.db")
llm=init_chat_model(model="gemini-2.5-flash",model_provider="google_genai")
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
tools=toolkit.get_tools()

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent=create_agent(
        llm,
        tools,
        system_prompt=system_prompt
        )

while True:
    question =input("H: ")
    if question.lower()== "exit":
        break
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
