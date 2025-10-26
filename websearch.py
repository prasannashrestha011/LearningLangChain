from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

duck_tool=DuckDuckGoSearchAPIWrapper()
duck_tool.run("Who is Messi")
