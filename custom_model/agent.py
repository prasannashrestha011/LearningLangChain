from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import requests
from dotenv import load_dotenv
load_dotenv()
# Initialize Gemini Flash 2.5
llm = init_chat_model(
    "google_genai:gemini-2.0-flash-exp",
    temperature=0
)

# Test Tool 1: Weather (Real API)
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city
    """
    try:
        # Using wttr.in - a free weather service, no API key needed!
        response = requests.get(f"https://wttr.in/{city}?format=%C+%t+%h", timeout=5)
        if response.status_code == 200:
            return f"Weather in {city}: {response.text}"
        return f"Could not fetch weather for {city}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create the agent
agent = create_agent(
    model=llm,
    tools=[
        get_weather,
    ],
    system_prompt="""You are a helpful AI assistant with access to real-world tools.

Available tools:
- Weather information for any city
- Web search for current information
- Mathematical calculations
- File reading
- Current time in any timezone

Use these tools when needed to provide accurate, up-to-date information.
Think step by step about which tools to use."""
)

# Test 1: Weather
print("\n1. Weather Query:")
print("-" * 60)
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather like in Bhaktapur?"}]
})
print(response["messages"][-1].content)

