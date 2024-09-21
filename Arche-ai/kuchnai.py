from llms import Cohere
from agents import Agent
from tools import Tool, get_current_time, web_search

def gcd(a:int, b:int):
    """
    Calculate the Greatest Common Divisor (GCD) of two numbers using the Euclidean algorithm.

    Parameters:
    a (int): The first number.
    b (int): The second number.

    Returns:
    int: The GCD of the two numbers.
    """
    while b:
        a, b = b, a % b
    return a+b

# Define the tools using the OwnTool class
gcd_tool = Tool(
    func=gcd,
    description="Provides the gcd of two provided numbers",
    returns_value=True
)

web_tool = Tool(
    func=web_search,
    description="Provides the current web results from Google for the given query, best for getting real-time data.",
    returns_value=True
)

time_tool = Tool(
    func=get_current_time,
    description="Provides the current time.",
    returns_value=True
)

# Initialize the language model instance

llm_instance = Cohere()

Chatbot = Agent(
        llm=llm_instance,
        name="ChatBot",
        tools=[web_tool, time_tool],
        description="a powerfull ai agent",
        expected_output="",
        task="",
        memory=False,
        max_memory_tokens=50,
        verbose=True,
        memory_history_offset=50
    )


while True:
    # Create the agent with multiple tools
    Chatbot.task = input(">>>")
    result = Chatbot.rollout()