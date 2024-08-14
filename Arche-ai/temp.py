from llms import Cohere
from agents import Agent
from tools import OwnTool, get_current_time, web_search

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
    print(a)
    # return a+b

# Define the tools using the OwnTool class
gcd_tool = OwnTool(
    func=gcd,
    description="Provides the gcd of two provided numbers",
    params={"l": {"type": "int", "description": "The first number includes only number such as 1 ,2"}, "h": {"type": "int", "description": "The second number such as 1,2 ,3"}},
    returns_value=False
)

web_tool = OwnTool(
    func=web_search,
    description="Provides the current web results from Google for the given query, best for getting real-time data.",
    params={"hello": {"type": "string", "description": "The query to do search for"}},
    returns_value=True
)

time_tool = OwnTool(
    func=get_current_time,
    description="Provides the current time.",
    returns_value=True
)

# Initialize the language model instance

llm_instance = Cohere()

Chatbot = Agent(
        llm=llm_instance,
        tools=[web_tool, gcd_tool, time_tool],
        name="ChatBot",
        description="a powerfull ai agent",
        sample_output="",
        task="",
        verbose=True,
    )


while True:
    # Create the agent with multiple tools
    Chatbot.task = input(">>>")
    Chatbot.run()
    # print(result)
