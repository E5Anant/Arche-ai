from llms import Gemini
from agents import Agent, AgentNetwork
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

llm_instance = Gemini()

Chatbot1 = Agent(
    llm=llm_instance,
    tools=[web_tool],
    name="Web Researcher",
    description="An agent that specializes in web searches.",
    sample_output="",
    task="",
    verbose=True,
)

Chatbot2 = Agent(
    llm=llm_instance,
    tools=[gcd_tool, time_tool],
    name="Math and Time Expert",
    description="An agent that can handle math and time-related queries.",
    sample_output="",
    task="",
    verbose=True,
)

agent_network = AgentNetwork(agents=[Chatbot1, Chatbot2], tasks = [
    {"agent": "Chatbot1", "request": "price of bitcoin??"},
    {"agent": "Chatbot2", "request": "what is the time now??"},
],
 verbose=False)

while True:
    response = agent_network.run()
    print(response)
