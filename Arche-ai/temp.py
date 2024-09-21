from llms import Cohere
from agents import Agent, TaskForce
from tools import Tool, get_current_time, web_search

def factorial(n: int) -> int:
    """
    Calculate the factorial of a given number.

    Parameters:
    n (int): The number to calculate the factorial of.

    Returns:
    int: The factorial of the number.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Define the tools using the OwnTool class
factorial_tool = Tool(
    func=factorial,
    description="Calculates the factorial of a given number.",
    returns_value=True
)

time_tool = Tool(
    func=get_current_time,
    description="Provides the current time.",
    returns_value=True
)

web_tool = Tool(
    func=web_search,
    description="Provides the current web results from Google for the given query, best for getting real-time data.",
    returns_value=True
)

# Initialize the language model instance
llm_instance = Cohere()

# Define agents with specific tasks and knowledge bases
Analyst = Agent(
    llm=llm_instance,
    name="Analyst",
    description="Analyzes a report based on information given and grammer",
    expected_output="",
    task="",
    verbose=True,
    memory=False
)

Writer = Agent(
    llm=llm_instance,
    name="Writer",
    description="writes a report for given info",
    expected_output="",
    skills="Over Powered Writing skill",
    task="",
    verbose=True,
    memory=False
)

Researcher = Agent(
    llm=llm_instance,
    name="Researcher",
    description="Performs web searches to gather real-time information.",
    expected_output="provide raw links and info",
    tools=[web_tool],
    skills="Can do Web Search",
    task="",
    verbose=True,
    memory=False
)

# Create a TaskForce with these agents
agent_network = TaskForce(agents=[Researcher, Writer, Analyst],
llm=llm_instance,
name="News_Distributer",
description="A team for research and report writing.",
verbose=True)

# Example 1: Simple task to solve a mathematical problem
# responses_math = agent_network.rollout("Calculate the factorial of 5")

# Example 2: Task involving Python development
# responses_dev = agent_network.rollout("Write a Python function to sort a list of numbers using bubble sort.")

# # Example 3: Task requiring a web search and summarization
# responses_web = agent_network.rollout("who won the T20 Wordcup 2024 and how??")

# # Example 4: Combination task involving multiple agents
while True:
    responses_combined = agent_network.rollout(input(">>> "), max_iterations=10)

# print("\nExample 1 Response:", responses_math)
# print("\nExample 2 Response:", responses_dev)
# print("\nExample 3 Response:", responses_web)
# print("\nExample 4 Response:", responses_combined)
