from llms import Gemini
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
llm_instance = Gemini()

# Define agents with specific tasks and knowledge bases
PythonDevAgent = Agent(
    llm=llm_instance,
    tools=[web_tool],
    name="Jarvis",
    description="a conversational ai agent which gives witty responses like jarvis from the movies and try to give answers to talks in a Jarvis manner using movie dialogs, your sole user's name is Anant Sharma call him using the word sir and always reply with a pinch or sarcasm and wit.",
    expected_output="",
    task="",
    verbose=True,
)

MathSolverAgent = Agent(
    llm=llm_instance,
    name="Time_tracker",
    description="gives current time using tools",
    expected_output="",
    tools=[time_tool],
    task="",
    verbose=True,
)

WebSearchAgent = Agent(
    llm=llm_instance,
    name="Web_Search",
    description="Performs web searches to gather real-time information.",
    expected_output="",
    tools=[web_tool],
    task="",
    verbose=True,
)

# Create a TaskForce with these agents
agent_network = TaskForce(agents=[WebSearchAgent, MathSolverAgent],
llm=llm_instance,
name="Jarvis",
description="a ai agent which gives witty responses like jarvis from the movies and try to give answers to talks in a Jarvis manner using movie dialogs, your sole user's name is Anant Sharma call him using the word sir and always reply with a pinch or sarcasm and wit.",
verbose=True)

# Example 1: Simple task to solve a mathematical problem
# responses_math = agent_network.rollout("Calculate the factorial of 5")

# Example 2: Task involving Python development
# responses_dev = agent_network.rollout("Write a Python function to sort a list of numbers using bubble sort.")

# # Example 3: Task requiring a web search and summarization
# responses_web = agent_network.rollout("who won the T20 Wordcup 2024 and how??")

# # Example 4: Combination task involving multiple agents
while True:
    responses_combined = agent_network.rollout(input(">>> "), max_iterations=3)

# print("\nExample 1 Response:", responses_math)
# print("\nExample 2 Response:", responses_dev)
# print("\nExample 3 Response:", responses_web)
# print("\nExample 4 Response:", responses_combined)
