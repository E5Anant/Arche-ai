from llms import GroqLLM
from tools import OwnTool
from typing import Type, List, Optional
import json
from colorama import Fore, Back, Style
import concurrent.futures

def convert_function(func_name, description, **params):
    """Converts function info to a JSON function schema."""

    function_dict = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

    for param_name, param_info in params.items():

        try:
            if "description" not in param_info:
                param_info["description"] = f"Description for {param_name} is missing. Defaulting to {param_info}"
                descri = f"{param_info}"
            else:
                descri = param_info["description"]
        except:
            descri = f"{str(param_info)}"

        try:
            param_type = param_info.get("type", "string")

            valid_types = ("string", "number", "boolean", "enum", "array", "object", "integer")
            if param_type not in valid_types:
                print(f"Warning: Invalid type '{param_type}' for '{param_name}'. Defaulting to 'string'.")
                param_type = "string"
        except:
            param_type = "string"

        param_properties = {
            "type": param_type,
            "description": descri
        }

        if param_type == "enum":
            if "options" not in param_info:
                raise ValueError(f"Parameter '{param_name}' of type 'enum' requires an 'options' list.")
            param_properties["enum"] = param_info["options"]
        try:
            if "default" in param_info:
                param_properties["default"] = param_info["default"]
        except:
            pass

        try:
            if param_info.get("required", False):
                function_dict["function"]["parameters"]["required"].append(param_name)
        except:
            function_dict["function"]["parameters"]["required"].append(param_name)

        function_dict["function"]["parameters"]["properties"][param_name] = param_properties

    return function_dict

class Agent:
    def __init__(
        self,
        llm: Type[GroqLLM],
        tools: List[OwnTool],
        name: str = "Agent",
        description: str = "A helpful AI agent.",
        sample_output: str = "Concise and informative text.",
        task: str = "Ask me a question or give me a task.",
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.name = name
        self.description = description
        self.sample_output = sample_output
        self.task_to_do = task
        self.verbose = verbose

        self.llm.__init__(system_prompt=f"You are {self.name}, {self.description}.")

        self.all_functions = []

        for i in self.tools:
            x = convert_function(func_name=i.func.__name__, description=i.description, **i.params)
            self.all_functions.append(x)

    def _run_no_tool(self) -> str:
        """Handles tasks without any tools."""
        self.llm.__init__(system_prompt=f"""
You are {self.name}, {self.description}.

### OUTPUT STYLE:
{self.sample_output}

***If output style not mentioned, generate in markdown format.***
""")
        return self.llm.run(self.task_to_do)

    def _run_with_tools(self) -> str:
        """Handles tasks that require using tools."""
        self.tools_info = "\n".join([
            f"Tool Name: {tool.func.__name__} - {tool.description}\nTool Parameters: {tool.params}"
            for tool in self.tools
        ])

        self.llm.__init__(system_prompt=f"""
You are an AI assistant designed to generate JSON responses based on provided tools.

Your task is to understand the tools, their parameters, and use them appropriately.

Available Tools:
llm_tool - A default tool that provides AI-generated text responses and it cannot answer real-time queries because of the knowledge cut off of October 2019.
{self.all_functions}

Instructions:
1. Read the task carefully.
2. Identify the required tool parameters.
3. Respond with a JSON object containing the tool_name and parameter.
4. Only provide the JSON response.
5. You are only trained to give json response and not text or conversation.

JSON Structure:
{{
    "func_calling": [
        {{
            "tool_name": "<tool_name>",
            "parameter": "<tool_params>"
        }}
    ]
}}

Example:
Task: Get the weather for New York
Response:
{{
    "func_calling": [
        {{
            "tool_name": "weather_tool",
            "parameter": "New York"
        }}
    ]
}}

For tools with no parameters:
{{
    "func_calling": [
        {{
            "tool_name": "time_tool",
            "parameter": ""
        }}
    ]
}}

Conversation with llm_tool:

Example:
User: Who are you?
Response:
{{
    "func_calling": [
        {{
            "tool_name": "llm_tool",
            "parameter": "Who are you?"
        }}
    ]
}}

***How to deal with double parameter tools***

- Remember this is just an example the tools and params varies according to the details given above

Example:
User: What is the sum of 43675 and 547?

Response:
{{
    "func_calling": [
        {{
            "tool_name": "add",
            "parameter": {{
                "a": 43675,
                "b": 547
            }}
        }}
    ]
}}


""")

        response = self.llm.run(self.task_to_do).strip()
        if self.verbose:
            print(f"{Fore.YELLOW}Raw LLM Response:{Style.RESET_ALL} {response}")

        try:
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            action = json.loads(response)

            results = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_tool = {
                    executor.submit(self._call_tool, call): call for call in action.get("func_calling", [])
                }
                for future in concurrent.futures.as_completed(future_to_tool):
                    call = future_to_tool[future]
                    try:
                        tool_name, tool_response = future.result()
                        results[tool_name] = tool_response
                        if self.verbose:
                            print(f"{Fore.GREEN}Tool Response ({tool_name}):{Style.RESET_ALL} {tool_response}")
                    except Exception as e:
                        if self.verbose:
                            print(f"{Fore.RED}Error calling tool {call['tool_name']}:{Style.RESET_ALL} {str(e)}")
                        results[call['tool_name']] = f"Failed to get info: {str(e)}."

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"{Fore.RED}JSON Decode Error:{Style.RESET_ALL} {str(e)}")
        except Exception as e:
            if self.verbose:
                print(f"{Fore.RED}Error:{Style.RESET_ALL} {str(e)}")

        try:
            if self.verbose:
                print()
                print(f"{Fore.GREEN}Tool_RESULTS:\n{results}{Style.RESET_ALL}")
                print()
        except:
            pass

        # Summarization Prompt
        self.llm.__init__(system_prompt=f"""
You are {self.name}, an AI agent. You are provided with output from the tools in JSON format. Your task is to use this information to give the best possible answer to the query. Reply in a natural language style, only in text, and to the point. Do not reply in JSON.

### TOOLS:
llm_tool - If this tool is used, you must answer the user's query in the best possible way.
{self.all_functions}

### OUTPUT STYLE:
{self.sample_output}

##Instructions:
- If the output style is not mentioned, just reply in the best possible way in only text form and not JSON.
- You are no longer generating JSON responses. Provide a natural language summary based on the information from the tools.

""")

        try:
            summary = self.llm.run(f"[QUERY]\n{self.task_to_do}\n\n[TOOLS]\n{results}")
            if self.verbose:
                print("Final Response:")
                print(summary)
                print()

            # Check if the response is still in JSON format
            if summary.startswith("{") and summary.endswith("}"):
                try:
                    json_summary = json.loads(summary)
                    if "func_calling" in json_summary:
                        summary = "It seems there was an issue with the response. Please try again."
                except json.JSONDecodeError:
                    pass

            return summary
        except Exception as e:
            summary = self.llm.run(f"[QUERY]\n{self.task_to_do}")
            if self.verbose:
                print("Final Response:")
                print(summary)
                print()
            return summary

    def _call_tool(self, call):
        tool_name = call["tool_name"]
        query = call["parameter"]

        if self.verbose:
            print(f"{Fore.BLUE}Parsed JSON:{Style.RESET_ALL} {call}")
            print(f"{Fore.CYAN}Extracted Tool Name:{Style.RESET_ALL} {tool_name}")
            print(f"{Fore.CYAN}Extracted Parameter:{Style.RESET_ALL} {query}")

        tool = next((tool for tool in self.tools if tool.func.__name__ == tool_name), None)
        if tool is None:
            if tool_name.lower() == "llm_tool":
                return tool_name, f"[REPLY QUERY]"
            else:
                raise ValueError(f"Tool '{tool_name}' not found.")

        # Check if the tool requires parameters
        if tool.params and query:
            if isinstance(query, dict):
                tool_response = tool(**query)
            else:
                tool_response = tool(query)
        else:
            tool_response = tool()

        return tool_name, tool_response

    def run(self) -> str:
        """Main execution logic of the agent."""
        if not self.tools:
            return self._run_no_tool()
        else:
            return self._run_with_tools()

