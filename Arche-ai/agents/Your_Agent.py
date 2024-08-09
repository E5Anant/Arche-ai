from llms import GroqLLM, Gemini, Cohere
from tools import OwnTool
from typing import Type, List, Optional, Dict, Any
import json
from colorama import Fore, Style
import concurrent.futures
import re

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
        self.task = task
        self.verbose = verbose

        self.all_functions = [
            convert_function(tool.func.__name__, tool.description, **tool.params) for tool in self.tools
        ] + [convert_function("llm_tool", "A default tool that provides AI-generated text responses and it cannot answer real-time queries because of the knowledge cut off of October 2019.")]

    def _initialize_llm(self, system_prompt: str) -> None:
        self.llm.__init__(system_prompt=system_prompt)

    def add_tool(self, tool: OwnTool):
        """Add a tool to the agent dynamically."""
        self.tools.append(tool)
        self.all_functions.append(convert_function(tool.func.__name__, tool.description, **tool.params))

    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent dynamically."""
        self.tools = [tool for tool in self.tools if tool.func.__name__ != tool_name]
        self.all_functions = [func for func in self.all_functions if func['function']['name'] != tool_name]

    def _run_no_tool(self) -> str:
        self._initialize_llm(f"""
You are {self.name}, {self.description}.
### OUTPUT STYLE:
{self.sample_output}
***If output style not mentioned, generate in markdown format.***
""")
        result = self.llm.run(self.task)
        self.llm.reset()
        return result

    def _run_with_tools(self) -> str:
        """Handles tasks that require using tools."""
        self.tools_info = "\n".join([
            f"Tool Name: {tool.func.__name__} - {tool.description}\nTool Parameters: {tool.params}"
            for tool in self.tools
        ])

        self._initialize_llm(f"""
You are an AI assistant designed to generate JSON responses based on provided tools.

Before responding, ask yourself:
1. Does the task require real-time or specific data retrieval (e.g., weather, time)?
2. If yes, identify the appropriate tool and its required parameters.
3. If no, answer the task directly without tool usage.

Available Tools:
{self.all_functions}

Instructions:
1. Read the task carefully.
2. Identify the required tool parameters.
3. Respond with a JSON object containing the tool_name and parameter.
4. Only provide the JSON response. Do not include any text outside of the JSON structure.
5. You are only trained to give JSON response and not text or conversation.

JSON Structure with tool params:
{{
    "func_calling": [
        {{
            "tool_name": "<tool_name>",
            "parameter": {{<param_name> : "<param_value>"}}
        }}
    ]
}}

JSON Structure for calling tools without tool params:

{{
    "func_calling": [
        {{
            "tool_name": "<tool_name>",
            "parameter": {{""}}
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
            "parameter": {{"query" : "New York"}}
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

***How to handle double parameter tools***

Example:
Task: Get the sum of 48 and 12
Response:
{{
    "func_calling": [
        {{
            "tool_name": "add",
            "parameter": {{"a": 48, "b": 12}}
        }}
    ]
}}

## always use exact same structure while calling double param tools, `make sure this is just an example showcasing the structure to call while the tools and params used here varies according to provided tools and params above.`

Example with llm_tool:
Task: Who are you?
Response:
{{
    "func_calling": [
        {{
            "tool_name": "llm_tool",
            "parameter": "Who are you?"
        }}
    ]
}}

***Remember these are just examples, the tools and parameters vary according to the details given above.***
""")

        response = self.llm.run(self.task).strip()

        if self.verbose:
            print(f"{Fore.YELLOW}Raw LLM Response:{Style.RESET_ALL} {response}")

        # Preprocess the response to fix common JSON errors
        response = self._preprocess_response(response)

        # Extract only the JSON part from the response
        json_part = self._extract_json(response)

        results = {}
        if json_part:
            try:
                # Ensure only JSON part is processed
                action = json.loads(json_part.strip())

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
                results = {"error": f"Failed to decode JSON: {str(e)}"}
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.RED}Error:{Style.RESET_ALL} {str(e)}")
                results = {"error": f"An error occurred: {str(e)}"}

            try:
                if self.verbose:
                    print()
                    print(f"{Fore.GREEN}Tool_RESULTS:\n{results}{Style.RESET_ALL}")
                    print()
            except:
                pass

            self.llm.reset()

            return self._generate_summary(results)

        else:
            # If no tools were needed, the response should be a direct answer.
            self.llm.reset()
            return response

    def _preprocess_response(self, response: str) -> str:
        """Preprocess the response to fix common JSON errors."""
        # Fix missing colons in parameter sections
        response = re.sub(r'("parameter":\s*)({)([^}]*)(})', r'\1{\2:\3\4}', response)
        return response

    def _extract_json(self, response: str) -> Optional[str]:
        """Enhanced JSON extraction with better pattern matching."""
        try:
            start = response.index('{')
            end = response.rindex('}') + 1
            return response[start:end]
        except ValueError:
            if self.verbose:
                print("No valid JSON structure found.")
            return None

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

        # Flatten the parameters if they are nested within any dictionary
        if isinstance(query, dict):
            # Extract the first nested dictionary if it exists
            nested_keys = [key for key in query if isinstance(query[key], dict)]
            if nested_keys:
                query = query[nested_keys[0]]

        # Check if the tool requires parameters
        if tool.params and isinstance(query, dict):
            try:
                # Pass the parameters as keyword arguments
                tool_response = tool.func(**query)
            except TypeError as e:
                if 'unexpected keyword argument' in str(e) or 'missing 1 required positional argument' in str(e):
                    # Handle case where parameters need to be passed positionally
                    tool_response = tool.func(*query.values())
                else:
                    raise e
        else:
            tool_response = tool.func()

        return tool_name, tool_response

    def _generate_summary(self, results: Dict[str, str]) -> str:
         summarizer_llm = self.llm.__class__()
        
        summarizer_llm.llm.__init__(system_prompt= f"""
You are {self.name}, an AI agent. You are provided with output from the tools in JSON format. Your task is to use this information to give the best possible answer to the query. Reply in a natural language style, only in text, and to the point. Do not reply in JSON.

### TOOLS:
llm_tool - If this tool is used, you must answer the user's query in the best possible way.
{self.all_functions}

### OUTPUT STYLE:
{self.sample_output}

## Instructions:
- If the output style is not mentioned, just reply in the best possible way in only text form and not JSON.
- You are no longer generating JSON responses. Provide a natural language summary based on the information from the tools.
""")
        try:
            summary = summarizer_llm.llm.run(f"[QUERY]\n{self.task}\n\n[TOOLS]\n{results}")
            if self.verbose:
                print("Final Response:")
                print(summary)
            return summary
        except Exception as e:
            if self.verbose:
                print(f"{Fore.RED}Error generating summary:{Style.RESET_ALL} {str(e)}")
            return "There was an error generating the summary."

    def _finalize_response(self, response: str) -> None:
        self.llm.reset()
        self.llm.add_message("user", self.task)
        if isinstance(self.llm, Gemini):
            self.llm.add_message("model", response)
        elif isinstance(self.llm, Cohere):
            self.llm.add_message("Chatbot", response)
        else:
            self.llm.add_message("assistant", response)

    def run(self) -> str:
        self.llm.reset()
        """Main execution logic of the agent."""
        if not self.tools:
            return self._run_no_tool()
        return self._run_with_tools()

# Example usage
if __name__ == "__main__":
    llm = GroqLLM()
    tools = [
        OwnTool(
            func=lambda location, date: f"Weather in {location} on {date}: 27°C, Sunny, Wind: 12 km/h W, Humidity: 46% Forecast for tomorrow: 18°C - 30°C, Sunny",
            description="Get weather information for a location on a specific date",
            params={
                "location": {"type": "string", "description": "The location to get weather for"},
                "date": {"type": "string", "description": "The date to get weather for"}
            }
        ),
        OwnTool(
            func=lambda: "23:18:03",
            description="Get the current time",
            params=None
        )
    ]

    agent = Agent(
        llm=llm,
        tools=tools,
        name="Example Agent",
        description="An example agent",
        verbose=True,
        task="What is the weather and time in Kolkata on 2023-10-01?"
    )

    response = agent.run()
    print(response)
