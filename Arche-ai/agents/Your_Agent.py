import json
import re
from llms import GroqLLM, Gemini, Cohere  # Make sure to import your LLM classes
from tools import Tool
from typing import Type, List, Optional, Dict, Any
from colorama import Fore, Style
import concurrent.futures

def convert_function(func_name, description, **params):
    """Converts function info to a JSON function schema.
    Handles cases where params might be None.
    """

    function_dict = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    if params is None:
        params = {}

    for param_name, param_info in params.items():
        try:
            if "description" not in param_info:
                param_info[
                    "description"
                ] = f"Description for {param_name} is missing. Defaulting to {param_info}"
                descri = f"{param_info}"
            else:
                descri = param_info["description"]
        except:
            descri = f"{str(param_info)}"

        # --- Type Handling Correction ---
        param_type = param_info.get("type", "string")
        valid_types = {
            "string": "string",
            "str": "string",  # Add common abbreviations
            "number": "number",
            "num": "number",
            "int": "integer",  # Map "int" to "integer"
            "integer": "integer",
            "boolean": "boolean",
            "bool": "boolean",  # Add common abbreviations
            "enum": "enum",
            "array": "array",
            "list": "array",
            "dict": "dictionary",
            "dictionary": "dictionary",
            "object": "object",
            "obj": "object",
        }
        param_type = valid_types.get(param_type.lower(), "string")  # Normalize and validate

        param_properties = {"type": param_type, "description": descri}

        if param_type == "enum":
            if "options" not in param_info:
                raise ValueError(
                    f"Parameter '{param_name}' of type 'enum' requires an 'options' list."
                )
            param_properties["enum"] = param_info["options"]

        try:
            if "default" in param_info:
                param_properties["default"] = param_info["default"]
        except:
            pass

        try:
            if param_info.get("required", False):
                function_dict["function"]["parameters"]["required"].append(
                    param_name
                )
        except:
            function_dict["function"]["parameters"]["required"].append(param_name)

        function_dict["function"]["parameters"]["properties"][
            param_name
        ] = param_properties

    return function_dict

class Agent:
    def __init__(
        self,
        llm: Type[GroqLLM],
        tools: List[Tool] = [],
        name: str = "Agent",
        description: str = "A helpful AI agent.",
        expected_output: str = "Concise and informative text.",
        task: str = "Ask me a question or give me a task.",
        skills:str = "Productive and helpful",
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.name = name
        self.description = description
        self.expected_output = expected_output
        self.task = task
        self.verbose = verbose
        self.skills = skills

        self.all_functions = [
            convert_function(tool.func.__name__, tool.description, **(tool.params or {})) for tool in self.tools # Add or {} to handle None
        ] + [convert_function("llm_tool", "A default tool that provides AI-generated text responses and it cannot answer real-time queries because of the knowledge cut off of October 2019.", **{})]

    def add_tool(self, tool: Tool):
        """Add a tool to the agent dynamically."""
        self.tools.append(tool)
        self.all_functions.append(convert_function(tool.func.__name__, tool.description, **tool.params))

    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent dynamically."""
        self.tools = [tool for tool in self.tools if tool.func.__name__ != tool_name]
        self.all_functions = [func for func in self.all_functions if func['function']['name'] != tool_name]

    def _run_no_tool(self) -> str:
        self.llm.__init__(system_prompt = f"""
You are {self.name}, {self.description}.
### OUTPUT STYLE:
{self.expected_output}
***If output style not mentioned, generate in markdown format.***
""", messages = [])
        result = self.llm.run(self.task)
        self.llm.reset()
        print("Final Response:")
        print()
        print(result)
        return result

    def _fix_json(self, raw_json: str) -> str:
        """Attempts to fix common JSON errors."""
        # Remove extraneous characters
        corrected_json = raw_json.replace("```json", "").replace("```", "").strip()

        # Replace single quotes with double quotes
        corrected_json = corrected_json.replace("'", "\"")

        # --- Additional Cleaning ---
        # 1. Remove problematic control characters (optional, but might help)
        corrected_json = "".join(c for c in corrected_json if c.isprintable())
        
        # 2. Normalize line endings to Unix-style (\n)
        corrected_json = corrected_json.replace('\r\n', '\n')

        return corrected_json

    def _run_with_tools(self) -> str:
        """Handles tasks that require using tools."""
        self.tools_info = "\n".join(
            [
                f"Tool Name: {tool.func.__name__} - {tool.description}\nTool Parameters: {tool.params}"
                for tool in self.tools
            ]
        )

        self.llm.__init__(system_prompt=
            f"""
You are an AI assistant designed to generate JSON responses based on provided tools.

Before responding, ask yourself:
1. Does the task require real-time or specific data retrieval (e.g., weather, time)?
2. If yes, identify the appropriate tool and its required parameters and call it.
3. If no, answer the task directly using `llm_tool` (make sure it should be your last option.).

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
""", messages=[]
        )

        response = self.llm.run(self.task).strip()
        self.llm.reset()

        if self.verbose:
            print(f"{Fore.YELLOW}Raw LLM Response:{Style.RESET_ALL} {response}")

        # --- Improved JSON Extraction and Handling ---
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)  # Use re.DOTALL

        if json_match:
            extracted_json = json_match.group(1)  # Get the captured group

            try:
                # Try parsing the extracted JSON
                action = json.loads(extracted_json)
            except json.JSONDecodeError as e:
                print(f"{Fore.RED}JSON Decode Error:{Style.RESET_ALL} {str(e)}")
                print("Attempting to fix JSON...")
                try:
                    corrected_response = self._fix_json(extracted_json)
                    action = json.loads(corrected_response)
                    print("JSON successfully fixed!")
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"{Fore.RED}Error: Unable to fix JSON: {str(e)}")
                    # Fallback: If JSON can't be extracted/fixed, treat the entire response as non-JSON
                    self.llm.reset()
                    return response
        else:
            # If no JSON-like structure is found, treat as a regular response
            self.llm.reset()
            return response

        # --- Tool Calling Logic (After JSON Handling) ---
        func_calling = action.get("func_calling", [])
        results = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_tool = {
                executor.submit(self._call_tool, call): call
                for call in func_calling
            }
            for future in concurrent.futures.as_completed(future_to_tool):
                call = future_to_tool[future]
                try:
                    tool_name, tool_response = future.result()
                    results[tool_name] = tool_response
                    if self.verbose:
                        print(
                            f"{Fore.GREEN}Tool Response ({tool_name}):{Style.RESET_ALL} {tool_response}"
                        )
                except Exception as e:
                    if self.verbose:
                        print(
                            f"{Fore.RED}Error calling tool {call['tool_name']}:{Style.RESET_ALL} {str(e)}"
                        )
                    results[call["tool_name"]] = (
                        f"Failed to get info: {str(e)}."
                    )

        try:
            if self.verbose:
                print()
                print(f"{Fore.GREEN}Tool_RESULTS:\n{results}{Style.RESET_ALL}")
                print()
        except:
            pass

        self.llm.reset()

        return self._generate_summary(results)

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

        # Handle action tools (those that don't return a value)
        if tool.returns_value:
            # If the tool returns a value, execute it as before
            if tool.params and isinstance(query, dict):
                try:
                    tool_response = tool.func(**query)
                except TypeError as e:
                    if (
                        "unexpected keyword argument" in str(e)
                        or "missing 1 required positional argument" in str(e)
                    ):
                        tool_response = tool.func(*query.values())
                    else:
                        raise e
            else:
                tool_response = tool.func()

            return tool_name, tool_response 
        else:
            # If it's an action tool:
            if tool.params and isinstance(query, dict):
                try:
                    # --- Convert parameters to integers here ---
                    if tool_name == "gcd":
                        query["l"] = int(query["l"])
                        query["h"] = int(query["h"])

                    tool.func(**query)
                except TypeError as e:
                    if (
                        "unexpected keyword argument" in str(e)
                        or "missing 1 required positional argument" in str(e)
                    ):
                        tool.func(*query.values())
                    else:
                        raise e
            else:
                tool.func()

            return tool_name, "Action performed successfully."  # Or a suitable message

    def _generate_summary(self, results: Dict[str, str]) -> str:
        self.llm.reset()
        self.llm.__init__(system_prompt=f"""
You are {self.name}, an AI agent. {self.description}.\n You are provided with output from the tools in JSON format. Your task is to use this information to give the best possible answer to the query.

### TOOLS:
llm_tool - If this tool is used, you must answer the user's query in the best possible way.
{self.all_functions}

### OUTPUT STYLE:
{self.expected_output}

## Instructions:
- If the output style is not mentioned, just reply in the best possible way in only text form and not JSON.
- You are no longer generating JSON responses. Provide a natural language summary based on the information from the tools.
""", messages=[])
        try:
            summary = self.llm.run(f"[QUERY]\n{self.task}\n\n[TOOLS]\n{results}")
            if self.verbose:
                print("Final Response:")
                print(summary)
            return summary
        except Exception as e:
            if self.verbose:
                print(f"{Fore.RED}Error generating summary:{Style.RESET_ALL} {str(e)}")
            return "There was an error generating the summary."

    def rollout(self) -> str: 
        self.llm.reset()
        if not self.tools:
            if not self.task: # Check if self.task is empty
                return "No task provided." 
            return self._run_no_tool()
        return self._run_with_tools()
