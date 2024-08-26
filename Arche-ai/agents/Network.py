import json
from typing import List, Dict, Optional, Tuple, Any
from llms import Gemini  # Or your preferred LLM class
from agents import Agent
from colorama import Fore, Style, init

init(autoreset=True)

class TaskForce:
    def __init__(self, agents: List[Agent], llm: Gemini, name:str, description:str, verbose: bool = False):
        self.agents = agents
        self.llm = llm
        self.name = name
        self.description = description
        self.verbose = verbose
        self.task_history: List[Tuple[str, str, str]] = []
        self.shared_workspace: Dict[str, Any] = {} 

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def remove_agent(self, agent_name: str):
        self.agents = [agent for agent in self.agents if agent.name != agent_name]

    def rollout(self, initial_task: str, max_iterations: int = 5) -> str:
        """Orchestrates task execution and agent collaboration."""

        current_task = initial_task
        self.task_history = []
        self.shared_workspace = {}

        print(f"{Fore.CYAN}TaskForce activated. Initial task: {initial_task}{Style.RESET_ALL}")

        for iteration in range(max_iterations):
            print(f"\n{Fore.CYAN}Iteration: {iteration + 1}/{max_iterations}{Style.RESET_ALL}")

            selected_agent, next_task, communication_plan = self._plan_iteration(current_task)

            if not selected_agent:
                print(f"{Fore.YELLOW}Task planning complete or no suitable agent found.{Style.RESET_ALL}")
                break

            print(f"{Fore.YELLOW}Selected Agent: {selected_agent.name}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Task: {current_task}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Communication Plan: {communication_plan}{Style.RESET_ALL}\n")

            self._handle_communication(communication_plan)

            print(f"{Fore.GREEN}Executing Agent: {selected_agent.name}...{Style.RESET_ALL}")
            
            # Delegate the task to the selected agent
            selected_agent.task = current_task  # Assign the task here
            agent_response = selected_agent.rollout()

            print(f"{Fore.GREEN}Agent {selected_agent.name} Response: {agent_response}{Style.RESET_ALL}")

            self.shared_workspace[selected_agent.name] = agent_response
            self.task_history.append((selected_agent.name, current_task, agent_response))

            if next_task.upper() == "TASK COMPLETE":
                print(f"{Fore.GREEN}Task successfully completed!{Style.RESET_ALL}")
                break
            current_task = next_task

        final_response = self._generate_final_response(initial_task)

        if iteration + 1 == max_iterations:
            print(
                f"{Fore.YELLOW}Maximum iterations reached. Task might not be fully complete.{Style.RESET_ALL}"
            )

        print(f"{Fore.CYAN}\nFinal Consolidated Response:{Style.RESET_ALL}\n{final_response}")
        return final_response

    def _plan_iteration(self, current_task: str) -> Tuple[Optional[Agent], str, dict]:
        """Plans the next iteration, handling multiple tasks for the same agent."""

        agent_info = self._get_agents_info()

        llm_prompt = f"""
        You are the TaskForce coordinator. Your goal is to break down tasks, assign them to 
        specialized agents, and manage communication between them. 

        Current Task: {current_task}
        Shared Workspace: 
        {json.dumps(self.shared_workspace, indent=4)}
        Available Agents: 
        {agent_info}
        Task History:
        {self._format_task_history()}

        Instructions:
        1. Read the agents description and find the agent suitable for the task, if no agent is suited for the task try generating it with the next most suitable agent.  
        2. If the current task can be broken down into smaller sub-tasks, do so. 
           This is especially important if multiple sub-tasks can be handled by the same agent.
        3. Select the most suitable agent for the NEXT sub-task.
        4. Create a communication plan, if needed.
        5. Create a communication plan ONLY if additional information needs to 
           be shared between agents.  Avoid redundant communication. 
        6. Output a JSON object with this structure:
           ```json
           {{
               "selected_agent": "<Agent Name>" or "None",
               "next_task": "<Task Description>" or "TASK COMPLETE",
               "communication_plan": {{
                   "<Recipient Agent Name>": {{
                       "message": "<Information to send to this agent>",
                       "source_agent": "<Source Agent Name (if applicable)>"
                   }}
               }}
           }}
           ``` 
        """

        response = self.llm.run(llm_prompt)
        self.llm.reset()

        if self.verbose:
            print(f"LLM Planning Response: {response}")

        plan = self._extract_json_plan(response)
        if not plan:
            if self.verbose:
                print("Error: Invalid plan format from LLM.")
            return None, "TASK COMPLETE", {}

        selected_agent_name = plan.get("selected_agent")
        next_task = plan.get("next_task", "TASK COMPLETE")
        communication_plan = plan.get("communication_plan", {})

        selected_agent = next(
            (agent for agent in self.agents if agent.name == selected_agent_name), None
        )
        if not selected_agent and selected_agent_name != "None":
            if self.verbose:
                print(f"Error: Agent '{selected_agent_name}' not found.")

        # Assign the task to the selected agent
        if selected_agent:
            selected_agent.task = next_task

        return selected_agent, next_task, communication_plan

    def _format_task_history(self) -> str:
        """Formats the task history for the LLM prompt."""
        if not self.task_history:
            return "Task History: None"
        history_str = ["Task History:"]
        for i, (agent_name, task, response) in enumerate(self.task_history):
            history_str.append(
                f"- Turn {i + 1}: Agent '{agent_name}' was given the task '{task}' and responded with '{response}'"
            )
        return "\n".join(history_str)        

    def _handle_communication(self, communication_plan: Dict[str, str]):
        """Manages communication between agents based on the plan."""
        for agent_name, communication_details in communication_plan.items():
            recipient = next(
                (agent for agent in self.agents if agent.name == agent_name), None
            )
            if recipient:
                message = communication_details.get("message")
                source_agent = communication_details.get("source_agent")

                # Format the communication message 
                communication_string = f"New Information From TaskForce:\n{message}" 
                if source_agent:
                    communication_string += f"\n(Source: {source_agent})" 

                # Update recipient's task based on communication
                recipient.task = f"Previous Task:\n{recipient.task}\n\n{communication_string}" 
            else:
                if self.verbose:
                    print(f"Warning: Agent '{agent_name}' not found for communication.")

    def _extract_json_plan(self, llm_response: str) -> Optional[Dict]:
        """Extracts and validates the JSON plan from the LLM's response."""
        try:
            # Extract JSON from within backticks 
            llm_response = llm_response.strip()
            if llm_response.startswith("```json") and llm_response.endswith("```"):
                llm_response = llm_response[7:-3].strip()  
            else:
                # If not in backticks, find JSON object directly
                start = llm_response.index('{')
                end = llm_response.rindex('}') + 1
                llm_response = llm_response[start:end]
            
            plan = json.loads(llm_response)
            assert "selected_agent" in plan and "next_task" in plan
            return plan
        except (json.JSONDecodeError, AssertionError, ValueError) as e: # Handle potential ValueError 
            if self.verbose:
                print(f"Error parsing JSON: {e}")  # Add error logging
            return None

    def _get_agents_info(self) -> str:
        """Provides a formatted string of agent information for the LLM."""
        agent_info = []
        for agent in self.agents:
            tool_names = [tool.func.__name__ for tool in agent.tools] if agent.tools else []
            tool_info = f"Tools: {', '.join(tool_names)}" if tool_names else "Tools: None"
            agent_info.append(f"Name: {agent.name}\nDescription: {agent.description}\n Skills: {agent.skills}\n{tool_info}")
        return "\n\n".join(agent_info)

    def _generate_final_response(self, initial_task: str) -> str:
        """Uses the LLM to combine agent responses into a coherent final answer."""

        # Prepare the final response prompt
        final_response_prompt = f"""
        You are {self.name}, {self.description}.
        
        Responsible for integrating the work of 
        individual agents to provide a complete and informative answer.

        Initial Task: {initial_task}
        Shared Workspace: {self.shared_workspace}
        Task History: 
        {self._format_task_history()}

        Instructions:
        - Combine and synthesize the information from the shared workspace and task history.
        - Provide a concise, well-organized final answer to the initial task. 
        - Do not simply list the agent responses; aim for a unified and coherent response. 
        """

        # Generate the final response
        final_response = self.llm.run(final_response_prompt)
        self.llm.reset()  # Reset the LLM's state after generating the final response
        return final_response