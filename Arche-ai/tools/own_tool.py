from typing import Callable, Dict, Any, Optional

class OwnTool:
    """Represents a tool that the agent can use."""

    def __init__(
        self, func: Callable, description: str, params: Optional[Dict[str, Any]] = None, returns_value: bool = True
    ):
        """
        Initializes an OwnTool instance.

        Args:
            func (Callable): The function to execute when the tool is called.
            description (str): A description of the tool.
            params (Optional[Dict[str, Any]], optional): A dictionary of parameters 
                                                        for the tool. Defaults to None.
            returns_value (bool, optional): If True, the tool returns a value. 
                                           If False, it's considered an action tool
                                           that doesn't return a value. Defaults to True.
        """
        self.func = func
        self.description = description
        self.params = params
        self.returns_value = returns_value  # Add this line 
