import inspect
from typing import Callable, Any, Dict
from llms import Cohere  # Or your chosen LLM class
import re 
# ... other imports

class OwnTool:
    def __init__(
        self,
        func: Callable,
        description: str,
        returns_value: bool = True,
    ):
        self.func = func
        self.description = description
        self.returns_value = returns_value
        self.params = self._extract_params() # Get params from function definition

    def _extract_params(self) -> Dict[str, Dict[str, Any]]:
        """Extracts parameter information, inferring types if necessary."""
        signature = inspect.signature(self.func)
        params = {}
        for name, param in signature.parameters.items():
            # Use type hints if available
            param_type = param.annotation.__name__ if param.annotation != inspect._empty else None
            
            # Attempt to infer type from default value (if any)
            if param_type is None and param.default != inspect._empty:
                param_type = type(param.default).__name__

            # Fallback to "string" if type cannot be determined
            param_type = param_type or "string" 

            params[name] = {
                "type": param_type,
                "description": param.annotation.__doc__.strip().split("\n")[0] if param.annotation != inspect._empty and hasattr(param.annotation, '__doc__') and param.annotation.__doc__ else f"No description provided for '{name}'.",
            }
        return params

# Example Tool Functions (with type hints and docstrings)
def gcd(a:int, b:int) -> int:
    """
    Calculate the Greatest Common Divisor (GCD) of two numbers using the Euclidean algorithm.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The GCD of the two numbers.
    """
    while b:
        a, b = b, a % b
    return a + b

def get_weather(location: str, date: str = "today") -> str:
    """
    Fetches weather information for a location on a given date.

    Args:
        location: The city or region.
        date: The date (default is "today").

    Returns:
        A string describing the weather conditions.
    """
    # (Replace with actual weather API call)
    return f"Weather in {location} on {date}: 25°C, Sunny"

# ... (Other tools functions with type hints and docstrings) 

# Initialize tools
# gcd_tool = OwnTool(func=gcd, description="Calculates the GCD of two numbers.")
# weather_tool = OwnTool(func=get_weather, description="Fetches weather information.")

def _extract_params(func) -> Dict[str, Dict[str, Any]]:
        """Extracts parameter information, inferring types if necessary."""
        signature = inspect.signature(func)
        params = {}
        for name, param in signature.parameters.items():
            # Use type hints if available
            param_type = param.annotation.__name__ if param.annotation != inspect._empty else None
            
            # Attempt to infer type from default value (if any)
            if param_type is None and param.default != inspect._empty:
                param_type = type(param.default).__name__

            # Fallback to "string" if type cannot be determined
            param_type = param_type or "string" 

            params[name] = {
                "type": param_type,
                "description": param.annotation.__doc__.strip().split("\n")[0] if param.annotation != inspect._empty and hasattr(param.annotation, '__doc__') and param.annotation.__doc__ else f"No description provided for '{name}'.",
            }
        return params

print(_extract_params(gcd))