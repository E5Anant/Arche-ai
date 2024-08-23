from typing import Callable, Dict, Any, Optional
import inspect

class Tool:
    def __init__(
        self,
        func: Callable,
        description: str,
        returns_value: bool,
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