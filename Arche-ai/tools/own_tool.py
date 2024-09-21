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
        """Extracts parameter information from the function signature.

        This method inspects the function signature to determine the parameters,
        their types, and any available descriptions. It attempts to infer
        type information from type hints, default values, and docstrings.

        Returns:
            A dictionary where keys are parameter names and values are
            dictionaries containing 'type' and 'description' of each parameter.
        """
        signature = inspect.signature(self.func)
        params = {}

        for name, param in signature.parameters.items():
            # Get the annotation type, handling Optional types and others
            if param.annotation != inspect._empty:
                if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is Optional:
                    param_type = param.annotation.__args__[0].__name__  # Extract the base type from Optional
                else:
                    param_type = param.annotation.__name__
            else:
                param_type = None

            # Attempt to infer type from default value (if any) if type hint is not available
            if param_type is None and param.default != inspect._empty:
                param_type = type(param.default).__name__

            # Fallback to "string" if type cannot be determined
            param_type = param_type or "string"

            # Extract description from docstring if available
            if param.annotation != inspect._empty and hasattr(param.annotation, '__doc__'):
                description = param.annotation.__doc__.strip().split("\n")[0] if param.annotation.__doc__ else f"No description provided for '{name}'."
            else:
                # Provide a default description if no docstring is found
                description = f"No description provided for '{name}'."

            # Include the parameter's default value in the description if available
            if param.default != inspect._empty:
                description += f" (default: {param.default})"

            params[name] = {
                "type": param_type,
                "description": description,
            }

        return params

if __name__=="__main__":
    from web_search import web_search
    boom = Tool(func=web_search, description="", returns_value=True)
    print(boom.params)