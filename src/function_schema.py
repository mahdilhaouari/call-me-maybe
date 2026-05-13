# src/function_schema.py
import json
from typing import Any, Dict, List, Union

from pydantic import create_model, ValidationError, StrictFloat


def _get_type_str(param_def: Union[str, Dict]) -> str:
    """
    The schema can define a parameter type in two ways:
        "a": "number"          (just a string)
        "a": {"type": "number"} (a dict)
    This helper normalises both to just the string.
    """
    if isinstance(param_def, str):
        return param_def
    return param_def["type"]


class FunctionSchema:
    """
    Loads a functions_definition.json file and provides:
    - the list of valid function names
    - the parameter types for each function
    - a Pydantic validator to check a generated function call
    """

    def __init__(self, schema_path: str) -> None:
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._names: List[str] = []
        self._param_types: Dict[str, Dict[str, str]] = {}
        self._models: Dict[str, Any] = {}

        # Map JSON type names to Python types for Pydantic
        type_map = {
            "number": StrictFloat,
            "string": str,
            "boolean": bool,
        }

        for fn in data:
            name = fn["name"]
            params = fn.get("parameters", {})

            self._names.append(name)
            self._param_types[name] = {
                p: _get_type_str(d) for p, d in params.items()
            }

            # Build a Pydantic model dynamically for this function
            fields = {
                p: (type_map.get(_get_type_str(d), Any), ...)
                for p, d in params.items()
            }
            Params = create_model(f"{name}_Params", **fields)
            self._models[name] = create_model(
                name, name=(str, ...), parameters=(Params, ...)
            )

    def get_allowed_function_names(self) -> List[str]:
        return list(self._names)

    def get_param_type_info(self, name: str) -> Dict[str, str]:
        return self._param_types.get(name, {})

    def validate_function_call(self, call: Dict[str, Any]) -> bool:
        """Return True if the call matches the schema, False otherwise."""
        model = self._models.get(call.get("name", ""))
        if not model:
            return False
        try:
            model(**call)
            return True
        except (ValidationError, Exception):
            return False