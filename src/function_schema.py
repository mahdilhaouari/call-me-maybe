# src/function_schema.py
import json
from typing import Any, Dict, List, Union

from pydantic import create_model, ValidationError, StrictFloat


def _type_str(param_def: Union[str, Dict]) -> str:
    if isinstance(param_def, str):
        return param_def
    return param_def["type"]


class FunctionSchema:
    """Loads function definitions and validates function calls."""

    def __init__(self, schema_path: str) -> None:
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._names:       List[str]            = []
        self._param_types: Dict[str, Dict]      = {}
        self._models:      Dict[str, Any]       = {}

        type_map = {"number": StrictFloat, "string": str, "boolean": bool}

        for fn in data:
            name   = fn["name"]
            params = fn.get("parameters", {})
            self._names.append(name)
            self._param_types[name] = {p: _type_str(d) for p, d in params.items()}
            fields = {p: (type_map.get(_type_str(d), Any), ...) for p, d in params.items()}
            Params = create_model(f"{name}_P", **fields)
            self._models[name] = create_model(name, name=(str, ...), parameters=(Params, ...))

    def get_allowed_function_names(self) -> List[str]:
        return list(self._names)

    def get_param_type_info(self, name: str) -> Dict[str, str]:
        return self._param_types.get(name, {})

    def validate_function_call(self, call: Dict[str, Any]) -> bool:
        model = self._models.get(call.get("name", ""))
        if not model:
            return False
        try:
            model(**call)
            return True
        except (ValidationError, Exception):
            return False