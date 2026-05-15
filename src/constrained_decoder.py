# src/constrained_decoder.py
import json
import torch
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, model_validator
from llm_sdk.llm_sdk import Small_LLM_Model
from src.json_state_tracker import JSONStateTracker
from src.function_schema import FunctionSchema
from src.token_trie import TokenTrie


# The Qwen tokenizer prefixes space-starting tokens with this character (Ġ).
# We replace it with a real space so our logic works with normal characters.
_SPACE_PREFIX = "\u0120"


def _to_visible(text: str) -> str:
    """Replace the Ġ space-prefix with a real space character."""
    return text.replace(_SPACE_PREFIX, " ")


# ---------------------------------------------------------------------------
# Helpers for extracting values from the prompt text
# ---------------------------------------------------------------------------

class NumberList(BaseModel):
    """Finds all numbers in a text string."""

    values: List[float] = []

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: Any) -> Any:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        values = []
        i = 0
        while i < len(text):
            starts_number = text[i].isdigit() or (
                text[i] == "-"
                and i + 1 < len(text)
                and text[i + 1].isdigit()
            )
            if starts_number:
                j = i + 1
                has_dot = False
                while j < len(text) and (
                    text[j].isdigit()
                    or (text[j] == "." and not has_dot)
                ):
                    if text[j] == ".":
                        has_dot = True
                    j += 1
                values.append(float(text[i:j]))
                i = j
            else:
                i += 1
        return {"values": values}


class QuotedStringList(BaseModel):
    """Finds all single- or double-quoted strings in a text string."""

    values: List[str] = []

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: Any) -> Any:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        values = []
        i = 0
        while i < len(text):
            if text[i] in ('"', "'"):
                quote = text[i]
                j = i + 1
                buf = []
                while j < len(text) and text[j] != quote:
                    buf.append(text[j])
                    j += 1
                values.append("".join(buf))
                i = j + 1
            else:
                i += 1
        return {"values": values}


class LastWord(BaseModel):
    """Returns the last word — fallback for unquoted string params."""

    value: str = ""

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: Any) -> Any:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        words = text.strip().split()
        return {"value": words[-1] if words else ""}


def _extract_values(
    prompt: str, param_types: Dict[str, str]
) -> Dict[str, Any]:
    """
    Try to pull parameter values directly from the prompt text so we
    don't rely on the small LLM to copy them correctly.

    - number params : pulled from numbers found in the prompt
    - string params : pulled from quoted strings; falls back to last word
    """
    numbers = NumberList(text=prompt).values        # type: ignore[call-arg]
    strings = QuotedStringList(text=prompt).values  # type: ignore[call-arg]

    result: Dict[str, Any] = {}
    num_idx = str_idx = 0

    for param_name, param_type in param_types.items():
        if param_type == "number" and num_idx < len(numbers):
            result[param_name] = numbers[num_idx]
            num_idx += 1
        elif param_type == "string":
            if str_idx < len(strings):
                result[param_name] = strings[str_idx]
                str_idx += 1
            else:
                # no quoted string left — use the last word as a fallback
                # e.g. "greet mahdi" → name = "mahdi"
                fallback = LastWord(text=prompt).value  # type: ignore[call-arg]
                result[param_name] = fallback

    return result


# ---------------------------------------------------------------------------
# Main decoder class
# ---------------------------------------------------------------------------

class ConstrainedDecoder:
    """
    Turns a natural-language prompt into a structured function call by
    running the LLM one token at a time and only allowing tokens that
    keep the output as valid JSON matching the function schema.
    """

    def __init__(
        self,
        schema_path: str,
        model: Optional[Small_LLM_Model] = None,
    ) -> None:
        self.model = model or Small_LLM_Model()
        self.schema = FunctionSchema(schema_path)
        self.names = self.schema.get_allowed_function_names()

        # Load the vocabulary so we know what text each token ID produces
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # Build a mapping from token ID → token text
        self.id_to_text: Dict[int, str] = {}
        first_value = next(iter(raw_vocab.values()))
        if isinstance(first_value, str):
            # format: {"0": "hello", "1": "world", ...}
            for k, v in raw_vocab.items():
                try:
                    self.id_to_text[int(k)] = v
                except ValueError:
                    pass
        else:
            # format: {"hello": 0, "world": 1, ...}
            for k, v in raw_vocab.items():
                self.id_to_text[int(v)] = k

        # Build a mapping: first visible character → set of token IDs.
        # We use _to_visible so that space-prefixed tokens (Ġhello) are
        # stored under ' ' rather than the Ġ character.
        self.tokens_starting_with: Dict[str, Set[int]] = {}
        for token_id, token_text in self.id_to_text.items():
            if not token_text:
                continue
            visible = _to_visible(token_text)
            if visible:
                self.tokens_starting_with.setdefault(
                    visible[0], set()
                ).add(token_id)

        # Build a trie from the encoded function names so we can constrain
        # the model to only generate valid function name tokens
        self.name_trie = TokenTrie(
            [self._encode(n) for n in self.names]
        )

        # Cache the token IDs for characters we need to check explicitly
        self.quote_id = self._single_token('"')
        self.rbrace_id = self._single_token("}")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        """Encode a string to a list of token IDs."""
        return self.model.encode(text)[0].tolist()

    def _single_token(self, text: str) -> Optional[int]:
        """Return the token ID only if text encodes to exactly one token."""
        ids = self._encode(text)
        return ids[0] if len(ids) == 1 else None

    def _get_allowed_tokens(self, valid_chars: Set[str]) -> Set[int]:
        """Convert a set of valid characters into valid token IDs."""
        result: Set[int] = set()
        for ch in valid_chars:
            result.update(self.tokens_starting_with.get(ch, set()))
        return result

    def _inject(
        self,
        text: str,
        token_ids: List[int],
        output_tokens: List[int],
        tracker: JSONStateTracker,
    ) -> None:
        """
        Write a known value directly without asking the model.
        Feed each character to the tracker, then append the token IDs.
        Using the original text (not the vocab representation) preserves
        spaces correctly.
        """
        for ch in text:
            try:
                tracker.update(ch)
            except ValueError:
                pass
        for tok in self._encode(text):
            output_tokens.append(tok)
            token_ids.append(tok)

    def _build_prompt(self, user_request: str) -> str:
        """Build the full instruction prompt we send to the model."""
        function_list = "\n".join(
            "  - {}({})".format(
                name,
                ", ".join(
                    f"{p}: {t}"
                    for p, t in self.schema.get_param_type_info(name).items()
                ),
            )
            for name in self.names
        )
        return (
            "You are a function-calling assistant.\n"
            "Available functions:\n"
            f"{function_list}\n"
            'Output ONLY a JSON object with keys "name" and "parameters".\n'
            f"User: {user_request}\n"
            "JSON:"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_steps: int = 600) -> str:
        """
        Run the constrained generation loop and return the raw output string.
        At each step we ask the model for the next token but only allow
        tokens that are consistent with valid JSON and the function schema.
        """
        # Per-call state
        name_tokens: List[int] = []
        function_name: Optional[str] = None
        keys_written: Set[str] = set()
        pre_extracted: Dict[str, Any] = {}
        pending_key: str = ""

        tracker = JSONStateTracker()
        token_ids = self._encode(self._build_prompt(prompt))
        output_tokens: List[int] = []

        for _ in range(max_steps):
            # Ask the model what comes next
            logits = torch.tensor(
                self.model.get_logits_from_input_ids(token_ids),
                dtype=torch.float32,
            )

            state = tracker.get_current_state()
            key = tracker.current_key

            # As soon as a key is fully written, save it
            if state == "AFTER_KEY" and key:
                pending_key = key

            # If we have a pre-extracted value ready for this parameter,
            # write it directly instead of letting the model generate it
            if (
                state == "AFTER_COLON"
                and function_name
                and pending_key
                and pending_key in pre_extracted
                and pending_key not in keys_written
            ):
                value = pre_extracted[pending_key]
                param_type = self.schema.get_param_type_info(
                    function_name
                ).get(pending_key, "string")

                if param_type == "number":
                    num_val = int(value) if float(value) == int(value) else value
                    self._inject(
                        str(num_val), token_ids, output_tokens, tracker
                    )
                    keys_written.add(pending_key)
                    pending_key = ""
                    continue

                if param_type == "string":
                    escaped = (
                        str(value).replace("\\", "\\\\").replace('"', '\\"')
                    )
                    self._inject(
                        f'"{escaped}"', token_ids, output_tokens, tracker
                    )
                    keys_written.add(pending_key)
                    pending_key = ""
                    continue

            # --- Figure out which tokens the model is allowed to pick ---

            valid_chars = tracker.get_valid_next_chars()
            if not valid_chars:
                break

            allowed = self._get_allowed_tokens(valid_chars)

            # While writing the function name, only allow tokens that
            # continue a valid function name according to the trie
            if state == "IN_STRING" and key == "name" and function_name is None:
                next_from_trie = self.name_trie.get_allowed_next_tokens(
                    name_tokens
                )
                # also allow closing quote if we have a complete name so far
                if (
                    self.name_trie.is_complete_prefix(name_tokens)
                    and self.quote_id
                ):
                    next_from_trie.add(self.quote_id)
                allowed &= next_from_trie
                if not allowed:
                    break

            # For boolean parameters, only allow chars from "true"/"false"
            elif (
                state == "IN_STRING"
                and function_name
                and key not in ("", "name")
            ):
                param_type = self.schema.get_param_type_info(
                    function_name
                ).get(key)
                if param_type == "boolean":
                    bool_tokens = self._get_allowed_tokens(set("truefals"))
                    if self.quote_id:
                        bool_tokens.add(self.quote_id)
                    allowed &= bool_tokens

            # Don't allow the object to close until both required keys exist
            required = {"name", "parameters"}
            if (
                state == "AFTER_VALUE"
                and tracker.get_depth() == 1
                and not required.issubset(keys_written)
                and self.rbrace_id
            ):
                allowed.discard(self.rbrace_id)
                allowed -= self.tokens_starting_with.get("}", set())

            # If constraints removed everything, fall back to syntax only
            if not allowed:
                allowed = self._get_allowed_tokens(valid_chars)
                if not allowed:
                    break

            # --- Mask the logits and pick the best allowed token ---

            mask = torch.full_like(logits, -float("inf"))
            for tid in allowed:
                if tid < len(mask):
                    mask[tid] = logits[tid]

            next_token_id = int(torch.argmax(mask).item())
            # convert Ġ prefix to a real space before feeding to the tracker
            next_token_text = _to_visible(
                self.id_to_text.get(next_token_id, "")
            )
            prev_state = state

            for ch in next_token_text:
                try:
                    tracker.update(ch)
                except ValueError:
                    break

            output_tokens.append(next_token_id)
            token_ids.append(next_token_id)

            # --- Bookkeeping ---

            # Collect function name tokens until we see the closing quote
            if (
                prev_state == "IN_STRING"
                and key == "name"
                and function_name is None
            ):
                if next_token_id == self.quote_id:
                    function_name = self.model.decode(name_tokens)
                    pre_extracted = _extract_values(
                        prompt,
                        self.schema.get_param_type_info(function_name),
                    )
                    keys_written.add("name")
                else:
                    name_tokens.append(next_token_id)

            new_state = tracker.get_current_state()

            # Mark a string-valued key as done when its value closes
            if (
                prev_state == "IN_STRING"
                and new_state == "AFTER_VALUE"
                and tracker.get_depth() == 1
                and key
            ):
                keys_written.add(key)

            # Mark "parameters" as done when its nested object closes
            if (
                prev_state == "AFTER_VALUE"
                and new_state == "AFTER_VALUE"
                and tracker.get_depth() == 1
            ):
                keys_written.add("parameters")

            if tracker.is_complete():
                break

        return self.model.decode(output_tokens)

    # ------------------------------------------------------------------

    def generate_function_call(self, prompt: str) -> Dict[str, Any]:
        """
        Run generation for the given prompt and return a dict with
        'name' and 'parameters' keys, or raise ValueError on failure.
        """
        raw = self.generate(prompt)

        # find the JSON object in the output
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in output: {raw!r}")

        try:
            call = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON: {e}") from e

        if "name" not in call or "parameters" not in call:
            raise ValueError(f"Output is missing required keys: {call}")

        # Make sure parameter values have the right Python types
        for param_name, param_type in self.schema.get_param_type_info(
            call["name"]
        ).items():
            value = call["parameters"].get(param_name)
            if value is None:
                continue
            try:
                if param_type == "number":
                    call["parameters"][param_name] = float(value)
                elif param_type == "boolean":
                    call["parameters"][param_name] = (
                        value.lower() == "true"
                        if isinstance(value, str)
                        else bool(value)
                    )
                elif param_type == "string":
                    call["parameters"][param_name] = str(value)
            except (ValueError, TypeError):
                pass

        return call