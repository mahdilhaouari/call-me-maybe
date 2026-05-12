# src/constrained_decoder.py
import json
import torch
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, model_validator
from llm_sdk import Small_LLM_Model
from src.json_state_tracker import JSONStateTracker
from src.function_schema import FunctionSchema
from src.token_trie import TokenTrie

# Qwen tokenizer uses U+0120 (Ġ) as a space prefix on BPE tokens
_SPACE_PREFIX = "\u0120"


# ---------------------------------------------------------------------------
# Pydantic value-extraction models (no regex)
# ---------------------------------------------------------------------------

class NumberList(BaseModel):
    values: List[float] = []

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: Any) -> Any:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        values, i = [], 0
        while i < len(text):
            if text[i].isdigit() or (
                text[i] == '-' and i + 1 < len(text) and text[i + 1].isdigit()
            ):
                j, has_dot = i + 1, False
                while j < len(text) and (
                    text[j].isdigit() or (text[j] == '.' and not has_dot)
                ):
                    if text[j] == '.':
                        has_dot = True
                    j += 1
                values.append(float(text[i:j]))
                i = j
            else:
                i += 1
        return {"values": values}


class QuotedStringList(BaseModel):
    values: List[str] = []

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: Any) -> Any:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        values, i = [], 0
        while i < len(text):
            if text[i] in ('"', "'"):
                quote, j, buf = text[i], i + 1, []
                while j < len(text) and text[j] != quote:
                    buf.append(text[j])
                    j += 1
                values.append("".join(buf))
                i = j + 1
            else:
                i += 1
        return {"values": values}


class LastWord(BaseModel):
    """Extracts the last word — fallback for unquoted string params."""
    value: str = ""

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: Any) -> Any:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        words = text.strip().split()
        return {"value": words[-1] if words else ""}


def _extract(prompt: str, param_types: Dict[str, str]) -> Dict[str, Any]:
    """Pre-extract typed parameter values from the prompt using Pydantic models."""
    nums    = NumberList(text=prompt).values        # type: ignore[call-arg]
    strings = QuotedStringList(text=prompt).values  # type: ignore[call-arg]
    result: Dict[str, Any] = {}
    ni = si = 0
    for name, ptype in param_types.items():
        if ptype == "number" and ni < len(nums):
            result[name] = nums[ni]
            ni += 1
        elif ptype == "string":
            if si < len(strings):
                result[name] = strings[si]
                si += 1
            else:
                result[name] = LastWord(text=prompt).value  # type: ignore[call-arg]
    return result


def _to_visible(text: str) -> str:
    """Convert vocab text to visible characters by replacing Ġ with space."""
    return text.replace(_SPACE_PREFIX, " ")


# ---------------------------------------------------------------------------
# Constrained decoder
# ---------------------------------------------------------------------------

class ConstrainedDecoder:
    """
    Generates structured function-call JSON from a natural-language prompt
    using constrained decoding — every token is chosen from the subset that
    keeps the output valid JSON conforming to the function schema.
    """

    def __init__(self, schema_path: str, model: Optional[Small_LLM_Model] = None) -> None:
        self.model  = model or Small_LLM_Model()
        self.schema = FunctionSchema(schema_path)
        self.names  = self.schema.get_allowed_function_names()

        # Build vocabulary maps
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.id2text: Dict[int, str] = {}
        first = next(iter(raw.values()))
        if isinstance(first, str):
            for k, v in raw.items():
                try:
                    self.id2text[int(k)] = v
                except ValueError:
                    pass
        else:
            for k, v in raw.items():
                self.id2text[int(v)] = k

        # visible_first_char → set of token IDs
        # Use _to_visible so Ġ-prefixed tokens are indexed under ' ' not 'Ġ'
        self.tokens_for: Dict[str, Set[int]] = {}
        for tid, text in self.id2text.items():
            if not text:
                continue
            visible = _to_visible(text)
            if visible:
                self.tokens_for.setdefault(visible[0], set()).add(tid)

        self.name_trie = TokenTrie([self._enc(n) for n in self.names])
        self.quote_id  = self._single('"')
        self.rbrace_id = self._single('}')

    # ------------------------------------------------------------------
    def _enc(self, text: str) -> List[int]:
        return self.model.encode(text)[0].tolist()

    def _single(self, text: str) -> Optional[int]:
        ids = self._enc(text)
        return ids[0] if len(ids) == 1 else None

    def _allowed(self, chars: Set[str]) -> Set[int]:
        out: Set[int] = set()
        for ch in chars:
            out.update(self.tokens_for.get(ch, set()))
        return out

    def _inject(
        self,
        text: str,
        ids: List[int],
        out: List[int],
        tracker: JSONStateTracker,
    ) -> None:
        """Inject a known string directly, bypassing the model."""
        # Feed original text chars to tracker (spaces preserved correctly)
        for ch in text:
            try:
                tracker.update(ch)
            except ValueError:
                pass
        for tok in self._enc(text):
            out.append(tok)
            ids.append(tok)

    def _prompt(self, user: str) -> str:
        funcs = "\n".join(
            "  - {}({})".format(
                n,
                ", ".join(f"{p}: {t}" for p, t in self.schema.get_param_type_info(n).items()),
            )
            for n in self.names
        )
        return (
            f"You are a function-calling assistant.\n"
            f"Available functions:\n{funcs}\n"
            f'Output ONLY a JSON object with keys "name" and "parameters".\n'
            f"User: {user}\n"
            f"JSON:"
        )

    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_steps: int = 600) -> str:
        """Run constrained decoding, return raw output string."""
        name_toks:    List[int]      = []
        fn_name:      Optional[str]  = None
        keys_written: Set[str]       = set()
        pre_ext:      Dict[str, Any] = {}
        pending_key:  str            = ""

        tracker = JSONStateTracker()
        ids     = self._enc(self._prompt(prompt))
        out:    List[int] = []

        for _ in range(max_steps):
            logits = torch.tensor(
                self.model.get_logits_from_input_ids(ids), dtype=torch.float32
            )
            state = tracker.get_current_state()
            key   = tracker.current_key

            # Save key name the moment it is fully written
            if state == "AFTER_KEY" and key:
                pending_key = key

            # Inject pre-extracted value instead of letting the LLM guess
            if (
                state == "AFTER_COLON"
                and fn_name
                and pending_key
                and pending_key in pre_ext
                and pending_key not in keys_written
            ):
                val   = pre_ext[pending_key]
                ptype = self.schema.get_param_type_info(fn_name).get(pending_key, "string")
                if ptype == "number":
                    self._inject(
                        str(int(val) if float(val) == int(val) else val),
                        ids, out, tracker,
                    )
                    keys_written.add(pending_key); pending_key = ""; continue
                if ptype == "string":
                    esc = str(val).replace("\\", "\\\\").replace('"', '\\"')
                    self._inject(f'"{esc}"', ids, out, tracker)
                    keys_written.add(pending_key); pending_key = ""; continue

            # Syntactic constraint
            valid = tracker.get_valid_next_chars()
            if not valid:
                break
            allowed = self._allowed(valid)

            # Function-name trie constraint
            if state == "IN_STRING" and key == "name" and fn_name is None:
                nxt = self.name_trie.get_allowed_next_tokens(name_toks)
                if self.name_trie.is_complete_prefix(name_toks) and self.quote_id:
                    nxt.add(self.quote_id)
                allowed &= nxt
                if not allowed:
                    break

            # Boolean value constraint
            elif state == "IN_STRING" and fn_name and key not in ("", "name"):
                if self.schema.get_param_type_info(fn_name).get(key) == "boolean":
                    allowed &= self._allowed(set("truefals")) | (
                        {self.quote_id} if self.quote_id else set()
                    )

            # Block premature closing brace
            if (
                state == "AFTER_VALUE"
                and tracker.get_depth() == 1
                and not {"name", "parameters"}.issubset(keys_written)
                and self.rbrace_id
            ):
                allowed.discard(self.rbrace_id)
                allowed -= self.tokens_for.get("}", set())

            if not allowed:
                allowed = self._allowed(valid)
                if not allowed:
                    break

            # Pick best token
            mask = torch.full_like(logits, -float("inf"))
            for tid in allowed:
                if tid < len(mask):
                    mask[tid] = logits[tid]

            next_id    = int(torch.argmax(mask).item())
            # Convert Ġ to space before feeding tracker
            next_text  = _to_visible(self.id2text.get(next_id, ""))
            prev_state = state

            for ch in next_text:
                try:
                    tracker.update(ch)
                except ValueError:
                    break

            out.append(next_id)
            ids.append(next_id)

            # Track function name tokens
            if prev_state == "IN_STRING" and key == "name" and fn_name is None:
                if next_id == self.quote_id:
                    fn_name = self.model.decode(name_toks)
                    pre_ext = _extract(prompt, self.schema.get_param_type_info(fn_name))
                    keys_written.add("name")
                else:
                    name_toks.append(next_id)

            new_state = tracker.get_current_state()
            if (
                prev_state == "IN_STRING"
                and new_state == "AFTER_VALUE"
                and tracker.get_depth() == 1
                and key
            ):
                keys_written.add(key)
            if (
                prev_state == "AFTER_VALUE"
                and new_state == "AFTER_VALUE"
                and tracker.get_depth() == 1
            ):
                keys_written.add("parameters")

            if tracker.is_complete():
                break

        return self.model.decode(out)

    # ------------------------------------------------------------------
    def generate_function_call(self, prompt: str) -> Dict[str, Any]:
        """Return {'name': str, 'parameters': dict} or raise ValueError."""
        raw   = self.generate(prompt)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON in output: {raw!r}")
        try:
            call = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}") from e
        if "name" not in call or "parameters" not in call:
            raise ValueError(f"Missing keys in: {call}")

        # Coerce parameter types
        for pname, ptype in self.schema.get_param_type_info(call["name"]).items():
            val = call["parameters"].get(pname)
            if val is None:
                continue
            try:
                if ptype == "number":
                    call["parameters"][pname] = float(val)
                elif ptype == "boolean":
                    call["parameters"][pname] = (
                        val.lower() == "true" if isinstance(val, str) else bool(val)
                    )
                elif ptype == "string":
                    call["parameters"][pname] = str(val)
            except (ValueError, TypeError):
                pass

        return call