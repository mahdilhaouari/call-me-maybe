# tests/test_suite.py
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict
import pytest
from src.json_state_tracker import JSONStateTracker  # type: ignore
from src.token_trie import TokenTrie  # type: ignore
from src.function_schema import FunctionSchema  # type: ignore
from src.constrained_decoder import (  # type: ignore
    NumberList,
    QuotedStringList,
    LastWord,
    _extract_values,
)
"""
Test suite for the call_me_maybe project.

Run unit tests only (no model needed):
    python tests/test_suite.py
    pytest tests/test_suite.py -v -m "not integration"

Run everything including model:
    pytest tests/test_suite.py -v
"""


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Minimal schema used for unit tests (no file needed)
# ---------------------------------------------------------------------------

MOCK_SCHEMA = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Greet someone.",
        "parameters": {
            "name": {"type": "string"},
        },
        "returns": {"type": "string"},
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string.",
        "parameters": {
            "s": {"type": "string"},
        },
        "returns": {"type": "string"},
    },
    {
        "name": "fn_get_square_root",
        "description": "Get square root of a number.",
        "parameters": {
            "a": {"type": "number"},
        },
        "returns": {"type": "number"},
    },
    {
        "name": "fn_substitute_string_with_regex",
        "description": "Substitute matches in a string.",
        "parameters": {
            "source_string": {"type": "string"},
            "regex": {"type": "string"},
            "replacement": {"type": "string"},
        },
        "returns": {"type": "string"},
    },
]


@pytest.fixture
def schema(tmp_path: Path) -> FunctionSchema:
    schema_file = tmp_path / "functions_definition.json"
    schema_file.write_text(json.dumps(MOCK_SCHEMA))
    return FunctionSchema(str(schema_file))


# ---------------------------------------------------------------------------
# Small helpers to keep test lines short
# ---------------------------------------------------------------------------

def get_numbers(text: str) -> list:
    """Shorthand for NumberList extraction."""
    return NumberList(text=text).values  # type: ignore[call-arg]


def get_strings(text: str) -> list:
    """Shorthand for QuotedStringList extraction."""
    return QuotedStringList(text=text).values  # type: ignore[call-arg]


def get_last_word(text: str) -> str:
    """Shorthand for LastWord extraction."""
    return LastWord(text=text).value  # type: ignore[call-arg]


# ===========================================================================
# 1. JSON STATE TRACKER
# ===========================================================================

class TestJSONStateTracker:

    def test_simple_object_reaches_end(self) -> None:
        t = JSONStateTracker()
        for ch in '{"name":"John"}':
            t.update(ch)
        assert t.is_complete()

    def test_nested_object_reaches_end(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":{"b":"c"}}':
            t.update(ch)
        assert t.is_complete()

    def test_empty_nested_object(self) -> None:
        """Empty object like {} must not crash."""
        t = JSONStateTracker()
        for ch in '{"a":{}}':
            t.update(ch)
        assert t.is_complete()

    def test_two_keys_with_empty_nested(self) -> None:
        """The real output format: name + empty parameters object."""
        t = JSONStateTracker()
        for ch in '{"name":"fn","parameters":{}}':
            t.update(ch)
        assert t.is_complete()

    def test_full_function_call_string_param(self) -> None:
        t = JSONStateTracker()
        for ch in '{"name":"fn_greet","parameters":{"name":"Alice"}}':
            t.update(ch)
        assert t.is_complete()

    def test_full_function_call_number_param(self) -> None:
        t = JSONStateTracker()
        for ch in '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}':
            t.update(ch)
        assert t.is_complete()

    def test_negative_number_value(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":-5}':
            t.update(ch)
        assert t.is_complete()

    def test_float_value(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":3.14}':
            t.update(ch)
        assert t.is_complete()

    def test_space_in_string_value(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":"hello world"}':
            t.update(ch)
        assert t.is_complete()

    def test_escaped_quote_in_string(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":"say \\"hi\\""}':
            t.update(ch)
        assert t.is_complete()

    def test_current_key_after_colon(self) -> None:
        """current_key should hold the key name at AFTER_COLON."""
        t = JSONStateTracker()
        for ch in '{"name":':
            t.update(ch)
        assert t.current_key == "name"

    def test_current_key_resets_on_nested_object(self) -> None:
        """Opening a nested { must reset current_key to empty string."""
        t = JSONStateTracker()
        for ch in '{"parameters":{':
            t.update(ch)
        assert t.current_key == ""

    def test_current_key_inside_nested_object(self) -> None:
        """current_key should be 'a' after opening the nested key."""
        t = JSONStateTracker()
        for ch in '{"parameters":{"a":':
            t.update(ch)
        assert t.current_key == "a"

    def test_depth_increases_on_open_brace(self) -> None:
        t = JSONStateTracker()
        t.update("{")
        assert t.get_depth() == 1

    def test_depth_increases_on_nested_object(self) -> None:
        t = JSONStateTracker()
        for ch in '{"x":{':
            t.update(ch)
        assert t.get_depth() == 2

    def test_enters_in_number_state(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":':
            t.update(ch)
        t.update("4")
        assert t.get_current_state() == "IN_NUMBER"

    def test_stays_in_number_for_multiple_digits(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":265':
            t.update(ch)
        assert t.get_current_state() == "IN_NUMBER"

    def test_number_ends_on_comma(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":265,':
            t.update(ch)
        assert t.get_current_state() == "AFTER_COMMA"

    def test_number_ends_on_closing_brace(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":265}':
            t.update(ch)
        assert t.is_complete()

    def test_valid_chars_at_start(self) -> None:
        t = JSONStateTracker()
        assert t.get_valid_next_chars() == {"{"}

    def test_valid_chars_after_key(self) -> None:
        t = JSONStateTracker()
        for ch in '{"name"':
            t.update(ch)
        assert t.get_valid_next_chars() == {":"}

    def test_valid_chars_after_colon_includes_quote(self) -> None:
        t = JSONStateTracker()
        for ch in '{"name":':
            t.update(ch)
        assert '"' in t.get_valid_next_chars()

    def test_valid_chars_in_number_includes_terminators(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":2':
            t.update(ch)
        valid = t.get_valid_next_chars()
        assert "," in valid
        assert "}" in valid

    def test_valid_chars_in_key_allows_close_brace(self) -> None:
        """Empty object: IN_KEY (not in quotes) must allow }."""
        t = JSONStateTracker()
        for ch in '{"a":{':
            t.update(ch)
        assert "}" in t.get_valid_next_chars()

    def test_valid_chars_in_string_includes_space(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":"':
            t.update(ch)
        assert " " in t.get_valid_next_chars()

    def test_invalid_start_raises(self) -> None:
        t = JSONStateTracker()
        with pytest.raises(ValueError):
            t.update('"')

    def test_extra_chars_after_end_raises(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":"b"}':
            t.update(ch)
        with pytest.raises(ValueError):
            t.update("x")

    def test_reset_returns_to_start(self) -> None:
        t = JSONStateTracker()
        for ch in '{"a":"b"}':
            t.update(ch)
        assert t.is_complete()
        t.reset()
        assert t.get_current_state() == "START"
        assert not t.is_complete()


# ===========================================================================
# 2. TOKEN TRIE
# ===========================================================================

class TestTokenTrie:

    def test_single_sequence_navigation(self) -> None:
        trie = TokenTrie([[1, 2, 3]])
        assert trie.get_allowed_next_tokens([]) == {1}
        assert trie.get_allowed_next_tokens([1]) == {2}
        assert trie.get_allowed_next_tokens([1, 2]) == {3}
        assert trie.get_allowed_next_tokens([1, 2, 3]) == set()

    def test_multiple_sequences_share_prefix(self) -> None:
        trie = TokenTrie([[1, 2], [1, 3], [4, 5]])
        assert trie.get_allowed_next_tokens([]) == {1, 4}
        assert trie.get_allowed_next_tokens([1]) == {2, 3}

    def test_complete_prefix_detection(self) -> None:
        trie = TokenTrie([[1, 2, 3], [1, 2]])
        assert trie.is_complete_prefix([1, 2]) is True
        assert trie.is_complete_prefix([1, 2, 3]) is True
        assert trie.is_complete_prefix([1]) is False
        assert trie.is_complete_prefix([]) is False

    def test_invalid_prefix_returns_empty(self) -> None:
        trie = TokenTrie([[1, 2, 3]])
        assert trie.get_allowed_next_tokens([9]) == set()
        assert trie.is_complete_prefix([9]) is False

    def test_empty_trie(self) -> None:
        trie = TokenTrie([])
        assert trie.get_allowed_next_tokens([]) == set()

    def test_single_token_sequence(self) -> None:
        trie = TokenTrie([[42]])
        assert trie.get_allowed_next_tokens([]) == {42}
        assert trie.is_complete_prefix([42]) is True


# ===========================================================================
# 3. VALUE EXTRACTION (NumberList, QuotedStringList, LastWord, _extract_values)
# ===========================================================================

class TestNumberList:

    def test_two_positive_numbers(self) -> None:
        result = get_numbers("sum of 2 and 3")
        assert result == [2.0, 3.0]

    def test_large_numbers(self) -> None:
        result = get_numbers("265 and 345")
        assert result == [265.0, 345.0]

    def test_negative_number(self) -> None:
        result = get_numbers("sum of -2 and 3")
        assert result == [-2.0, 3.0]

    def test_float_number(self) -> None:
        result = get_numbers("value is 3.14")
        assert result == [3.14]

    def test_no_numbers(self) -> None:
        result = get_numbers("greet Alice")
        assert result == []

    def test_square_root_prompt(self) -> None:
        result = get_numbers("square root of 16")
        assert result == [16.0]

    def test_number_144(self) -> None:
        result = get_numbers("Calculate the square root of 144")
        assert result == [144.0]


class TestQuotedStringList:

    def test_single_quoted_string(self) -> None:
        result = get_strings("reverse 'hello'")
        assert result == ["hello"]

    def test_double_quoted_string(self) -> None:
        result = get_strings('greet "Alice"')
        assert result == ["Alice"]

    def test_multiple_quoted_strings(self) -> None:
        result = QuotedStringList(  # type: ignore[call-arg]
            text="replace 'cat' with 'dog'"
        ).values
        assert result == ["cat", "dog"]

    def test_string_with_spaces(self) -> None:
        result = QuotedStringList(  # type: ignore[call-arg]
            text="'Programming is fun'"
        ).values
        assert result == ["Programming is fun"]

    def test_string_with_numbers_inside(self) -> None:
        result = QuotedStringList(  # type: ignore[call-arg]
            text='"Hello 34 I\'m 233 years old"'
        ).values
        assert result == ["Hello 34 I'm 233 years old"]

    def test_no_quoted_strings(self) -> None:
        result = get_strings("greet Alice")
        assert result == []

    def test_three_quoted_strings(self) -> None:
        result = QuotedStringList(  # type: ignore[call-arg]
            text="'The cat sat on the mat' 'cat' 'dog'"
        ).values
        assert len(result) == 3


class TestLastWord:

    def test_last_word_simple(self) -> None:
        result = get_last_word("greet mahdi")
        assert result == "mahdi"

    def test_last_word_single(self) -> None:
        result = get_last_word("Alice")
        assert result == "Alice"

    def test_last_word_long_sentence(self) -> None:
        result = get_last_word("heey how are you greet mahdi")
        assert result == "mahdi"

    def test_empty_text(self) -> None:
        result = get_last_word("")
        assert result == ""


class TestExtractValues:

    def test_two_number_params(self) -> None:
        result = _extract_values(
            "sum of 265 and 345",
            {"a": "number", "b": "number"},
        )
        assert result == {"a": 265.0, "b": 345.0}

    def test_negative_and_positive(self) -> None:
        result = _extract_values(
            "sum of -2 and 3",
            {"a": "number", "b": "number"},
        )
        assert result == {"a": -2.0, "b": 3.0}

    def test_single_number_param(self) -> None:
        result = _extract_values(
            "square root of 16",
            {"a": "number"},
        )
        assert result == {"a": 16.0}

    def test_quoted_string_param(self) -> None:
        result = _extract_values(
            "reverse 'hello'",
            {"s": "string"},
        )
        assert result == {"s": "hello"}

    def test_fallback_to_last_word(self) -> None:
        """No quoted string → fall back to last word of prompt."""
        result = _extract_values(
            "greet mahdi",
            {"name": "string"},
        )
        assert result == {"name": "mahdi"}

    def test_three_string_params(self) -> None:
        result = _extract_values(
            "Replace 'cat' with 'dog' in 'The cat sat on the mat'",
            {
                "source_string": "string",
                "regex": "string",
                "replacement": "string",
            },
        )
        assert result["source_string"] == "cat"
        assert result["regex"] == "dog"
        assert result["replacement"] == "The cat sat on the mat"

    def test_no_match_returns_empty(self) -> None:
        result = _extract_values("greet Alice", {"a": "number"})
        assert result == {}


# ===========================================================================
# 4. FUNCTION SCHEMA
# ===========================================================================

class TestFunctionSchema:

    def test_correct_function_names_loaded(
        self, schema: FunctionSchema
    ) -> None:
        names = schema.get_allowed_function_names()
        assert "fn_add_numbers" in names
        assert "fn_greet" in names
        assert "fn_reverse_string" in names
        assert "fn_get_square_root" in names
        assert "fn_substitute_string_with_regex" in names

    def test_param_types_for_add_numbers(
        self, schema: FunctionSchema
    ) -> None:
        info = schema.get_param_type_info("fn_add_numbers")
        assert info == {"a": "number", "b": "number"}

    def test_param_types_for_greet(self, schema: FunctionSchema) -> None:
        info = schema.get_param_type_info("fn_greet")
        assert info == {"name": "string"}

    def test_param_types_for_reverse_string(
        self, schema: FunctionSchema
    ) -> None:
        info = schema.get_param_type_info("fn_reverse_string")
        assert info == {"s": "string"}

    def test_param_types_for_square_root(
        self, schema: FunctionSchema
    ) -> None:
        info = schema.get_param_type_info("fn_get_square_root")
        assert info == {"a": "number"}

    def test_param_types_for_substitute(
        self, schema: FunctionSchema
    ) -> None:
        info = schema.get_param_type_info("fn_substitute_string_with_regex")
        assert info == {
            "source_string": "string",
            "regex": "string",
            "replacement": "string",
        }

    def test_validate_valid_add_numbers(
        self, schema: FunctionSchema
    ) -> None:
        assert schema.validate_function_call({
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }) is True

    def test_validate_valid_greet(self, schema: FunctionSchema) -> None:
        assert schema.validate_function_call({
            "name": "fn_greet",
            "parameters": {"name": "Alice"},
        }) is True

    def test_validate_valid_substitute(
        self, schema: FunctionSchema
    ) -> None:
        assert schema.validate_function_call({
            "name": "fn_substitute_string_with_regex",
            "parameters": {
                "source_string": "hello world",
                "regex": "\\d+",
                "replacement": "NUM",
            },
        }) is True

    def test_validate_missing_parameter(
        self, schema: FunctionSchema
    ) -> None:
        assert schema.validate_function_call({
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0},
        }) is False

    def test_validate_wrong_type(self, schema: FunctionSchema) -> None:
        assert schema.validate_function_call({
            "name": "fn_add_numbers",
            "parameters": {"a": "two", "b": 3.0},
        }) is False

    def test_validate_unknown_function(
        self, schema: FunctionSchema
    ) -> None:
        assert schema.validate_function_call({
            "name": "fn_unknown",
            "parameters": {},
        }) is False

    def test_unknown_function_returns_empty_param_info(
        self, schema: FunctionSchema
    ) -> None:
        assert schema.get_param_type_info("fn_unknown") == {}


# ===========================================================================
# 5. INTEGRATION TESTS  (require model + real schema file)
# ===========================================================================

INTEGRATION_CASES = [
    (
        "What is the sum of -2 and 3?",
        "fn_add_numbers",
        {"a": -2.0, "b": 3.0},
    ),
    (
        "What is the sum of 265 and 345?",
        "fn_add_numbers",
        {"a": 265.0, "b": 345.0},
    ),
    (
        "heey how are you greet mahdi",
        "fn_greet",
        {"name": "mahdi"},
    ),
    (
        "Greet shrek",
        "fn_greet",
        {"name": "shrek"},
    ),
    (
        "Greet john",
        "fn_greet",
        {"name": "john"},
    ),
    (
        "Reverse the string 'hello'",
        "fn_reverse_string",
        {"s": "hello"},
    ),
    (
        "Reverse the string 'world'",
        "fn_reverse_string",
        {"s": "world"},
    ),
    (
        "What is the square root of 16?",
        "fn_get_square_root",
        {"a": 16.0},
    ),
    (
        "Calculate the square root of 144",
        "fn_get_square_root",
        {"a": 144.0},
    ),
    (
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
        "fn_substitute_string_with_regex",
        {"source_string": "Hello 34 I'm 233 years old"},
    ),
    (
        "Replace all vowels in 'Programming is fun' with asterisks",
        "fn_substitute_string_with_regex",
        {"source_string": "Programming is fun"},
    ),
    (
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        "fn_substitute_string_with_regex",
        {"source_string": "The cat sat on the mat with another cat"},
    ),
]

SCHEMA_PATH = "data/input/functions_definition.json"


@pytest.mark.integration
class TestEndToEnd:

    @pytest.fixture(scope="class")
    def decoder(self):  # type: ignore[override]
        from src.constrained_decoder import ConstrainedDecoder
        if not Path(SCHEMA_PATH).exists():
            pytest.skip("functions_definition.json not found")
        return ConstrainedDecoder(SCHEMA_PATH)

    @pytest.mark.parametrize(
        "prompt,expected_name,expected_params", INTEGRATION_CASES
    )
    def test_correct_function_selected(
        self,
        decoder: Any,
        prompt: str,
        expected_name: str,
        expected_params: Dict[str, Any],
    ) -> None:
        result = decoder.generate_function_call(prompt)
        assert result["name"] == expected_name, (
            f"Wrong function for: {prompt!r}\n"
            f"Expected: {expected_name}\n"
            f"Got:      {result['name']}"
        )

    @pytest.mark.parametrize(
        "prompt,expected_name,expected_params", INTEGRATION_CASES
    )
    def test_correct_parameters(
        self,
        decoder: Any,
        prompt: str,
        expected_name: str,
        expected_params: Dict[str, Any],
    ) -> None:
        result = decoder.generate_function_call(prompt)
        for param, expected_value in expected_params.items():
            assert param in result["parameters"], (
                f"Missing param {param!r} for: {prompt!r}"
            )
            actual = result["parameters"][param]
            if isinstance(expected_value, float):
                assert abs(actual - expected_value) < 1e-6, (
                    f"Wrong value for {param!r} in: {prompt!r}\n"
                    f"Expected: {expected_value}\n"
                    f"Got:      {actual}"
                )
            else:
                assert actual == expected_value, (
                    f"Wrong value for {param!r} in: {prompt!r}\n"
                    f"Expected: {expected_value!r}\n"
                    f"Got:      {actual!r}"
                )

    def test_output_is_always_valid_json(self, decoder: Any) -> None:
        """Every prompt must produce parseable JSON with name + parameters."""
        for prompt, _, _ in INTEGRATION_CASES:
            result = decoder.generate_function_call(prompt)
            assert isinstance(result, dict)
            assert "name" in result
            assert "parameters" in result
            assert isinstance(result["parameters"], dict)

    def test_function_name_always_in_schema(self, decoder: Any) -> None:
        """The generated function name must always be one we know about."""
        valid_names = set(
            FunctionSchema(SCHEMA_PATH).get_allowed_function_names()
        )
        for prompt, _, _ in INTEGRATION_CASES:
            result = decoder.generate_function_call(prompt)
            assert result["name"] in valid_names, (
                f"Unknown function {result['name']!r} for: {prompt!r}"
            )


# ===========================================================================
# 6. STANDALONE RUNNER (no pytest needed)
# ===========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNIT TESTS  (no model needed)")
    print("=" * 60)

    passed = failed = 0

    # Run tracker, trie, extraction tests
    unit_classes = [
        TestJSONStateTracker,
        TestTokenTrie,
        TestNumberList,
        TestQuotedStringList,
        TestLastWord,
        TestExtractValues,
    ]

    for cls in unit_classes:
        methods = sorted(m for m in dir(cls) if m.startswith("test_"))
        print(f"\n── {cls.__name__} ({len(methods)} tests) ──")
        for method in methods:
            try:
                getattr(cls(), method)()
                print(f"  ✅ {method}")
                passed += 1
            except Exception:
                print(f"  ❌ {method}")
                traceback.print_exc()
                failed += 1

    # Run schema tests (need a temp file)
    print("\n── TestFunctionSchema ──")
    with tempfile.TemporaryDirectory() as tmp:
        sf = os.path.join(tmp, "functions_definition.json")
        with open(sf, "w") as fh:
            json.dump(MOCK_SCHEMA, fh)
        schema_obj = FunctionSchema(sf)
        methods = sorted(
            m for m in dir(TestFunctionSchema) if m.startswith("test_")
        )
        for method in methods:
            try:
                getattr(TestFunctionSchema(), method)(schema_obj)
                print(f"  ✅ {method}")
                passed += 1
            except Exception:
                print(f"  ❌ {method}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"UNIT RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    # Integration tests if schema file exists
    if Path(SCHEMA_PATH).exists():
        print(f"\n{'=' * 60}")
        print("INTEGRATION TESTS  (loads model — may be slow)")
        print("=" * 60)
        from src.constrained_decoder import ConstrainedDecoder
        decoder_obj = ConstrainedDecoder(SCHEMA_PATH)
        ip = if_ = 0
        for prompt, exp_name, exp_params in INTEGRATION_CASES:
            try:
                result = decoder_obj.generate_function_call(prompt)
                assert result["name"] == exp_name
                for k, v in exp_params.items():  # type: ignore
                    actual = result["parameters"][k]
                    if isinstance(v, float):
                        assert abs(actual - v) < 1e-6
                    else:
                        assert actual == v
                print(f"  ✅ {prompt[:60]}")
                ip += 1
            except Exception as ex:
                print(f"  ❌ {prompt[:60]}")
                print(f"     {ex}")
                if_ += 1
        print(f"\nINTEGRATION RESULTS: {ip} passed, {if_} failed")
    else:
        print("\n⚠️  Skipping integration tests")
        print(f"   ({SCHEMA_PATH} not found)")
