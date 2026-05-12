# tests/test_suite.py
"""
Comprehensive test suite for the call_me_maybe project.

Run (unit tests only, no model):
    python tests/test_suite.py
    pytest tests/test_suite.py -v -m "not integration"

Run everything including model:
    pytest tests/test_suite.py -v
"""

import json
import sys
import traceback
import tempfile
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.json_state_tracker import JSONStateTracker
from src.token_trie import TokenTrie
from src.function_schema import FunctionSchema
from src.constrained_decoder import (
    _extract_numbers,
    _extract_quoted_strings,
    _extract_param_values,
)


# ===========================================================================
# 1. JSON STATE TRACKER
# ===========================================================================

class TestJSONStateTracker:

    def test_simple_object(self):
        t = JSONStateTracker()
        for ch in '{"name":"John"}':
            t.update(ch)
        assert t.is_complete()

    def test_nested_object(self):
        t = JSONStateTracker()
        for ch in '{"a":{"b":"c"}}':
            t.update(ch)
        assert t.is_complete()

    def test_two_keys(self):
        """{'name':'fn','parameters':{}} — empty nested object must work."""
        t = JSONStateTracker()
        for ch in '{"name":"fn","parameters":{}}':
            t.update(ch)
        assert t.is_complete()

    def test_empty_nested_object(self):
        t = JSONStateTracker()
        for ch in '{"a":{}}':
            t.update(ch)
        assert t.is_complete()

    def test_number_value(self):
        t = JSONStateTracker()
        for ch in '{"a":42}':
            t.update(ch)
        assert t.is_complete()

    def test_float_value(self):
        t = JSONStateTracker()
        for ch in '{"a":3.14}':
            t.update(ch)
        assert t.is_complete()

    def test_negative_number(self):
        t = JSONStateTracker()
        for ch in '{"a":-5}':
            t.update(ch)
        assert t.is_complete()

    def test_current_key_simple(self):
        t = JSONStateTracker()
        for ch in '{"name":':
            t.update(ch)
        assert t.current_key == "name"

    def test_current_key_resets_on_nested(self):
        t = JSONStateTracker()
        for ch in '{"parameters":{':
            t.update(ch)
        assert t.current_key == ""

    def test_current_key_inside_nested(self):
        t = JSONStateTracker()
        for ch in '{"parameters":{"a":':
            t.update(ch)
        assert t.current_key == "a"

    def test_depth_tracking(self):
        t = JSONStateTracker()
        t.update('{')
        assert t.get_depth() == 1
        t2 = JSONStateTracker()
        for ch in '{"x":{':
            t2.update(ch)
        assert t2.get_depth() == 2

    def test_in_number_state(self):
        t = JSONStateTracker()
        for ch in '{"a":':
            t.update(ch)
        t.update('4')
        assert t.get_current_state() == "IN_NUMBER"

    def test_in_number_multi_digit(self):
        t = JSONStateTracker()
        for ch in '{"a":265':
            t.update(ch)
        assert t.get_current_state() == "IN_NUMBER"

    def test_number_terminates_on_comma(self):
        t = JSONStateTracker()
        for ch in '{"a":265,':
            t.update(ch)
        assert t.get_current_state() == "AFTER_COMMA"

    def test_number_terminates_on_rbrace(self):
        t = JSONStateTracker()
        for ch in '{"a":265}':
            t.update(ch)
        assert t.is_complete()

    def test_valid_chars_start(self):
        t = JSONStateTracker()
        assert t.get_valid_next_chars() == {'{'}

    def test_valid_chars_after_key(self):
        t = JSONStateTracker()
        for ch in '{"name"':
            t.update(ch)
        assert t.get_valid_next_chars() == {':'}

    def test_valid_chars_after_colon(self):
        t = JSONStateTracker()
        for ch in '{"name":':
            t.update(ch)
        valid = t.get_valid_next_chars()
        assert '"' in valid
        assert '{' in valid
        assert '0' in valid

    def test_valid_chars_in_key_allows_close_brace(self):
        """IN_KEY (not in quotes) must allow '}' for empty objects."""
        t = JSONStateTracker()
        for ch in '{"a":{':
            t.update(ch)
        assert t.get_current_state() == "IN_KEY"
        assert '}' in t.get_valid_next_chars()

    def test_valid_chars_in_number_includes_terminators(self):
        t = JSONStateTracker()
        for ch in '{"a":2':
            t.update(ch)
        valid = t.get_valid_next_chars()
        assert ',' in valid
        assert '}' in valid
        assert '5' in valid

    def test_escape_in_string(self):
        t = JSONStateTracker()
        for ch in '{"a":"he said \\"hi\\""}':
            t.update(ch)
        assert t.is_complete()

    def test_space_in_string_value(self):
        t = JSONStateTracker()
        for ch in '{"a":"hello world"}':
            t.update(ch)
        assert t.is_complete()

    def test_full_function_call_json(self):
        t = JSONStateTracker()
        for ch in '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}':
            t.update(ch)
        assert t.is_complete()

    def test_full_function_call_with_string_param(self):
        t = JSONStateTracker()
        for ch in '{"name":"fn_greet","parameters":{"name":"Alice"}}':
            t.update(ch)
        assert t.is_complete()

    def test_reset(self):
        t = JSONStateTracker()
        for ch in '{"a":"b"}':
            t.update(ch)
        assert t.is_complete()
        t.reset()
        assert t.get_current_state() == "START"
        assert not t.is_complete()

    def test_invalid_start_raises(self):
        t = JSONStateTracker()
        with pytest.raises(ValueError):
            t.update('"')

    def test_extra_chars_after_end_raises(self):
        t = JSONStateTracker()
        for ch in '{"a":"b"}':
            t.update(ch)
        with pytest.raises(ValueError):
            t.update('x')


# ===========================================================================
# 2. TOKEN TRIE
# ===========================================================================

class TestTokenTrie:

    def test_single_sequence(self):
        trie = TokenTrie([[1, 2, 3]])
        assert trie.get_allowed_next_tokens([]) == {1}
        assert trie.get_allowed_next_tokens([1]) == {2}
        assert trie.get_allowed_next_tokens([1, 2]) == {3}
        assert trie.get_allowed_next_tokens([1, 2, 3]) == set()

    def test_multiple_sequences(self):
        trie = TokenTrie([[1, 2], [1, 3], [4, 5]])
        assert trie.get_allowed_next_tokens([]) == {1, 4}
        assert trie.get_allowed_next_tokens([1]) == {2, 3}

    def test_is_complete_prefix(self):
        trie = TokenTrie([[1, 2, 3], [1, 2]])
        assert trie.is_complete_prefix([1, 2]) is True
        assert trie.is_complete_prefix([1, 2, 3]) is True
        assert trie.is_complete_prefix([1]) is False
        assert trie.is_complete_prefix([]) is False

    def test_invalid_prefix(self):
        trie = TokenTrie([[1, 2, 3]])
        assert trie.get_allowed_next_tokens([9]) == set()
        assert trie.is_complete_prefix([9]) is False

    def test_empty_sequences(self):
        trie = TokenTrie([])
        assert trie.get_allowed_next_tokens([]) == set()

    def test_single_token_sequence(self):
        trie = TokenTrie([[42]])
        assert trie.get_allowed_next_tokens([]) == {42}
        assert trie.is_complete_prefix([42]) is True


# ===========================================================================
# 3. EXTRACTION HELPERS
# ===========================================================================

class TestExtractionHelpers:

    def test_extract_numbers_simple(self):
        assert _extract_numbers("sum of 2 and 3") == [2.0, 3.0]

    def test_extract_numbers_large(self):
        assert _extract_numbers("sum of 265 and 345") == [265.0, 345.0]

    def test_extract_numbers_float(self):
        assert _extract_numbers("value is 3.14") == [3.14]

    def test_extract_numbers_negative(self):
        assert _extract_numbers("temperature is -5") == [-5.0]

    def test_extract_numbers_none(self):
        assert _extract_numbers("greet Alice") == []

    def test_extract_numbers_square_root(self):
        assert _extract_numbers("square root of 16") == [16.0]

    def test_extract_quoted_strings_single(self):
        assert _extract_quoted_strings("reverse 'hello'") == ["hello"]

    def test_extract_quoted_strings_double(self):
        assert _extract_quoted_strings('greet "Alice"') == ["Alice"]

    def test_extract_quoted_strings_multiple(self):
        assert _extract_quoted_strings("replace 'cat' with 'dog'") == ["cat", "dog"]

    def test_extract_quoted_strings_none(self):
        assert _extract_quoted_strings("greet Alice") == []

    def test_extract_param_values_numbers(self):
        result = _extract_param_values(
            "sum of 265 and 345", {"a": "number", "b": "number"}
        )
        assert result == {"a": 265.0, "b": 345.0}

    def test_extract_param_values_string(self):
        result = _extract_param_values("reverse 'hello'", {"s": "string"})
        assert result == {"s": "hello"}

    def test_extract_param_values_two_strings(self):
        """Two quoted strings, two string params — both filled."""
        result = _extract_param_values(
            "replace 'cat' with 'dog'",
            {"source_string": "string", "replacement": "string"},
        )
        assert result["source_string"] == "cat"
        assert result["replacement"] == "dog"

    def test_extract_param_values_fewer_strings_than_params(self):
        """Fewer quoted strings than params — fill what we can."""
        result = _extract_param_values(
            "Replace 'cat' with 'dog'",
            {"source_string": "string", "regex": "string", "replacement": "string"},
        )
        assert result.get("source_string") == "cat"
        assert result.get("regex") == "dog"
        assert "replacement" not in result   # no third quoted string

    def test_extract_param_values_no_match(self):
        result = _extract_param_values("greet Alice", {"a": "number"})
        assert result == {}

    def test_extract_param_values_single_number(self):
        result = _extract_param_values("square root of 16", {"a": "number"})
        assert result == {"a": 16.0}


# ===========================================================================
# 4. FUNCTION SCHEMA
# ===========================================================================

MOCK_SCHEMA = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Greet someone.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_toggle",
        "description": "Toggle a flag.",
        "parameters": {"flag": {"type": "boolean"}},
        "returns": {"type": "boolean"},
    },
]


@pytest.fixture
def schema(tmp_path):
    sf = tmp_path / "functions_definition.json"
    sf.write_text(json.dumps(MOCK_SCHEMA))
    return FunctionSchema(str(sf))


class TestFunctionSchema:

    def test_allowed_names(self, schema):
        names = schema.get_allowed_function_names()
        assert "fn_add_numbers" in names
        assert "fn_greet" in names
        assert "fn_toggle" in names

    def test_param_type_info_number(self, schema):
        assert schema.get_param_type_info("fn_add_numbers") == {"a": "number", "b": "number"}

    def test_param_type_info_string(self, schema):
        assert schema.get_param_type_info("fn_greet") == {"name": "string"}

    def test_param_type_info_boolean(self, schema):
        assert schema.get_param_type_info("fn_toggle") == {"flag": "boolean"}

    def test_validate_valid_number_call(self, schema):
        assert schema.validate_function_call(
            {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
        ) is True

    def test_validate_missing_param(self, schema):
        assert schema.validate_function_call(
            {"name": "fn_add_numbers", "parameters": {"a": 2.0}}
        ) is False

    def test_validate_wrong_type(self, schema):
        assert schema.validate_function_call(
            {"name": "fn_add_numbers", "parameters": {"a": "two", "b": 3.0}}
        ) is False

    def test_validate_unknown_function(self, schema):
        assert schema.validate_function_call(
            {"name": "fn_unknown", "parameters": {}}
        ) is False

    def test_validate_string_param(self, schema):
        assert schema.validate_function_call(
            {"name": "fn_greet", "parameters": {"name": "Alice"}}
        ) is True

    def test_validate_wrong_string_type(self, schema):
        assert schema.validate_function_call(
            {"name": "fn_greet", "parameters": {"name": 123}}
        ) is False


# ===========================================================================
# 5. END-TO-END INTEGRATION TESTS  (require model + schema file)
# ===========================================================================

INTEGRATION_CASES = [
    ("What is the sum of 2 and 3?",      "fn_add_numbers",    {"a": 2.0,   "b": 3.0}),
    ("What is the sum of 265 and 345?",  "fn_add_numbers",    {"a": 265.0, "b": 345.0}),
    ("Greet shrek",                      "fn_greet",          {"name": "shrek"}),
    ("Greet john",                       "fn_greet",          {"name": "john"}),
    ("Reverse the string 'hello'",       "fn_reverse_string", {"s": "hello"}),
    ("Reverse the string 'world'",       "fn_reverse_string", {"s": "world"}),
    ("What is the square root of 16?",   "fn_get_square_root",{"a": 16.0}),
    ("Calculate the square root of 144", "fn_get_square_root",{"a": 144.0}),
]


@pytest.mark.integration
class TestEndToEnd:

    @pytest.fixture(scope="class")
    def decoder(self):
        from src.constrained_decoder import ConstrainedDecoder
        schema_file = Path("data/input/functions_definition.json")
        if not schema_file.exists():
            pytest.skip("functions_definition.json not found — skipping integration tests")
        return ConstrainedDecoder(str(schema_file))

    @pytest.mark.parametrize("prompt,expected_name,expected_params", INTEGRATION_CASES)
    def test_function_call(self, decoder, prompt, expected_name, expected_params):
        result = decoder.generate_function_call(prompt)
        assert result["name"] == expected_name, (
            f"Wrong function for '{prompt}': got {result['name']}"
        )
        for param, expected_val in expected_params.items():
            assert param in result["parameters"], (
                f"Missing param '{param}' for '{prompt}'"
            )
            actual = result["parameters"][param]
            if isinstance(expected_val, float):
                assert abs(actual - expected_val) < 1e-6, (
                    f"Wrong value for '{param}' in '{prompt}': "
                    f"got {actual}, expected {expected_val}"
                )
            else:
                assert actual == expected_val, (
                    f"Wrong value for '{param}' in '{prompt}': "
                    f"got {actual!r}, expected {expected_val!r}"
                )

    def test_output_always_has_required_keys(self, decoder):
        for prompt, _, _ in INTEGRATION_CASES:
            result = decoder.generate_function_call(prompt)
            assert isinstance(result, dict)
            assert "name" in result
            assert "parameters" in result
            assert isinstance(result["parameters"], dict)

    def test_output_is_valid_json_always(self, decoder):
        for prompt, _, _ in INTEGRATION_CASES:
            raw = decoder.generate(prompt)
            start, end = raw.find("{"), raw.rfind("}") + 1
            assert start != -1 and end > 0, f"No JSON found for: {prompt}"
            parsed = json.loads(raw[start:end])
            assert isinstance(parsed, dict)

    def test_function_name_always_in_schema(self, decoder):
        schema = FunctionSchema("data/input/functions_definition.json")
        valid_names = set(schema.get_allowed_function_names())
        for prompt, _, _ in INTEGRATION_CASES:
            result = decoder.generate_function_call(prompt)
            assert result["name"] in valid_names, (
                f"Unknown function '{result['name']}' for: {prompt}"
            )

    def test_ambiguous_prompt_still_returns_json(self, decoder):
        result = decoder.generate_function_call("Do something with 42")
        assert isinstance(result, dict)
        assert "name" in result
        assert "parameters" in result


# ===========================================================================
# 6. STANDALONE RUNNER (no pytest needed)
# ===========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNIT TESTS  (no model needed)")
    print("=" * 60)

    passed = failed = 0

    for cls in [TestJSONStateTracker, TestTokenTrie, TestExtractionHelpers]:
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

    print(f"\n── TestFunctionSchema ──")
    with tempfile.TemporaryDirectory() as tmp:
        sf = os.path.join(tmp, "functions_definition.json")
        with open(sf, "w") as fh:
            json.dump(MOCK_SCHEMA, fh)
        schema_obj = FunctionSchema(sf)
        methods = sorted(m for m in dir(TestFunctionSchema) if m.startswith("test_"))
        for method in methods:
            try:
                getattr(TestFunctionSchema(), method)(schema_obj)
                print(f"  ✅ {method}")
                passed += 1
            except Exception:
                print(f"  ❌ {method}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"UNIT RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    schema_path = Path("data/input/functions_definition.json")
    if schema_path.exists():
        print(f"\n{'='*60}")
        print("INTEGRATION TESTS  (loads model — may be slow)")
        print("=" * 60)
        from src.constrained_decoder import ConstrainedDecoder
        decoder_obj = ConstrainedDecoder(str(schema_path))
        ip = if_ = 0
        for prompt, exp_name, exp_params in INTEGRATION_CASES:
            try:
                result = decoder_obj.generate_function_call(prompt)
                assert result["name"] == exp_name
                for k, v in exp_params.items():
                    actual = result["parameters"][k]
                    if isinstance(v, float):
                        assert abs(actual - v) < 1e-6
                    else:
                        assert actual == v
                print(f"  ✅ {prompt}")
                ip += 1
            except Exception as ex:
                print(f"  ❌ {prompt}")
                print(f"     {ex}")
                if_ += 1
        print(f"\nINTEGRATION RESULTS: {ip} passed, {if_} failed")
    else:
        print("\n⚠️  Skipping integration tests (functions_definition.json not found)")