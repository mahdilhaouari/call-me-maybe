*This project has been created as part of the 42 curriculum by mal-haou

## call me maybe
introduction to function calling in LLMs

this project is about tool that translates natural language prompts into structured function calls
using constrained decoding — guaranteeing 100% valid JSON output even with a
small 0.6B parameter language model.

## Description

Large Language Models are powerful at understanding language, but they don't
naturally produce structured, machine-executable output. This project bridges
that gap by implementing constrained decoding: at every generation step,
we intercept the model's token probabilities and mask out any token that would
produce invalid JSON or violate the function schema. The model is then forced
to pick only from the tokens we allow.
Given a prompt like:

    What is the sum of 265 and 345?

The system outputs:

    {
        "name": "fn_add_numbers",
        "parameters": {
            "a": 265.0,
            "b": 345.0
        }
    }

The model used is Qwen/Qwen3-0.6B — only 500 million parameters. Without
constraints it produces valid JSON roughly 30% of the time. With constrained
decoding it reaches near-perfect reliability.



### Project Structure
```
├── src
│   ├── __init__.py
│   ├── main.py                  > entry point
│   ├── constrained_decoder.py   > core logic
│   ├── json_state_tracker.py    > JSON state machine
│   ├── function_schema.py       > schema loader and validator
│   └── token_trie.py            > prefix trie for function names
├── llm_sdk
│   ├── __init__.py
│   └── llm_sdk.py               > provided model wrapper
├── data
│   ├── input
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output                   > generated at runtime
├── tests
│   └── test_suite.py
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Instructions

Requirements:

- Python 3.10 or later
- uv package manager

Install uv:

- curl -LsSf https://astral.sh/uv/install.sh | sh

then you can control the project using the makefile.


### I_ Algorithm Explanation:

```
1. Build a prompt listing all available functions
2. Encode the prompt into token IDs
3. Loop:
   a. Ask the model for logits (one score per vocabulary token)
   b. Ask the JSON state machine which characters are valid right now
   c. Convert valid characters → valid token IDs
   d. Apply trie constraint if we are generating the function name
   e. Set logits of all other tokens to -infinity
   f. Pick the token with the highest remaining score (argmax)
   g. Feed the token's character(s) to the state machine
   h. Append the token to the output
   i. Stop when the state machine reaches END
```

### II_ JSON State Machine (JSONStateTracker):
Tracks where we are in the JSON structure character by character. Nine states:
```
START → IN_KEY → AFTER_KEY → AFTER_COLON → IN_STRING → AFTER_VALUE → AFTER_COMMA → IN_KEY → END
                                         → IN_NUMBER
```

At every step it tells the decoder exactly which characters are legal, so the
output is always syntactically valid JSON.

### III_ Token Trie (TokenTrie):
Function names are stored as sequences of token IDs in a prefix trie. While
the model generates the function name string, the trie constrains it to only
tokens that continue a valid function name. This guarantees the model always
picks a function that actually exists.


### IV_ Value Pre-extraction:
The 0.6B model is too small to reliably copy numbers from the prompt. Instead
of relying on it, we extract values directly from the prompt text using Pydantic
models as soon as we know the function name, then inject them character by
character — bypassing the model for that step. The model still does the hard
work: choosing the function and the parameter names.

### V_ Design Decisions:
- ***Why constrained decoding instead of prompting?***

Prompting a small model to produce JSON works only ~30% of the time. Constrained
decoding works 100% of the time because invalid tokens are mathematically
impossible to select.

- ***Why a state machine instead of a JSON parser?***

A full parser works on complete documents. We need to validate partial output
character by character during generation, which requires a streaming approach —
exactly what a state machine provides.

- ***Why pre-extract values instead of letting the model generate them?***

The model consistently fails to copy exact numbers from the prompt (it generates
the most statistically common digits instead). Pre-extraction with Pydantic
models is reliable and deterministic.

- ***Why Pydantic for extraction?***

The subject requires Pydantic for all validation. We extended its use to value
extraction via model_validator, keeping the codebase consistent and removing
any dependency on the re module.

- ***Why a token trie for function names?***

Without the trie, the model might generate a function name that doesn't exist.
The trie constrains generation to only valid continuations at every token step,
making it impossible to produce an unknown function name.

### VI_ Challenges Faced
**1. Numbers with more than one digit**

The first version of the state machine would finish reading a number value the moment it saw the very first digit. So when the prompt said 265, it would read 2 and immediately move on, ignoring 65. We fixed this by adding a dedicated IN_NUMBER state that keeps reading digits until it sees a comma or closing brace — only then does it consider the number finished.

**2. Key names getting mixed together**

Our JSON tracker keeps track of the current key name by building it character by character. The problem was that when we opened the nested parameters object — going from {"name": ..., "parameters": { to the keys inside — the tracker never cleared the key name it had built. So when it started reading the key a inside parameters, it still remembered parameters from before and ended up with parametersa instead of just a. The fix was simple: clear the key name to an empty string every time we open a new nested object with {.

**3. Spaces disappearing from string values**

The Qwen model's vocabulary stores space-starting tokens with a special invisible character Ġ at the front — so the token for " hello" (space + hello) is written as Ġhello in the vocabulary file. Our code was reading the first character of each token to build a lookup table, but it was reading Ġ instead of a space — so space-starting tokens were never found when we needed them. The result was that "Hello 34 I'm 233 years old" came out as "Hello34I'm233yearsold" with all spaces removed. We fixed it by converting Ġ to a real space character before doing any lookups.

**4. The JSON closing too early**

The model kept generating {"name": "fn_add_numbers"} and stopping — a valid JSON object, but missing the parameters key entirely. From the model's perspective, closing the object early was perfectly reasonable because it had already written one key. We fixed this by checking at every step whether both name and parameters have been written, and simply blocking the } token from being selected until both are present.

**5. Losing track of which key we were writing**

When the decoder needs to inject a pre-extracted value (like 265 for parameter a), it needs to know which parameter it is currently writing. The problem was that the tracker resets the current key name to empty as soon as it sees a comma — so by the time we reached the point where we wanted to inject the value, the key name was already gone. We fixed this by saving a copy of the key name the moment the key is fully written, before the tracker has a chance to clear it.

### VII_ Testing Strategy

We wrote a test suite in tests/test_suite.py covering every component individually — the state machine, the trie, the value extraction, and the schema validator — without needing to load the model. Then we ran the full pipeline on all 12 real prompts and checked that each one produced the correct function name and parameter values.

## Resources

### References:

- Qwen3 Model — the language model used.

- Pydantic documentation — used for validation and extraction.

- Constrained decoding paper — Guidance: A Grammar-Based Approach to Constrained Generation.

- BPE tokenization — how the Qwen tokenizer splits text into tokens.

### How AI was used:

AI was used in this project for the following tasks:

- Debugging the IN_NUMBER state machine issue and the current_key
accumulation bug.

- Explaining the Ġ (U+0120) BPE space-prefix behaviour of the Qwen tokenizer.

- Generating the initial structure of the test suite.

- review every part of my code and help in optimization.

