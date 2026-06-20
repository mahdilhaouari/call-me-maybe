# src/main.py
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constrained_decoder import ConstrainedDecoder


def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {path}: {e}")
        sys.exit(1)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Output saved to {path}")


def get_prompts(data: Any) -> List[str]:
    """Accept a list of strings or a list of {"prompt": "..."} objects."""
    if not isinstance(data, list):
        print("❌ Input file must contain a JSON array.")
        sys.exit(1)

    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(item["prompt"])
        else:
            print(f"⚠️  Skipping unrecognised item: {item}")

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to write the output JSON file.",
    )
    args = parser.parse_args()

    prompts = get_prompts(load_json(args.input))
    if not prompts:
        print("❌ No valid prompts found in input file.")
        sys.exit(1)

    print("🚀 Loading model and schema ...")
    decoder = ConstrainedDecoder(args.functions_definition)
    print(f"✅ Ready — processing {len(prompts)} prompt(s)\n")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:70]}")
        try:
            call = decoder.generate_function_call(prompt)

            # Check if the decoder returned a no-match response
            if call.get("name") == "no_function_found":
                print(f"        ⚠️  {call.get('error', 'No match found')}")
                results.append({
                    "prompt": prompt,
                    "fn_name": "no_function_found",
                    "args": {},
                    "error": call.get("error", ""),
                })
            else:
                results.append({
                    "prompt": prompt,
                    "fn_name": call["name"],
                    "args": call["parameters"],
                })
                print(f"        → {call['name']}({call['parameters']})")

        except Exception as e:
            print(f"        ❌ Error: {e}")
            results.append({
                "prompt": prompt,
                "fn_name": "error",
                "args": {},
                "error": str(e),
            })

    save_json(args.output, results)


if __name__ == "__main__":
    main()