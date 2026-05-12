# src/main.py
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constrained_decoder import ConstrainedDecoder


def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {path}"); sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {path}: {e}"); sys.exit(1)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Output written to {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--functions_definition", default="data/input/functions_definition.json")
    p.add_argument("--input",  default="data/input/function_calling_tests.json")
    p.add_argument("--output", default="data/output/function_calling_results.json")
    args = p.parse_args()

    raw = load_json(args.input)
    if not isinstance(raw, list):
        print("❌ Input must be a JSON array."); sys.exit(1)

    prompts: List[str] = []
    for item in raw:
        if isinstance(item, str):              prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item: prompts.append(item["prompt"])

    if not prompts:
        print("❌ No prompts found."); sys.exit(1)

    print(f"🚀 Loading model …")
    decoder = ConstrainedDecoder(args.functions_definition)
    print(f"✅ Ready — {len(prompts)} prompt(s)\n")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:70]}")
        try:
            call = decoder.generate_function_call(prompt)
            results.append({"prompt": prompt, "name": call["name"], "parameters": call["parameters"]})
            print(f"       → {call['name']}({call['parameters']})")
        except Exception as e:
            print(f"       ❌ {e}")
            results.append({"prompt": prompt, "name": "error", "parameters": {}})

    save_json(args.output, results)


if __name__ == "__main__":
    main()