PYTHON = uv run python

ARGS = \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json

.PHONY: install run debug clean lint lint-strict

install:

	@uv sync
	@ uv add --editable ./llm_sdk
run:
	@$(PYTHON) -m src.main $(ARGS)

debug:
	@uv run python -m pdb -m src.main $(ARGS)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -exec rm -f {} +

lint:
	@uv run flake8 src/ --exclude .venv,llm_sdk
	@uv run python -m mypy src/ \
		--exclude .venv \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs