PYTHON = python3

ARGS = \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json


.PHONY: install run debug clean lint lint-strict

install:
	@uv sync

run:
	@$(PYTHON) -m src.main $(ARGS)

debug:
	@$(PYTHON) -m pdb -m src.main $(ARGS)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc"       -exec rm -f  {} +

lint:
	@flake8 .
	@$(PYTHON) -m mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
