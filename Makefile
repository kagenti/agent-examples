.PHONY: lint fmt

lint:
	pre-commit run --all-files

fmt:
	ruff format .
	ruff check --fix .
