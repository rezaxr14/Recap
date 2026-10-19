.PHONY: help install up down init lint format typecheck test test-integration test-all demo eval smoke

help:
	@echo "install          install the package with dev extras"
	@echo "up               start postgres and elasticsearch"
	@echo "down             stop the stack"
	@echo "init             apply migrations and initialize the search index"
	@echo "lint             ruff check + format check"
	@echo "format           apply ruff formatting and safe fixes"
	@echo "typecheck        mypy src"
	@echo "test             unit tests only (no services required)"
	@echo "test-integration integration tests (requires live services)"
	@echo "test-all         full suite with --run-integration"
	@echo "demo             run the end-to-end demo"
	@echo "eval             run the dev-split benchmark"
	@echo "smoke            run the operator smoke test"

install:
	python -m pip install -e ".[dev]"

up:
	docker compose up -d postgres elasticsearch

down:
	docker compose down

init:
	recap db migrate
	recap search init

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest tests/unit

test-integration:
	pytest tests/integration tests/e2e --run-integration

test-all:
	pytest --run-integration

demo:
	recap demo run

eval:
	recap eval run --split dev

smoke:
	python scripts/run_smoke_test.py
