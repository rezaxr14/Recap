.PHONY: up down init lint format typecheck test test-integration test-all demo eval smoke install

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
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest -m "not integration"

test-integration:
	pytest -m integration

test-all:
	pytest

demo:
	recap demo run

eval:
	recap eval run --split dev

smoke:
	python scripts/run_smoke_test.py
