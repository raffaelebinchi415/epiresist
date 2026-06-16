.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "install  install runtime deps"
	@echo "dev      install dev deps + package (editable)"
	@echo "lint     ruff + black --check + isort --check"
	@echo "type     mypy"
	@echo "test     pytest"
	@echo "smoke    triturate + compound + assay on the smoke formula"
	@echo "docker   build the container image"
	@echo "clean    remove caches and build artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy

test:
	pytest -q

smoke:
	python -m epiresist.counter triturate --formula _smoke
	python -m epiresist.counter compound --formula _smoke
	python -m epiresist.counter assay --formula _smoke

docker:
	docker build -t epiresist:0.1.0 .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache **/__pycache__ *.egg-info build dist runs
