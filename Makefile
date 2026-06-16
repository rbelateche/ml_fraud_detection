# Developer shortcuts. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install data eda baseline bakeoff test lint fmt clean docker-build docker-run

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev + ml extras (editable)
	$(PY) -m pip install -e '.[ml,dev]'

data: ## Build/refresh the dataset
	fraud-data

eda: ## Run EDA profiling (writes figures)
	fraud-eda

baseline: ## Train Phase 0 baselines (dummy + logistic)
	fraud-baseline

bakeoff: ## Run the Phase 0.5 model tournament
	fraud-bakeoff

test: ## Run the test suite
	pytest -q

lint: ## Lint with ruff
	ruff check src tests

fmt: ## Auto-format with ruff
	ruff check --fix src tests

docker-build: ## Build the Docker image
	docker compose build

docker-run: ## Run the full pipeline in Docker
	docker compose run --rm pipeline

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/processed/*.parquet reports/figures/*.png
