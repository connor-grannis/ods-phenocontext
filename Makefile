# Makefile — single source of truth for contributor commands.
# All commands run inside the uv-managed virtual environment.
# Prerequisites: uv installed (https://docs.astral.sh/uv/).

.PHONY: setup dev test lint format typecheck lock clean data train help

DATA_SRC    := data/all_training_samples.parquet
DATA_MANIFEST := data/gold/split_manifest.jsonl
DATA_OUT    := data/processed/instances.jsonl

# Default target
help:
	@echo "Available targets:"
	@echo "  setup      Install runtime deps (no dev/teacher)"
	@echo "  dev        Install all dev deps + pre-commit hooks"
	@echo "  test       Run test suite"
	@echo "  lint       Check code style with ruff"
	@echo "  format     Auto-format with ruff"
	@echo "  typecheck  Run mypy"
	@echo "  lock       Regenerate uv.lock without upgrading"
	@echo "  data       Build split manifest and process instances JSONL"
	@echo "  train      Fine-tune BioBERT (set TRAIN_OUT to override output dir)"
	@echo "  clean      Remove .venv and cache directories"

setup:
	uv sync

dev:
	uv sync --group dev --extra rules
	uv run pre-commit install

test:
	uv run --group dev pytest

lint:
	uv run --group dev ruff check .

format:
	uv run --group dev ruff format .
	uv run --group dev ruff check --fix .

typecheck:
	uv run --group dev mypy src

# Regenerate the lockfile from current pyproject.toml without upgrading deps
lock:
	uv lock

data: $(DATA_SRC)
	uv run python -m ods_phenocontext.data build-manifest \
		--input $(DATA_SRC) \
		--out $(DATA_MANIFEST) \
		--max-confirmed 15000
	uv run python -m ods_phenocontext.data process \
		--input $(DATA_SRC) \
		--manifest $(DATA_MANIFEST) \
		--out $(DATA_OUT)

TRAIN_OUT ?= checkpoints/biobert_v1

train: $(DATA_MANIFEST)
	uv run python -m ods_phenocontext.train_biobert \
		--parquet  $(DATA_SRC) \
		--manifest $(DATA_MANIFEST) \
		--out      $(TRAIN_OUT)

clean:
	rm -rf .venv .ruff_cache .mypy_cache .pytest_cache
