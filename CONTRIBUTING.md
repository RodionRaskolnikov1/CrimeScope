# Contributing to CrimeScope

Thanks for your interest in contributing!

## Setup

```bash
git clone https://github.com/RodionRaskolnikov1/CrimeScope
cd crimescope
uv sync
cp .env.example .env
# Add your Gemini API key to .env
```

## Running the pipeline

```bash
uv run python main.py
```

## Running the API

```bash
uv run uvicorn crimescope.api.main:app --reload --port 8000
```

## Code style

This project uses Ruff for linting and formatting.

```bash
# Check
uv run ruff check .

# Fix
uv run ruff check . --fix

# Format
uv run ruff format .
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Project structure

See [README.md](README.md) for the full project structure and architecture overview.

## Pull request guidelines

- Keep PRs focused on a single change
- Add tests for new functionality
- Run `ruff check .` before submitting
- Update README if you change any API endpoints or pipeline steps