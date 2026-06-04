# Common developer tasks. Run `just` (with no args) to list them.

default:
    @just --list

# Create/refresh the workspace virtual environment and lockfile.
sync:
    uv sync --all-packages

# Auto-format and auto-fix lint issues.
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Lint + format check (no changes made).
lint:
    uv run ruff check .
    uv run ruff format --check .

# Static type check.
types:
    uv run mypy

# Enforce the thermo < sim < copilot import boundaries.
imports:
    uv run lint-imports

# Run the test suite.
test:
    uv run pytest

# Everything CI runs, in order.
check: lint types imports test
