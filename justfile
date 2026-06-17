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

# Run the test suite (fast, hermetic; oracle tests excluded).
test:
    uv run pytest

# Opt-in differential tests against external references (thermo / chemicals).
oracles:
    uv run --group oracles pytest -m oracle

# Everything CI runs, in order.
check: lint types imports test

# Serve the docs site locally with live reload (http://127.0.0.1:8000).
# Social cards are CI-only, so no Cairo/Pango is needed for a local preview.
docs-serve:
    uv run --group docs mkdocs serve

# Build the docs site exactly as CI does (warnings are errors). Enables the
# social-card plugin via CI=true, which needs Cairo/Pango installed locally.
docs-build:
    CI=true uv run --group docs mkdocs build --strict
