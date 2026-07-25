.PHONY: install test lint format typecheck version build

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

# Regenerate src/pydantic_ai_extensions/_version.py from the current git state.
# Also runs automatically on every build (hatch_build.py hook), so this is only
# needed when you want the dev tree's version refreshed without building.
version:
	@uv run python -c "from pathlib import Path; import hatch_build as h; info = h.write_version_file(Path('.')); print('version:', info['version'], '| commit:', info['commit'] or '(no-git)', '| branch:', info['branch'])"

# Build wheel + sdist into dist/ (the build hook bakes git provenance automatically).
build:
	uv build
	@ls -1 dist/
