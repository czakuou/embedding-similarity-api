.PHONY: help install sync test test-unit test-integration lint format typecheck \
	pre-commit audit clean run-api run-worker compose-up compose-down compose-logs

help:
	@echo "Setup:"
	@echo "  install          Sync all dependency groups (dev + worker) and install pre-commit hooks"
	@echo "  sync             Sync dependencies only (no pre-commit hook install)"
	@echo ""
	@echo "Tests:"
	@echo "  test             Run the full suite (unit + integration)"
	@echo "  test-unit        Run unit tests only (-m 'not behavior')"
	@echo "  test-integration Run BEHAVIOR.md integration tests only (-m behavior, real model + Redis)"
	@echo ""
	@echo "Quality:"
	@echo "  lint             Run ruff check"
	@echo "  format           Run ruff format"
	@echo "  typecheck        Run mypy --strict"
	@echo "  pre-commit       Run all pre-commit hooks against all files"
	@echo "  audit            Export locked deps and run pip-audit"
	@echo ""
	@echo "Local services:"
	@echo "  run-api          Run the API with live reload against local Redis"
	@echo "  run-worker       Run one worker process against local Redis"
	@echo "  compose-up       Start api, worker, and Redis via docker compose"
	@echo "  compose-down     Stop and remove the docker compose stack"
	@echo "  compose-logs     Follow logs from the docker compose stack"
	@echo ""
	@echo "  clean            Remove caches and build artifacts"

# testcontainers can't talk to colima's Docker socket through Ryuk locally;
# see project_testcontainers_colima_ryuk memory / README for why.
export TESTCONTAINERS_RYUK_DISABLED=true

install:
	uv sync --all-groups
	uv run pre-commit install

sync:
	uv sync --all-groups

test:
	uv run pytest

test-unit:
	uv run pytest -m "not behavior"

test-integration:
	uv run pytest -m behavior

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/similarity tests

pre-commit:
	uv run pre-commit run --all-files

audit:
	uv export --format requirements.txt --all-groups --no-emit-project -o requirements-audit.txt
	uv run pip-audit -r requirements-audit.txt

# Local run targets assume Redis is reachable at localhost:6379 (e.g. `docker
# run -p 6379:6379 redis:7.4-alpine`), not the `redis` compose hostname.
run-api:
	REDIS_URL=redis://localhost:6379/0 \
	uv run uvicorn similarity.api:create_app --factory --reload --port 8000

run-worker:
	REDIS_URL=redis://localhost:6379/0 \
	MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
	MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
	uv run python -m similarity.worker.main

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache requirements-audit.txt
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
