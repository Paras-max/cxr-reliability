.PHONY: install test lint typecheck api dashboard docker-build docker-up

install:
	pip install -r requirements-dev.txt && pip install -e .

test:
	pytest

lint:
	ruff check src tests scripts

typecheck:
	mypy src

api:
	uvicorn cxr_reliability.api.main:create_app --factory --host $${CXR_API_HOST:-0.0.0.0} --port $${CXR_API_PORT:-8000}

dashboard:
	streamlit run src/cxr_reliability/dashboard/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up
