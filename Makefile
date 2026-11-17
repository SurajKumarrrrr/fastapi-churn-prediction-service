.PHONY: install train test lint run smoke docker-build docker-run clean

install:
	pip install -r requirements-dev.txt -e .

train:
	python -m churn_service.train --out artifacts --model-card MODEL_CARD.md

test:
	pytest -q

lint:
	ruff check .

run: artifacts/model.joblib
	uvicorn churn_service.api:app --host 0.0.0.0 --port 8000

artifacts/model.joblib:
	python -m churn_service.train --out artifacts

smoke:
	python scripts/smoke_test.py --url http://localhost:8000 --requests 200

docker-build:
	docker build -t churn-service .

docker-run:
	docker run --rm -p 8000:8000 churn-service

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
