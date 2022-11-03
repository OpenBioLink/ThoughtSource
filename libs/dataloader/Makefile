.PHONY: format lint tests integration_tests

format:
	black .
	isort .

lint:
	black . --check
	isort . --check
	flake8 . --max-line-length=140

tests:
	pytest tests/unit_tests

integration_tests:
	pytest tests/integration_tests