.PHONY: format lint tests integration_tests

quality:
	black --line-length 140 $(check_file)
	isort $(check_file)
	flake8 $(check_file) --max-line-length=140

quality_all:
	black --line-length 140 .
	isort .
	flake8 . --max-line-length=140 --ignore=E203,W503

# defines which tests will be run by 'make tests'
# working directory is defined by pytest. 
# E.g. in vscode under settings.json in "python.testing.cwd"
tests:
	pytest tests/unit_tests -v