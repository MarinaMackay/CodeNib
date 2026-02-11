.PHONY: install scip dev test

install:
	pip install -e .

scip:
	pip install -e ".[scip]"
	./setup-scip.sh

dev:
	pip install -e ".[dev,test]"

test:
	pytest
