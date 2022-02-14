TEST_PATH=./tests
MODULES_PATH=./ssi main.py

.PHONY: format lint test commit

format:
	isort $(MODULES_PATH) $(TEST_PATH)
	black $(MODULES_PATH) $(TEST_PATH)

lint:
	isort -c $(MODULES_PATH) $(TEST_PATH)
	black --check $(MODULES_PATH) $(TEST_PATH)
	mypy $(MODULES_PATH) $(TEST_PATH)

test:
	python3 -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)

