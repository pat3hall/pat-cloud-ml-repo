
# don't create files based on task names (e.g. install, test)
.PHONY: install test

# run 'test' task (instead of 'install') by default
default: test


# install development dependencies (--dev) and
#   Skip locking mechanisms and use the Pipfile (--skip-lock)
install:
	pipenv install --dev --skip-lock

# run pytest which is a test runner/framework for python
test:
	PYTHONPATH=./src pytest
