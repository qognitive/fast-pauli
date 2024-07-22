.PHONY: build
build:
	cmake -B build
	cmake --build build --parallel
	# TODO in general python build should internally trigger cmake, but for now
	# let's keep cmake lines here as we don't have any python build process yet
	python -m pip cache purge
	python -m pip install --upgrade pip
	python -m pip install ".[dev]"
	python -m build .

.PHONY: tests
tests:
	ctest --test-dir build
	python -m pytest fast_pauli/py/tests
	python -m pytest tests

.PHONY: clean
clean:
	rm -rf build dist

lint:
	pre-commit run --all-files -- ruff
	pre-commit run --all-files -- mypy
	pre-commit run --all-files -- codespell

# run with make -i to ignore errors and run all three formatters
format:
	pre-commit run --all-files -- ruff-format
	pre-commit run --all-files -- yamlfmt
	pre-commit run --all-files -- trailing-whitespace

.PHONY: pre-commit-setup
pre-commit-setup:
	pre-commit install
