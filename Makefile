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

lint-check:
	ruff check ./fast_pauli/py ./tests && \
	mypy ./fast_pauli/py ./tests

lint-fix:
	ruff check --fix ./fast_pauli/py ./tests

lint-write:
	ruff format ./fast_pauli/py ./tests
