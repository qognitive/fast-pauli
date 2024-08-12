build-cpp:
	cmake -B build
	cmake --build build --parallel
	cmake --install build
# TODO in general python build should internally trigger cmake, but for now
# let's keep cmake lines here as we don't have any python build process yet

build-py:
	python -m pip cache purge
	python -m pip install --upgrade pip
	python -m pip install ".[dev]"
	python -m build .

.PHONY: build
build: build-cpp build-py

test-cpp:
	ctest --test-dir build

test-py:
	python -m pytest -v tests

.PHONY: test
test: test-cpp test-py

.PHONY: benchmark
benchmark:
	pytest -v benchmarks --benchmark-group-by=func --benchmark-sort=fullname \
	--benchmark-columns='mean,median,min,max,stddev,iqr,outliers,ops,rounds,iterations'

.PHONY: clean
clean:
	rm -rf build dist

lint:
	pre-commit run --all-files -- ruff
	pre-commit run --all-files -- mypy
	pre-commit run --all-files -- codespell

# run with make -i to ignore errors and run all three formatters
format:
	pre-commit run --all-files -- cmake-format
	pre-commit run --all-files -- clang-format
	pre-commit run --all-files -- ruff-format
	pre-commit run --all-files -- yamlfmt
	pre-commit run --all-files -- trailing-whitespace

.PHONY: pre-commit-setup
pre-commit-setup:
	pre-commit install
