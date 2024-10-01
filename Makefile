###############################################################################
# BUILD
###############################################################################

# Build the C++/Python package will try to reuse existing build directory
.PHONY: build
build:
	python -m pip install -e ".[dev]" --no-build-isolation

###############################################################################
# DOCS
###############################################################################

.PHONY: docs
docs: build
	python -m pip install ".[docs]"
	cd docs && doxygen Doxyfile
	sphinx-autobuild -b html docs/ docs/html --host 0.0.0.0 --port 1900

.PHONY: docs-clean
docs-clean:
	rm -rf  docs/_build/  docs/html/  docs/latex/  dosc/_static/  docs/_templates/  docs/xml/

###############################################################################
# TEST
###############################################################################

test-cpp:
	ctest --test-dir build --verbose

test-py:
	python -m pytest -v tests/fast_pauli

.PHONY: test
test: test-cpp test-py

###############################################################################
# BENCHMARK
###############################################################################

benchmark:
	python -m pytest -v tests/benchmarks --benchmark-group-by=func --benchmark-sort=fullname \
	--benchmark-columns='mean,median,min,max,stddev,iqr,outliers,ops,rounds,iterations'


###############################################################################
# STATIC ANALYSIS
###############################################################################

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



###############################################################################
# UTILITY
###############################################################################

.PHONY: clean
clean:
	rm -rf build dist

.PHONY: pre-commit-setup
pre-commit-setup:
	pre-commit install