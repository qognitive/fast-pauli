###############################################################################
# BUILD
###############################################################################

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

###############################################################################
# DOCS
###############################################################################

.PHONY: docs
docs: build-cpp
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

benchmark-for-release:
	python -m pytest -v tests/benchmarks \
	    -k "apply_batch_n_qubits_n_states" \
		--benchmark-group-by=func \
		--benchmark-sort=fullname \
		--benchmark-json=benchmark_results.json
	py.test-benchmark compare \
		./benchmark_results.json \
		--group-by=func \
		--csv=benchmark_results.csv
	python tests/benchmarks/process_benchmark_data.py

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