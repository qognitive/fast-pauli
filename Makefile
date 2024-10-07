#############################################################################
# This code is part of Fast Pauli.
#
# (C) Copyright Qognitive Inc 2024.
#
# This code is licensed under the BSD 2-Clause License. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#############################################################################


###############################################################################
# BUILD
###############################################################################

# Build the C++/Python package from scratch (ignore existing build dir)
.PHONY: build
build:
	python -m pip install -e ".[dev]"

# Build the C++/Python package will try to reuse existing build directory, will
# not always work on a fresh checkout bc it requires prereqs to be installed.
# See https://scikit-build-core.readthedocs.io/en/latest/configuration.html#editable-installs
.PHONY: rebuild
rebuild:
	python -m pip install -e ".[dev]" --no-build-isolation


###############################################################################
# DOCS
###############################################################################

.PHONY: docs
docs:
	python -m pip install -q ".[docs]"
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

benchmark-qiskit-adv:
	EXTRA_BENCHMARKS=true pytest -vs tests/benchmarks/test_qiskit_adv.py \
		--benchmark-columns='mean,stddev,rounds' \
		--benchmark-json=qiskit_adv.json
	py.test-benchmark compare ./qiskit_adv.json \
		--group-by=func \
		--csv=docs/benchmark_results/qiskit_adv.csv
	python tests/benchmarks/process_qiskit_benchmarks.py docs/benchmark_results/qiskit_adv.csv

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
