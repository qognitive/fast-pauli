.. fast_pauli documentation master file, created by
   sphinx-quickstart on Fri Sep 20 16:32:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: https://raw.githubusercontent.com/qognitive/fast-pauli/refs/heads/feature/logo/docs/logo/FP-banner.svg
   :alt: Fast Pauli logo

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   benchmarks
   python_api
   cpp_api


Installation
============
Welcome to ``fast_pauli``! This package provides fast and efficient implementations of pauli operators and strings,
with a focus on performance and usability. In order to get started, we'll need to install the package and its dependencies.

Here are a few options for installing ``fast_pauli``:

Install the Latest Release
--------------------------
.. code-block:: bash

   pip install fast_pauli

Build from Source (Default Config)
-----------------------------------------
.. code-block:: bash

   git clone git@github.com:qognitive/fast-pauli.git
   cd fast_pauli
   python -m pip install -e ".[dev]"


Build From Source (Custom Config)
---------------------------------
.. code-block:: bash

   git clone git@github.com:qognitive/fast-pauli.git
   cd fast-pauli

   # Configure/Compile the project
   cmake -B build -G Ninja # <...custom cmake flags...>
   cmake --build build --target install --parallel
   ctest --test-dir build

   # Install the python package
   python -m pip install -e ".[dev]" --no-build-isolation

Verify / Test Build
-------------------

.. code-block:: bash

   pytest -v tests/fast_pauli # + other pytest flags
