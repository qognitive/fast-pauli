.. fast_pauli documentation master file, created by
   sphinx-quickstart on Fri Sep 20 16:32:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: https://raw.githubusercontent.com/qognitive/fast-pauli/refs/heads/main/docs/logo/FP-banner.svg
   :alt: Fast Pauli logo

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   benchmarks
   python_api
   cpp_api

Introduction
============
Welcome to :code:`fast-pauli` from `Qognitive <https://www.qognitive.io/>`_, an open-source Python / C++ library for optimized operations on Pauli matrices and Pauli strings,
inspired by `PauliComposer <https://arxiv.org/abs/2301.00560>`_ paper.
:code:`fast-pauli` aims to provide a fast and efficient alternative to existing libraries for working with Pauli matrices and strings,
with a focus on performance and usability.
For example, :code:`fast-pauli` provides optimized functions to apply Pauli strings and operators to a batch of states rather than just a single state vector.
See our :doc:`getting_started` guide for an introduction to some of the core functionality in :code:`fast-pauli` and our :doc:`benchmarks` for more details about how :code:`fast-pauli` can speed up certain functions compared to Qiskit.

Installation
============
In order to get started, we'll need to install the package and its dependencies.

Requirements
------------
- `CMake <https://pypi.org/project/cmake/>`_ >= 3.25
- `Ninja <https://pypi.org/project/ninja/>`_ >= 1.11
- C++ compiler with OpenMP and C++20 support (`LLVM <https://apt.llvm.org/>`_ recommended)
- `Python <https://www.python.org/downloads/>`_ >= 3.10
- `scikit-build-core <https://pypi.org/project/scikit-build-core/>`_ (ONLY for building from source with custom configuration)

In the following subsections, we describe several options for installing ``fast_pauli``.

Install the Latest Release
--------------------------
.. code-block:: bash

   pip install fast_pauli

Build from Source (Linux)
-----------------------------------------
.. code-block:: bash

   git clone git@github.com:qognitive/fast-pauli.git
   cd fast_pauli
   python -m pip install -e ".[dev]"

Build from Source (MacOS)
-----------------------------------------
.. code-block:: bash

   git clone git@github.com:qognitive/fast-pauli.git
   cd fast_pauli
   python -m pip install --upgrade pip
   python -m pip install scikit-build-core
   brew install llvm
   pip install -e . -C cmake.args="-DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++;-DCMAKE_CXX_FLAGS='-stdlib=libc++ -fexperimental-library'"

Build from Source (Custom Config)
---------------------------------
.. code-block:: bash

   git clone git@github.com:qognitive/fast-pauli.git
   cd fast-pauli
   python -m pip install --upgrade pip
   python -m pip install scikit-build-core
   python -m pip install --no-build-isolation -ve ".[dev]" -C cmake.args="-DCMAKE_CXX_COMPILER=<compiler> + <other cmake flags>"

Verify / Test Build
-------------------

.. code-block:: bash

   pytest -v tests/fast_pauli # + other pytest flags
