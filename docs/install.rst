============
Installation
============

Welcome to ``fast_pauli``! This package provides fast and efficient implementations of pauli operators and strings,
with a focus on performance and usability. In order to get started, we'll need to install the package and its dependencies.

Here are a few options for installing ``fast_pauli``:

Install using ``pip``
------------------
.. code-block:: bash

    pip install fast_pauli

Install from source with Python
-------------------------------
.. code-block:: bash

    git clone git@github.com:qognitive/fast-pauli.git
    cd fast_pauli
    python -m pip install -e ".[dev]"


Build From Source with CMake
----------------------------
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