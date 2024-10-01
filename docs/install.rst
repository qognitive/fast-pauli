Installation
============

Build From Source
-----------------

.. code-block:: bash

    cmake -B build -DCMAKE_CXX_COMPILER=clang++
    cmake --build build --target install --parallel
    ctest --test-dir build

    python -m pip install -e .