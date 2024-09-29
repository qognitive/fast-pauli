# `fast_pauli`
[![All push](https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml/badge.svg)](https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml)

- [`fast_pauli`](#fast_pauli)
  - [Installation](#installation)
    - [Pre-built Binaries](#pre-built-binaries)
    - [From Source](#from-source)
      - [Requirements](#requirements)
      - [Quick Start](#quick-start)
      - [Configurable Build](#configurable-build)
  - [Developer Setup](#developer-setup)
    - [Dev Requirements](#dev-requirements)
    - [Pre-commit Hooks](#pre-commit-hooks)
    - [Design Choices](#design-choices)

---
## Installation

### Pre-built Binaries
:CONSTRUCTION: TODO SETUP PYPI
```bash
pip install fast_pauli
```

### From Source

There are two strategies for building `fast_pauli` from source. One is a quick and easy method that uses all the default configuration settings. The other is a more configurable method that involves invoking `CMake`, `pip`, `pytest`, and other tools directly.

#### Requirements

- CMake >= 3.25
- C++ compiler with OpenMP and C++20 support (LLVM recommended)
- Python >= 3.10

#### Quick Start
```bash
make build
make test
```

#### Configurable Build
```bash
cmake -B build # + other cmake flags
cmake --build build --target install --parallel
ctest --test-dir build

python -m pip install -e .
pytest -v tests/fast_pauli # + other pytest flags
```
Compiled `_fast_pauli` python module gets installed into `fast_pauli` directory.


---
## Developer Setup

### Dev Requirements

```bash
python -m pip install -e ".[dev]"
```

### Pre-commit Hooks
> **You need to install the `pre-commit` hooks to ensure they run before you commit code.**

```shell
# From root project dir
pre-commit install # installs the checks as pre-commit hooks
```

### Design Choices

The C++ portion of this library relies heavily on spans and views.
These lightweight accessors are helpful and performant, but can lead to dangling spans or accessing bad memory if used improperly.
Developers should familiarize themselves with these dangers by reviewing [this post](https://hackingcpp.com/cpp/std/span.html).

