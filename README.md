# `fast_pauli`
[![Build](https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml/badge.svg)](https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml) | [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

[Installation](#installation) | [Contributing](CONTRIBUTING.md)

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

python -m pip install -e ".[dev]"
pytest -v tests/fast_pauli # + other pytest flags
```
Compiled `_fast_pauli` python module gets installed into `fast_pauli` directory.

