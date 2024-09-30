# `fast_pauli`


<table>
  <tr>
    <td>Status</td>
    <td><a href="https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml"><img src="https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml/badge.svg" alt="Build Status"></a>
    <a href="https://github.com/qognitive/fast-pauli/actions/workflows/pre-commit.yml"><img src="https://github.com/qognitive/fast-pauli/actions/workflows/pre-commit.yml/badge.svg" alt="Linting"></a>
    </td>
  </tr>
  <tr>
    <td>Usage</td>
    <td>TODO RTD INFO
    <a href="https://github.com/qognitive/fast-pauli/tree/develop?tab=readme-ov-file#installation"><img src="https://img.shields.io/badge/Docs-Installation-blue" alt="Installation"></a>
     </td>
  </tr>
  <tr>
    <td>Package</td>
    <td>TODO PyPI INFO</td>
  </tr>
  <tr>
    <td>Legal</td>
    <td>
    <a href="https://opensource.org/licenses/BSD-2-Clause"><img src="https://img.shields.io/badge/License-BSD_2--Clause-orange.svg" alt="License"></a>
    <a href="https://github.com/qognitive/fast-pauli/blob/develop/CODE_OF_CONDUCT.md"><img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"></a>
    </td>
  </tr>
</table>


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

