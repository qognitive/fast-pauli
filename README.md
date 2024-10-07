<h1 align="center">
<img src="https://raw.githubusercontent.com/qognitive/fast-pauli/refs/heads/main/docs/logo/FP-banner.svg">
</h1>

<table>
  <tr>
    <td>Status</td>
    <td><a href="https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml"><img src="https://github.com/qognitive/fast-pauli/actions/workflows/all_push.yml/badge.svg" alt="Linux Build Status"></a>
    <a href="https://github.com/qognitive/fast-pauli/actions/workflows/pre-commit.yml"><img src="https://github.com/qognitive/fast-pauli/actions/workflows/pre-commit.yml/badge.svg" alt="Linting"></a>
    </td>
  </tr>
  <tr>
    <td>Usage</td>
    <td>
      <a href='https://qognitive-fast-pauli.readthedocs-hosted.com/en/latest/?badge=latest'>
    <img src='https://readthedocs.com/projects/qognitive-fast-pauli/badge/?version=latest' alt='Documentation Status' />
</a>
    <a href="https://github.com/qognitive/fast-pauli/tree/develop?tab=readme-ov-file#installation"><img src="https://img.shields.io/badge/Docs-Installation-blue" alt="Installation"></a>
     </td>
  </tr>
  <tr>
    <td>Package</td>
    <td>
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/fast_pauli?color=green">
    </td>
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
```bash
pip install fast_pauli
```

### From Source

There are two strategies for building `fast_pauli` from source. One is a quick and easy method that uses all the default configuration settings. The other is a more configurable method that involves invoking `CMake`, `pip`, `pytest`, and other tools directly.

#### Requirements

- CMake >= 3.25
- C++ compiler with OpenMP and C++20 support (LLVM recommended)
- Python >= 3.10

#### Quick Start (Users)
```bash
python -m pip install -e ".[dev]"
pytest -v tests/fast_pauli
```

#### Configurable Build (Developers)
```bash
cmake -B build # + other cmake flags
cmake --build build --target install --parallel
ctest --test-dir build

python -m pip install --no-build-isolation -ve.
pytest -v tests/fast_pauli # + other pytest flags
```
Compiled `_fast_pauli` python module gets installed into `fast_pauli` directory.

