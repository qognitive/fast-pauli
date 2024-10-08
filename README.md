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
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/fast_pauli?color=4c1">
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
## Introduction

Welcome to `fast-pauli` from [Qognitive](https://www.qognitive.io/), an open-source Python / C++ library for optimized operations on Pauli matrices and Pauli strings,
inspired by [PauliComposer](https://arxiv.org/abs/2301.00560) paper.
`fast-pauli` aims to provide a fast and efficient alternative to existing libraries for working with Pauli matrices and strings,
with a focus on performance and usability.
For example, `fast-pauli` provides optimized functions to apply Pauli strings and operators to a batch of states rather than just a single state vector.
See our [benchmarks](https://qognitive-fast-pauli.readthedocs-hosted.com/en/latest/benchmarks.html) for more details about how `fast-pauli` can speed up certain functions compared to Qiskit.

Our [Getting Started](https://qognitive-fast-pauli.readthedocs-hosted.com/en/latest/getting_started.html) guide offers an introduction to some of the core functionality in `fast-pauli`.

## Installation

### Pre-built Binaries
```bash
pip install fast_pauli
```

### From Source

There are two strategies for building `fast_pauli` from source. One is a quick and easy method that uses all the default configuration settings. The other is a more configurable method that involves invoking `CMake`, `pip`, `pytest`, and other tools directly.

#### Requirements

- [CMake](https://pypi.org/project/cmake/) >= 3.25
- [Ninja](https://pypi.org/project/ninja/) >= 1.11
- C++ compiler with OpenMP and C++20 support ([LLVM](https://apt.llvm.org/) recommended)
- [Python](https://www.python.org/downloads/) >= 3.10

#### Quick Start (Users)
```bash
python -m pip install -e ".[dev]"
pytest -v tests/fast_pauli
```

#### Configurable Build (Developers)
```bash
python -m pip install --upgrade pip
python -m pip install scikit-build-core
python -m pip install --no-build-isolation -ve ".[dev]" -C cmake.args="-DCMAKE_CXX_COMPILER=<compiler> + <other cmake flags>"
pytest -v tests/fast_pauli # + other pytest flags
```
Compiled `_fast_pauli` python module gets installed into `fast_pauli` directory.

