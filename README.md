# Summary

## Requirements

- CMake >= 3.25
- C++ compiler with OpenMP and C++20 support (LLVM recommended)
  - Tested Compilers GCC@12, LLVM@17, LLVM@18
- Python >= 3.10

## Tested Configurations

| OS      | Compiler | Python |
| ------- | -------- | ------ |
| Ubuntu 22.04 | GCC 12   | 3.10   |
| Ubuntu 22.04 | LLVM 17  | 3.10   |

## Developer Setup

### Dev Requirements

- C/C++: `clang-format` defaults
- Python: `ruff` lint/format, [`pre-commit`](https://pre-commit.com/)
- CMake: `cmake-format`

> **You need to install the `pre-commit` hooks to ensure they run before you commit code.**

```shell
# From root project dir
pre-commit install # installs the checks as pre-commit hooks
```

## Build and Test

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build --parallel
cmake --install build
ctest --test-dir build
```
Compiled `_fast_pauli` python module gets installed into `fast_pauli` directory.

## Design Choices

The C++ portion of this library relies heavily on spans and views.
These lightweight accessors are helpful and performant, but can lead to dangling spans or accessing bad memory if used improperly.
Developers should familiarize themselves with these dangers by reviewing [this post](https://hackingcpp.com/cpp/std/span.html).

