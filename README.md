# Summary

# TODOs
- [ ] Add docstrings
  - [X] Pauli
  - [X] PauliString
  - [ ] PauliOp
  - [ ] SummedPauliOp
- [ ] Figure out the transpose non-sense or support both (some functions take the transpose of the states and others don't)
- [ ] Clean up `apply_batch` we shouldn't need to pass a coeff
- [ ] Clean up tests
- [ ] Add apply method to SummedPauliOp that takes precomputed weighted data
- [ ] Add pybind11 interface and python examples
- [ ] Change functions that may run in parallel to take [`std::execution_policy`](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)
- [ ] Possibly add levels to methods like BLAS to group methods by scaling
- [ ] Migrate `PauliOp` and `SummedPauliOp` to only store mdspans rather than copies of the data itself

## Requirements

- CMake >= 3.25
- C++ compiler with OpenMP and C++20 support (LLVM recommended)
  - Tested Compilers GCC@12, LLVM@17, LLVM@18
- Python >= 3.10

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
cmake --build build
cmake --install build
ctest --test-dir build
```

## Design Choices

The C++ portion of this library relies heavily on spans and views.
These lightweight accessors are helpful and performant, but can lead to dangling spans or accessing bad memory if used improperly.
Developers should familiarize themselves with these dangers by reviewing [this post](https://hackingcpp.com/cpp/std/span.html).

