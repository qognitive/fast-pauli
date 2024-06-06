# Summary

# TODOs
- [ ] Figure out the tranpose non-sense or support both
- [ ] Clean up `apply_batch` we shouldn't need to pass a coeff
- [ ] Add docstrings
  - [X] Pauli
  - [X] PauliString
  - [ ] PauliOp
  - [ ] SummedPauliOp
- [ ] Clean up tests
- [X] Clean up test utils
- [X] Add type aliases and factory functions to utils for fast_pauli
- [X] Seach the names and make sure we don't have any overlap with other projects
- [ ] Build out pauli decomposer
- [X] Remove the weights argument and rename to data
- [X] Add namespace
- [ ] Add apply method to SummedPauliOp that takes precomputed weighted data
- [ ] Writeup for docs
- [ ] Add pybind11 interface and python examples
- [ ] Change functions names over to default to parallel impl and use `_serial` for the serial implementation
- [ ] Migrate `PauliOp` and `SummedPauliOp` to only store mdspans rather than copies of the data itself

## Requirements

- CMake >= 3.20
- C++ compiler with OpenMP and C++20 support (LLVM recommended)
  - Tested Compilers GCC@12, LLVM@17, LLVM@18
- Python >= 3.10


## Build and Test

```bash
cmake -B build -DCMAKE_CXX_COMPILER=<your_favorite_c++_compiler>
cmake --build build
ctest --test-dir build
```

## Design Choices

The C++ portion of this library relies heavily on spans and views.
These lightweight accessors are helpful and performant, but can lead to dangling spans or accessing bad memory if used improperly.
Developers should familiarize themselves with these dangers by reviewing [this post](https://hackingcpp.com/cpp/std/span.html).
