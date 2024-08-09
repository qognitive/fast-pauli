# `nb::ndarray` :handshake: `std::mdspan`

How to connect `nanobind`'s `nb::ndarray` and `std::mdspan`.

Properties of `nb::ndarray`
- work with numpy, torch, tensorflow, jax, cupy, and anything that supports DLPack
- Supports zero-copy exchange using two protocols: 1) buffer protocol and 2) DLPack
- Can return data that is owned by a Python object

## Useful guides from `nanobind`
- https://github.com/wjakob/nanobind/blob/b1531b9397c448c6b784520a4f052608f28a5e8d/include/nanobind/eigen/dense.h#L93C7-L93C24
- https://github.com/wjakob/nanobind/blob/b1531b9397c448c6b784520a4f052608f28a5e8d/include/nanobind/eigen/dense.h#L124
- https://github.com/wjakob/nanobind/blob/b1531b9397c448c6b784520a4f052608f28a5e8d/include/nanobind/eigen/dense.h#L242-L245