#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <experimental/mdspan>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T, size_t ndim>
std::mdspan<T, std::dextents<size_t, ndim>>
ndarray_cast_from_py(nb::ndarray<T> &a) {

  std::array<size_t, ndim> shape;
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = a.shape(i);
  }

  return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
}

void scale_mdspan(std::mdspan<double, std::dextents<size_t, 3>> a,
                  double scale) {
  for (size_t i = 0; i < a.extent(0); ++i) {
    for (size_t j = 0; j < a.extent(1); ++j) {
      for (size_t k = 0; k < a.extent(2); ++k) {
        a[i, j, k] *= scale;
      }
    }
  }
}

void scale_ndarray(nb::ndarray<double> &a, double scale) {
  scale_mdspan(ndarray_cast_from_py<double, 3>(a), scale);
}

template <typename T> struct PauliOp {
  // As n_pauli_strings x n_qubits
  std::mdspan<int, std::dextents<size_t, 2>> pauli_strings;
  std::mdspan<T, std::dextents<size_t, 1>> coeffs;

  PauliOp(nb::ndarray<int> &ps, nb::ndarray<T> &c) {
    pauli_strings = ndarray_cast_from_py<int, 2>(ps);
    coeffs = ndarray_cast_from_py<T, 1>(c);
  }

  bool operator==(const PauliOp &other) const {
    if (pauli_strings.extent(0) != other.pauli_strings.extent(0) ||
        pauli_strings.extent(1) != other.pauli_strings.extent(1)) {
      return false;
    }

    if (coeffs.extent(0) != other.coeffs.extent(0)) {
      return false;
    }

    fmt::println("Dimensions match");

    for (size_t i = 0; i < pauli_strings.extent(0); ++i) {
      for (size_t j = 0; j < pauli_strings.extent(1); ++j) {
        if (pauli_strings[i, j] != other.pauli_strings[i, j]) {
          return false;
        }
      }
    }

    fmt::println("Pauli strings match");

    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      if (coeffs[i] != other.coeffs[i]) {
        fmt::println("{} != {}", coeffs[i], other.coeffs[i]);
        return false;
      }
    }

    fmt::println("Coeffs match");

    return true;
  }

  void scale(T scale) {
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      coeffs[i] *= scale;
    }
  }

  void print() {
    fmt::print("Coeffs[");
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      fmt::print("{}, ", coeffs[i]);
    }
    fmt::print("]\n");
  }
};

NB_MODULE(mdspan_wrapper, m) {
  //
  m.def("scale_ndarray", &scale_ndarray, "a"_a.noconvert(), "scale"_a);
  nb::class_<PauliOp<double>>(m, "PauliOp")
      .def(nb::init<nb::ndarray<int> &, nb::ndarray<double> &>())
      .def("scale", &PauliOp<double>::scale, "scale"_a)
      .def("print", &PauliOp<double>::print)
      .def("__eq__", &PauliOp<double>::operator==);
}