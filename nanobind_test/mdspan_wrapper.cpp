#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <experimental/mdspan>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <iostream>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

template <typename T, size_t ndim>
std::pair<std::vector<T>, std::array<size_t, ndim>>
cast_ndarray_to_blob(nb::ndarray<T> &a) {
  // Shape info
  size_t size = 1;
  std::array<size_t, ndim> shape;
  for (size_t i = 0; i < a.ndim(); ++i) {
    shape[i] = a.shape(i);
    size *= a.shape(i);
  }

  // Copy the raw data
  std::vector<T> _data(size);
  std::memcpy(_data.data(), a.data(), size * sizeof(T));
  return std::make_pair(_data, shape);
}
template <typename T> struct PauliOp {
  // As n_pauli_strings x n_qubits
  std::mdspan<int, std::dextents<size_t, 2>> pauli_strings;
  std::mdspan<T, std::dextents<size_t, 1>> coeffs;
  std::vector<T> _coeffs;

  PauliOp(nb::ndarray<int> &ps, nb::ndarray<T> &c) {
    pauli_strings = ndarray_cast_from_py<int, 2>(ps);
    std::array<size_t, 1> shape;
    std::tie(_coeffs, shape) = cast_ndarray_to_blob<T, 1>(c);
    coeffs = std::mdspan<T, std::dextents<size_t, 1>>(_coeffs.data(), shape);
  }

  bool operator==(const PauliOp &other) const {
    if (pauli_strings.extent(0) != other.pauli_strings.extent(0) ||
        pauli_strings.extent(1) != other.pauli_strings.extent(1)) {
      return false;
    }

    if (coeffs.extent(0) != other.coeffs.extent(0)) {
      return false;
    }
    // fmt::println("Dimensions match");

    for (size_t i = 0; i < pauli_strings.extent(0); ++i) {
      for (size_t j = 0; j < pauli_strings.extent(1); ++j) {
        if (pauli_strings[i, j] != other.pauli_strings[i, j]) {
          return false;
        }
      }
    }

    // fmt::println("Pauli strings match");
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      if (coeffs[i] != other.coeffs[i]) {
        return false;
      }
    }

    // fmt::println("Coeffs match");

    return true;
  }

  void scale(T scale) {
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      coeffs[i] *= scale;
    }
  }

  void multiply_coeff(std::mdspan<T, std::dextents<size_t, 1>> other) {
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      coeffs[i] *= other[i];
    }
  }

  void print() {
    fmt::print("Coeffs[");
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      fmt::print("{}, ", coeffs[i]);
    }
    fmt::print("]\n");
  }

  void return_coeffs(std::mdspan<T, std::dextents<size_t, 1>> &out) {
    for (size_t i = 0; i < coeffs.extent(0); ++i) {
      out[i] = coeffs[i];
    }
  }
};

NB_MODULE(mdspan_wrapper, m) {
  //
  nb::class_<PauliOp<double>>(m, "PauliOp")
      .def(nb::init<nb::ndarray<int> &, nb::ndarray<double> &>())
      .def("scale", &PauliOp<double>::scale, "scale"_a)
      .def("print", &PauliOp<double>::print)
      .def("__eq__", &PauliOp<double>::operator==)
      .def("multiply_coeff",
           [](PauliOp<double> &op, nb::ndarray<double> &c) {
             op.multiply_coeff(ndarray_cast_from_py<double, 1>(c));
           })
      .def("return_coeffs",
           [](PauliOp<double> &op, nb::ndarray<double> &out) {
             auto out_mdspan = ndarray_cast_from_py<double, 1>(out);
             op.return_coeffs(out_mdspan);
           })
      .def("return_coeffs_owning", [](PauliOp<double> &op) {
        struct Temp {
          std::vector<double> data;
        };

        Temp *tmp = new Temp{op._coeffs};

        fmt::println("copied data: [{}]", fmt::join(tmp->data, ", "));
        std::cout << std::flush;

        nb::capsule deleter(
            tmp, [](void *data) noexcept { delete static_cast<Temp *>(data); });

        return nb::ndarray<nb::numpy, double>(
            /*data*/ tmp->data.data(),
            /*shape */ {tmp->data.size()},
            /*deleter*/ deleter);
      });

  m.def("return_coeffs",
        [](size_t n) {
          std::vector<double> data(n);
          for (size_t i = 0; i < n; ++i) {
            data[i] = i;
          }

          struct Temp {
            std::vector<double> data;
          };

          Temp *tmp = new Temp{data};

          nb::capsule deleter(tmp, [](void *data) noexcept {
            delete static_cast<Temp *>(data);
          });

          return nb::ndarray<nb::numpy, double>(
              /*data*/ tmp->data.data(),
              /*shape */ {tmp->data.size()},
              /*deleter*/ deleter);

          // nb::ndarray<nb::numpy> out{data.data(), {data.size()}, deleter};

          // nb::object res = nb::cast(out, nb::rv_policy::copy);
          // return res;
        }

        /**/
  );
}
