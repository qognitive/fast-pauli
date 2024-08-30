#include <experimental/mdspan>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

#include "fast_pauli.hpp"
#include "nanobind/nb_defs.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace fp = fast_pauli;

/*
MDSPAN Helper Functions

*/

// TODO create get shape function for mdspan

template <typename T, size_t ndim>
std::mdspan<T, std::experimental::dims<ndim>>
mdspan_like(std::mdspan<T, std::experimental::dims<ndim>> arr, T *new_data) {
  std::array<size_t, ndim> shape;
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = arr.extent(i);
  }
  return std::mdspan<T, std::experimental::dims<ndim>>(new_data, shape);
} // namespace std::mdspan

/*
Helper functions
================

These functions are used to convert between Python's ndarray and C++'s mdspan
and there are three main use cases:

1. Initialize a C++ object by passing it data from Python: here we copy the data
so the C++ objects own their data.

2. Call a C++ function with some python array as an argument: here we can just
pass a view of the data to the C++ function.

3. Return an array from a C++ function: here we copy the data so there is no
ambiguity about ownership.
*/

/**
 * @brief This function converts nb::ndarray to std::mdspan.
 *
 * NOTE: This only allows the default memory layout and accessor for the mdspan.
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Number of dimensions
 * @param a
 * @return std::mdspan<T, std::dextents<size_t, ndim>>
 */
template <typename T, size_t ndim>
std::mdspan<T, std::dextents<size_t, ndim>>
ndarray_to_mdspan(nb::ndarray<T> a) {
  // Collect shape information
  std::array<size_t, ndim> shape;
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = a.shape(i);
  }

  return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
}

template <typename T, size_t ndim, typename py_lib>
std::mdspan<T, std::dextents<size_t, ndim>>
ndarray_to_mdspan(nb::ndarray<py_lib, T> a) {
  // Collect shape information
  std::array<size_t, ndim> shape;
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = a.shape(i);
  }

  return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
}

// template <typename T, size_t ndim>
// std::mdspan<T, std::dextents<size_t, ndim>>
// ndarray_to_mdspan(nb::ndarray<T> a) {
//   // Collect shape information
//   std::array<size_t, ndim> shape;
//   for (size_t i = 0; i < ndim; ++i) {
//     shape[i] = a.shape(i);
//   }

//   return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
// }

/**
 * @brief This function copyies the data in the nb::ndarray to "raw" data in a
 * std::vector. It also return the shape informate so we can easily create an
 * mdspan of this array. We choose *not* to reuturn the mdspan directly because
 * it can create a dangling reference if the vector is moved as well (which is
 * often the case).
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Number of dimensions
 * @param a
 * @return std::pair<std::vector<T>, std::array<size_t, ndim>>
 */
template <typename T, size_t ndim>
std::pair<std::vector<T>, std::array<size_t, ndim>>
ndarray_to_raw(nb::ndarray<T> a) {
  // Shape info
  size_t size = 1;
  std::array<size_t, ndim> shape;
  for (size_t i = 0; i < a.ndim(); ++i) {
    shape[i] = a.shape(i);
    size *= a.shape(i);
  }

  // Copy the raw data
  std::vector<T> _data(size);
  // TODO I should switch this over to std::copy instead of std::memcpy
  std::memcpy(_data.data(), a.data(), size * sizeof(T));
  return std::make_pair(_data, shape);
}

template <typename T, size_t ndim>
nb::ndarray<nb::numpy, T>
owning_ndarray_like_mdspan(std::mdspan<T, std::dextents<size_t, ndim>> a) {
  // Collect shape information
  std::array<size_t, ndim> shape;
  size_t size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = a.extent(i);
    size *= a.extent(i);
  }

  // Raw data
  std::vector<T> data(size);

  // weirdness required by nanobind to properly pass ownership through
  // nb::handle, see https://github.com/wjakob/nanobind/discussions/573
  struct Temp {
    std::vector<T> data;
  };
  Temp *tmp = new Temp{data};
  nb::capsule deleter(
      tmp, [](void *data) noexcept { delete static_cast<Temp *>(data); });

  // TODO can we do this without speciyfin that it's a numpy array?
  return nb::ndarray<nb::numpy, T>(
      /*data*/ tmp->data.data(),
      /*ndim*/ shape.size(),
      /*shape */ shape.data(),
      /*deleter*/ deleter);
}

/*
Python Bindings for PauliOp
*/

NB_MODULE(fppy, m) {
  // TODO init default threading behaviour for the module
  // TODO give up GIL when calling into long-running C++ code
  using float_type = double;

  nb::class_<fp::Pauli>(m, "Pauli")
      .def(nb::init<>())
      .def(nb::init<int const>(), "code"_a)
      .def(nb::init<char const>(), "symbol"_a)
      .def("to_tensor", &fp::Pauli::to_tensor<float_type>)
      .def("multiply", [](fp::Pauli const &self,
                          fp::Pauli const &rhs) { return self * rhs; })
      .def("__str__",
           [](fp::Pauli const &self) { return fmt::format("{}", self); });

  nb::class_<fp::PauliString>(m, "PauliString")
      .def(nb::init<>())
      .def(nb::init<std::string const &>(), "string"_a)
      .def("__str__",
           [](fp::PauliString const &self) { return fmt::format("{}", self); })
      .def("apply",

           [](fp::PauliString const &self,
              nb::ndarray<std::complex<float_type>> states,
              std::complex<float_type> c = std::complex<float_type>{1.0}) {
             auto states_mdspan =
                 ndarray_to_mdspan<std::complex<float_type>, 2>(states);
             nb::ndarray<nb::numpy, std::complex<float_type>> new_states =
                 owning_ndarray_like_mdspan<std::complex<float_type>, 2>(
                     states_mdspan);
             auto new_states_mdspan =
                 ndarray_to_mdspan<std::complex<float_type>, 2>(new_states);

             self.apply_batch<float_type>(new_states_mdspan, states_mdspan, c);

             //  nb::ndarray<nb::numpy, std::complex<float_type>> result =
             //      nb::ndarray<nb::numpy, std::complex<float_type>>(
             //          /*data*/ new_states.data(),
             //          //  /*ndim*/ static_cast<size_t>(new_states.ndim()),
             //          /*shape */ {new_states.shape(0), new_states.shape(1)},
             //          /*deleter*/ *new_states.handle());
             //  return result;
             return new_states;
           })
      //
      ;
  ;
}