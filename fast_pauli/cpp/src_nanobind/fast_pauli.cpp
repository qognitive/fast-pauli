#include <iostream>
#include <numeric>
#include <ranges>

#include <experimental/mdspan>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "__pauli.hpp"
#include "__summed_pauli_op.hpp"
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

  if (a.ndim() != ndim) {
    throw std::invalid_argument(fmt::format(
        "ndarray_to_mdspan: expected {} dimensions, got {}", ndim, a.ndim()));
  }

  // Collect shape information
  std::array<size_t, ndim> shape;
  std::array<size_t, ndim> strides;

  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = a.shape(i);
    strides[i] = a.stride(i);
  }

  // Check if the strides are C-style (row-major)
  std::array<size_t, ndim> expected_strides;

  // Calculate the expected strides using a prefix product (reversed)
  std::exclusive_scan(shape.rbegin(), shape.rend(), expected_strides.rbegin(),
                      1, std::multiplies<>{});
  if (!std::ranges::equal(strides, expected_strides)) {
    throw std::invalid_argument("nb::ndarray MUST have C-style strides.");
  }

  return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
}

// TODO this second function is really sloppy, we should be able to do much
// better with some template metaprogramming
template <typename T, size_t ndim, typename ndarray_framework>
std::mdspan<T, std::dextents<size_t, ndim>>
ndarray_to_mdspan(nb::ndarray<ndarray_framework, T> a) {

  if (a.ndim() != ndim) {
    throw std::invalid_argument(fmt::format(
        "ndarray_to_mdspan: expected {} dimensions, got {}", ndim, a.ndim()));
  }

  // Collect shape information
  std::array<size_t, ndim> shape;
  std::array<size_t, ndim> strides;

  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = a.shape(i);
    strides[i] = a.stride(i);
  }

  // Check if the strides are C-style (row-major)
  std::array<size_t, ndim> expected_strides;

  // Calculate the expected strides using a prefix product (reversed)
  std::exclusive_scan(shape.rbegin(), shape.rend(), expected_strides.rbegin(),
                      1, std::multiplies<>{});
  if (!std::ranges::equal(strides, expected_strides)) {
    throw std::invalid_argument("nb::ndarray MUST have C-style strides.");
  }

  return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
}

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
  std::memcpy(_data.data(), a.data(), size * sizeof(T));
  return std::make_pair(std::move(_data), shape);
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
  fmt::println("shape [{}] and size {}", fmt::join(shape, ", "), size);

  // std::vector<T> data(size);

  // weirdness required by nanobind to properly pass ownership through
  // nb::handle, see https://github.com/wjakob/nanobind/discussions/573
  struct Temp {
    std::vector<T> data;
  };

  // Raw data
  // Temp *tmp = new Temp{std::vector<T>(size)};
  Temp *tmp = new Temp{std::vector<T>(size)};

  nb::capsule deleter(tmp, [](void *p) noexcept {
    fmt::println("deleting data in nb::capsule with shape");
    std::cout << std::flush;
    delete static_cast<Temp *>(p);
    // delete (Temp *)p;
  });

  // TODO can we do this without speciyfin that it's a numpy array?
  return nb::ndarray<nb::numpy, T>(
      /*data*/ tmp->data.data(),
      /*ndim*/ shape.size(),
      /*shape */ shape.data(),
      /*deleter*/ deleter);
}

template <typename T, size_t ndim>
nb::ndarray<nb::numpy, T>
owning_ndarray_from_shape(std::array<size_t, ndim> shape) {
  // Collect shape information
  size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());

  // Raw data

  // weirdness required by nanobind to properly pass ownership through
  // nb::handle, see https://github.com/wjakob/nanobind/discussions/573
  struct Temp {
    std::vector<T> data;
  };
  Temp *tmp = new Temp{std::vector<T>(size)};

  nb::capsule deleter(tmp, [](void *data) noexcept {
    fmt::println("deleting data in nb::capsule with shape");
    std::cout << std::flush;
    delete static_cast<Temp *>(data);
  });

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
  // TODO init default threading behavior for the module
  // TODO give up GIL when calling into long-running C++ code
  using float_type = double;
  using cfloat_t = std::complex<float_type>;

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
      // Constructors
      .def(nb::init<>())
      .def(nb::init<std::string const &>(), "string"_a)
      .def("__str__",
           [](fp::PauliString const &self) { return fmt::format("{}", self); })

      // Properties
      .def_prop_ro("n_qubits", &fp::PauliString::n_qubits)
      .def_prop_ro("dim", &fp::PauliString::dims)
      .def_prop_ro("weight",
                   [](fp::PauliString const &self) { return self.weight; })
      // Methods
      .def(
          "apply",
          [](fp::PauliString const &self, nb::ndarray<cfloat_t> states,
             cfloat_t c) {
            // TODO handle the non-transposed case since that's likely the most
            // common

            if (states.ndim() == 1) {
              // TODO lots of duplicate code here
              // TODO we can do better with this, right now it takes a 1D array
              // and returns a 2D one which isn't very intuitive
              // clang-format off
               auto states_mdspan = ndarray_to_mdspan<cfloat_t, 1>(states);
               auto states_mdspan_2d =std::mdspan(states_mdspan.data_handle(),states_mdspan.extent(0),1);
               auto new_states = owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan_2d);
               auto new_states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(new_states);
               self.apply_batch<float_type>(new_states_mdspan, states_mdspan_2d, c);
              // clang-format on
              return new_states;

            } else if (states.ndim() == 2) {
              // clang-format off
               auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
               auto new_states = owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
               auto new_states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(new_states);
               self.apply_batch<float_type>(new_states_mdspan, states_mdspan, c);
              // clang-format on
              return new_states;
            } else {
              throw std::invalid_argument(fmt::format(
                  "apply: expected 1 or 2 dimensions, got {}", states.ndim()));
            }
          },
          "states"_a, "coeff"_a = cfloat_t{1.0})
      .def(
          // TODO we should handle when users pass a single state (i.e. a 1D
          // array here)
          "expectation_value",
          [](fp::PauliString const &self, nb::ndarray<cfloat_t> states,
             cfloat_t c) {
            auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
            std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
            auto expected_vals_out =
                owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
            auto expected_vals_out_mdspan =
                ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);

            self.expectation_value(expected_vals_out_mdspan, states_mdspan, c);

            return expected_vals_out;
          },
          "states"_a, "coeff"_a = cfloat_t{1.0})

      //
      ;

  nb::class_<fp::SummedPauliOp<float_type>>(m, "SummedPauliOp")
      // Constructors
      // See
      // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
      .def(nb::init<>())
      .def("__init__",
           [](fp::SummedPauliOp<float_type> *new_obj,
              std::vector<std::string> &pauli_strings,
              nb::ndarray<cfloat_t> coeffs) {
             //
             auto coeffs_mdspan = ndarray_to_mdspan<cfloat_t, 2>(coeffs);
             //  std::vector<fp::PauliString> ps(pauli_strings.size());
             //  for (size_t i = 0; i < pauli_strings.size(); ++i) {
             //    ps[i] = fp::PauliString(pauli_strings[i]);
             //  }

             new (new_obj)
                 fp::SummedPauliOp<float_type>(pauli_strings, coeffs_mdspan);
             //  new (new_obj) fp::SummedPauliOp<float_type>(ps, coeffs_mdspan);

             std::vector<double> fake_vector(10000);
             for (size_t i = 0; i < 10; ++i) {
               fmt::println("fake vector: {}", fake_vector[i]);
             }
           })
      // .def("setup",
      //      [](fp::SummedPauliOp<float_type> const &self,
      //         std::vector<std::string> const &pauli_strings,
      //         nb::ndarray<cfloat_t> coeffs) {

      //      })
      .def_prop_ro("n_dimensions", &fp::SummedPauliOp<float_type>::n_dimensions)
      .def_prop_ro("n_operators", &fp::SummedPauliOp<float_type>::n_operators)
      .def_prop_ro("n_pauli_strings",
                   &fp::SummedPauliOp<float_type>::n_pauli_strings)

      .def("apply",
           [](fp::SummedPauliOp<float_type> const &self,
              nb::ndarray<cfloat_t> states, nb::ndarray<float_type> data) {
             auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
             auto data_mdspan = ndarray_to_mdspan<float_type, 2>(data);

             // clang-format off
             auto new_states        = owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
             auto new_states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(new_states);
             // clang-format on

             //  fmt::println("new_states ptr        {}",
             //               fmt::ptr(new_states.data()));
             //  fmt::println("new_states_mdspan ptr {}",
             //               fmt::ptr(new_states_mdspan.data_handle()));

             // DEBUG
             //  fmt::println("shape of states: ({},{})  size {}",
             //               states_mdspan.extent(0), states_mdspan.extent(1),
             //               states_mdspan.size());
             //  std::vector<cfloat_t> new_states(states_mdspan.size(), 0);
             //  std::mdspan<cfloat_t, std::dextents<size_t, 2>>
             //  new_states_mdspan(
             //      new_states.data(), states_mdspan.extent(0),
             //      states_mdspan.extent(1));
             // END DEBUG

             //  fmt::println("shape of states: ({},{})",
             //  states_mdspan.extent(0),
             //               states_mdspan.extent(1));
             //  fmt::println("shape of data:   ({},{})", data_mdspan.extent(0),
             //               data_mdspan.extent(1));
             //  fmt::println("self.dim: {}", self.n_dimensions());
             //  std::cout << std::flush;

             self.apply_parallel<float_type>(new_states_mdspan, states_mdspan,
                                             data_mdspan);

             //  self.apply(new_states_mdspan, states_mdspan, data_mdspan);
             return new_states;
           })
      //
      ;
}