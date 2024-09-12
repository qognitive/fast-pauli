#ifndef __NB_HELPERS_HPP
#define __NB_HELPERS_HPP

#include <iostream>
#include <numeric>
#include <ranges>

#include <experimental/mdspan>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

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
    for (size_t i = 0; i < ndim; ++i) {
      fmt::print("stride[{}] = {}, expected = {} (shape[{}] = {})\n", i,
                 strides[i], expected_strides[i], i, shape[i]);
    }
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
    for (size_t i = 0; i < ndim; ++i) {
      fmt::print("stride[{}] = {}, expected = {} (shape[{}] = {})\n", i,
                 strides[i], expected_strides[i], i, shape[i]);
    }
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
  // fmt::println("shape [{}] and size {}", fmt::join(shape, ", "), size);

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
    // fmt::println("deleting data in nb::capsule with shape");
    // std::cout << std::flush;
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

#endif // __NB_HELPERS_HPP