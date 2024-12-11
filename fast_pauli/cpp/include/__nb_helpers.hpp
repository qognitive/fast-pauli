/**
 * This code is part of Fast Pauli.
 *
 * (C) Copyright Qognitive Inc 2024.
 *
 * This code is licensed under the BSD 2-Clause License. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef __NB_HELPERS_HPP
#define __NB_HELPERS_HPP

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <experimental/mdspan>
#include <iostream>
#include <numeric>
#include <ranges>

namespace nb = nanobind;
using namespace nb::literals;

namespace fast_pauli::__detail
{
/*
Helper functions
================

These functions are used to convert between Python's ndarray and C++'s mdspan
and there are three main use cases:

1. Initialize a C++ object by passing it data from Python: here we copy the
data so the C++ objects own their data.

2. Call a C++ function with some python array as an argument: here we can just
pass a view of the data to the C++ function.

3. Return an array from a C++ function: here we copy the data so there is no
ambiguity about ownership.
*/

template <size_t ndim>
void assert_row_major(std::array<size_t, ndim> const &shape, std::array<size_t, ndim> const &strides)
{
    // Check if the strides are C-style (row-major)
    std::array<size_t, ndim> expected_strides;

    // Calculate the expected strides using a prefix product (reversed)
    std::exclusive_scan(shape.rbegin(), shape.rend(), expected_strides.rbegin(), 1, std::multiplies<>{});

    //
    if (!std::ranges::equal(strides, expected_strides))
    {
        throw std::invalid_argument(fmt::format("ndarray MUST have C-style strides.\n"
                                                "Expected strides: {}\n"
                                                "Got strides:      {}\n"
                                                "Shape:            {}",
                                                fmt::join(expected_strides, ", "), fmt::join(strides, ", "),
                                                fmt::join(shape, ", ")));
    }
}

/**
 * @brief This function converts nb::ndarray to std::mdspan.
 *
 * NOTE: This only allows the default memory layout and accessor for the
 * mdspan.
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Number of dimensions
 * @tparam U Type of the input ndarray, can be nb::ndarray<T>,
 * nb::ndarray<nb::numpy, T>, etc ...
 * @param a
 * @return std::mdspan<T, std::dextents<size_t, ndim>>
 */

template <typename T, size_t ndim, typename U> std::mdspan<T, std::dextents<size_t, ndim>> ndarray_to_mdspan(U a)
{
    if (a.ndim() != ndim)
    {
        throw std::invalid_argument(fmt::format("ndarray_to_mdspan: expected {} dimensions, got {}", ndim, a.ndim()));
    }

    // Collect shape information
    std::array<size_t, ndim> shape;
    std::array<size_t, ndim> strides;

    for (size_t i = 0; i < ndim; ++i)
    {
        shape[i] = a.shape(i);
        strides[i] = a.stride(i);
    }

    assert_row_major(shape, strides);
    return std::mdspan<T, std::dextents<size_t, ndim>>(a.data(), shape);
}

/**
 * @brief This function copyies the data in the nb::ndarray to "raw" data in a
 * std::vector. It also return the shape informate so we can easily create an
 * mdspan of this array. We choose *not* to reuturn the mdspan directly
 * because it can create a dangling reference if the vector is moved as well
 * (which is often the case).
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Number of dimensions
 * @param a
 * @return std::pair<std::vector<T>, std::array<size_t, ndim>>
 */
template <typename T, size_t ndim> std::pair<std::vector<T>, std::array<size_t, ndim>> ndarray_to_raw(nb::ndarray<T> a)
{
    // Shape info
    std::array<size_t, ndim> shape;
    std::array<size_t, ndim> strides;
    for (size_t i = 0; i < a.ndim(); ++i)
    {
        shape[i] = a.shape(i);
        strides[i] = a.stride(i);
    }
    assert_row_major(shape, strides);

    // Copy the raw data
    std::vector<T> _data(a.size());
    std::memcpy(_data.data(), a.data(), a.size() * sizeof(T));
    return std::make_pair(std::move(_data), shape);
}

/**
 * @brief This function creates a new nb::ndarray that owns the data.
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Rank of the array
 * @param shape Shape array
 * @return nb::ndarray<nb::numpy, T>
 */
template <typename T, size_t ndim> nb::ndarray<nb::numpy, T> owning_ndarray_from_shape(std::array<size_t, ndim> shape)
{
    // Collect shape information
    size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());

    // Raw data

    // weirdness required by nanobind to properly pass ownership through
    // nb::handle, see https://github.com/wjakob/nanobind/discussions/573
    struct Temp
    {
        std::vector<T> data;
    };
    Temp *tmp = new Temp{std::vector<T>(size)};

    nb::capsule deleter(tmp, [](void *data) noexcept { delete static_cast<Temp *>(data); });

    // TODO can we do this without speciyfin that it's a numpy array?
    return nb::ndarray<nb::numpy, T>(
        /*data*/ tmp->data.data(),
        /*ndim*/ shape.size(),
        /*shape */ shape.data(),
        /*deleter*/ deleter);
}

/**
 * @brief Creates a new nb::ndarray that owns the data and has the same shape
 * as an mdspan.
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Rank of the array
 * @param a The mdspan
 * @return nb::ndarray<nb::numpy, T>
 */
template <typename T, size_t ndim>
nb::ndarray<nb::numpy, T> owning_ndarray_like_mdspan(std::mdspan<T, std::dextents<size_t, ndim>> a)
{
    // Collect shape information
    std::array<size_t, ndim> shape;
    for (size_t i = 0; i < ndim; ++i)
    {
        shape[i] = a.extent(i);
    }

    return owning_ndarray_from_shape<T, ndim>(shape);
}

/**
 * @brief Creates a new nb::ndarray that owns the data and has the same content and shape
 * as an mdspan.
 *
 * @tparam T Type of the underlying data in ndarray/mdspan
 * @tparam ndim Rank of the array
 * @param a The mdspan
 * @return nb::ndarray<nb::numpy, T>
 */
template <typename T, size_t ndim>
nb::ndarray<nb::numpy, T> owning_ndarray_from_mdspan(std::mdspan<T, std::dextents<size_t, ndim>> a)
{
    auto ndarray = owning_ndarray_like_mdspan<T, ndim>(a);
    std::memcpy(ndarray.data(), a.data_handle(), a.size() * sizeof(T));
    return ndarray;
}

} // namespace fast_pauli::__detail

#endif // __NB_HELPERS_HPP