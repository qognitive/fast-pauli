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

#ifndef __FAST_PAULI_FACTORY_HPP
#define __FAST_PAULI_FACTORY_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <experimental/mdspan>
#include <random>
#include <vector>

#include "__type_traits.hpp"

namespace fast_pauli
{

//
// Factory functions
//

/*
Modelled after the torch factory functions here:
https://pytorch.org/cppdocs/notes/tensor_creation.html#picking-a-factory-function
*/

// Empty
template <typename T, size_t n_dim>
    requires is_complex<T>::value || std::floating_point<T>
constexpr auto empty(std::vector<T> &blob, std::array<size_t, n_dim> extents)
{
    // Calculate the total size and reserve the memory
    size_t total_size = 1;
    for (auto ei : extents)
    {
        total_size *= ei;
    }
    blob.resize(total_size);

    return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

template <typename T, typename... Is>
    requires(is_complex<T>::value || std::floating_point<T>) && (std::integral<Is> && ...)
constexpr auto empty(std::vector<T> &blob, Is... dims)
{
    constexpr size_t n_dim = sizeof...(Is);
    std::array<size_t, n_dim> extents = {dims...};

    // Calculate the total size and reserve the memory
    size_t total_size = 1;
    for (auto ei : extents)
    {
        total_size *= ei;
    }
    blob.resize(total_size);

    return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

// Zeros
template <typename T, size_t n_dim>
    requires is_complex<T>::value || std::floating_point<T>
constexpr auto zeros(std::vector<T> &blob, std::array<size_t, n_dim> extents)
{
    blob.clear(); // Clear so we have consistent behavior (e.g. not overwriting
                  // some of the values)

    // Calculate the total size and reserve the memory
    size_t total_size = 1;
    for (auto ei : extents)
    {
        total_size *= ei;
    }
    blob = std::vector<T>(total_size, 0);

    return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

// Rand
template <typename T, size_t n_dim>
    requires is_complex<T>::value || std::floating_point<T>
auto rand(std::vector<T> &blob, std::array<size_t, n_dim> extents, size_t seed = 18)
{
    blob.clear(); // Clear so we have consistent behavior (e.g. not overwriting
                  // some of the values)

    // Calculate the total size and reserve the memory
    size_t total_size = 1;
    for (auto ei : extents)
    {
        total_size *= ei;
    }
    blob = std::vector<T>(total_size);

    // Fill with random numbers
    // std::random_device rd;
    std::mt19937 gen(seed);

    // Internal specialization depending on whether we're dealing with regular FP
    // or complex
    if constexpr (is_complex<T>::value)
    {
        std::uniform_real_distribution<typename T::value_type> dis(0, 1.0);

        std::generate(blob.begin(), blob.end(), [&]() { return T{dis(gen), dis(gen)}; });
    }
    else
    {
        std::uniform_real_distribution<T> dis(0, 1.0);

        std::generate(blob.begin(), blob.end(), [&]() { return T{dis(gen)}; });
    }

    return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

} // namespace fast_pauli

#endif