#ifndef __FAST_PAULI_FACTORY_HPP
#define __FAST_PAULI_FACTORY_HPP

#include <array>
#include <complex>
#include <experimental/mdspan>
#include <random>
#include <vector>

namespace fast_pauli {

//
// Types traits and concepts
//
template <typename T> struct is_complex : std::false_type {};

template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> struct type_identity {
  typedef T type;
};

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
constexpr auto empty(std::vector<T> &blob, std::array<size_t, n_dim> extents) {

  // Calculate the total size and reserve the memory
  size_t total_size = 1;
  for (auto ei : extents) {
    total_size *= ei;
  }
  blob.reserve(total_size);

  return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

// Zeros
template <typename T, size_t n_dim>
  requires is_complex<T>::value || std::floating_point<T>
constexpr auto zeros(std::vector<T> &blob, std::array<size_t, n_dim> extents) {
  blob.clear(); // Clear so we have consistent behavior (e.g. not overwriting
                // some of the values)

  // Calculate the total size and reserve the memory
  size_t total_size = 1;
  for (auto ei : extents) {
    total_size *= ei;
  }
  blob = std::vector<T>(total_size, 0);

  return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

// Rand
template <typename T, size_t n_dim>
  requires is_complex<T>::value || std::floating_point<T>
auto rand(std::vector<T> &blob, std::array<size_t, n_dim> extents) {
  blob.clear(); // Clear so we have consistent behavior (e.g. not overwriting
                // some of the values)

  // Calculate the total size and reserve the memory
  size_t total_size = 1;
  for (auto ei : extents) {
    total_size *= ei;
  }
  blob.reserve(total_size);

  // Fill with random numbers
  std::random_device rd;
  std::mt19937 gen(rd());

  // Internal specialization depending on whether we're dealing with regular FP
  // or complex
  if constexpr (is_complex<T>::value) {
    std::uniform_real_distribution<typename T::value_type> dis(0, 1.0);

    std::ranges::generate(blob, [&]() { return T{dis(gen), dis(gen)}; });
  } else {
    std::uniform_real_distribution<T> dis(0, 1.0);

    std::ranges::generate(blob, [&]() { return T{dis(gen)}; });
  }

  return std::mdspan<T, std::dextents<size_t, n_dim>>(blob.data(), extents);
}

} // namespace fast_pauli

#endif