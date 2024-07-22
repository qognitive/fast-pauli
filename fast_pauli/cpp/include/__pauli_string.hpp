#ifndef __PAULI_STRING_HPP
#define __PAULI_STRING_HPP

#include <cstddef>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstring>
#include <experimental/mdspan>
#include <ranges>
#include <string>

#include "__pauli.hpp"

namespace fast_pauli {

/**
 * @brief A class representation of a Pauli string (i.e. a tensor product of 2x2
 * pauli matrices) \f$ $\mathcal{\hat{P}} = \bigotimes_i \sigma_i \f$
 * where \f$ \sigma_i \in \{ I,X,Y,Z \} \f$
 *
 */
struct PauliString {
  uint8_t weight;
  std::vector<Pauli> paulis;

  //
  // Constructors
  //

  /**
   * @brief Default constructor, initialize weight and empty vector for paulis.
   *
   */
  PauliString() noexcept = default;

  /**
   * @brief Constructs a PauliString from a span of pauli operators and
   * calculates the weight.
   *
   */
  PauliString(std::span<fast_pauli::Pauli> const &paulis)
      : weight(0), paulis(paulis.begin(), paulis.end()) {
    for (auto const &pauli : paulis) {
      weight += pauli.code > 0;
    }
  }

  /**
   * @brief Constructs a PauliString from a string and calculates the weight.
   * This is often the most compact way to initialize a PauliString.
   *
   */
  PauliString(std::string const &str) : weight(0) {
    for (auto const &c : str) {
      switch (c) {
      case 'I':
        paulis.push_back(fast_pauli::Pauli{0});
        break;
      case 'X':
        paulis.push_back(fast_pauli::Pauli{1});
        weight += 1;
        break;
      case 'Y':
        paulis.push_back(fast_pauli::Pauli{2});
        weight += 1;
        break;
      case 'Z':
        paulis.push_back(fast_pauli::Pauli{3});
        weight += 1;
        break;
      default:
        throw std::invalid_argument(std::string("Invalid Pauli character") + c);
      }
    }
  }

  /**
   * @brief Allows implicit conversion of string literals to PauliStrings.
   * Ex: std::vector<PauliString> pauli_strings = {"IXYZ", "IIIII"};
   *
   */
  PauliString(char const *str) : PauliString(std::string(str)) {}

  PauliString(PauliString const &other)
      : weight(other.weight), paulis(other.paulis) {};
  PauliString &operator=(PauliString const &other) {
    this->weight = other.weight;
    this->paulis = other.paulis;
    return *this;
  };

  //
  friend auto operator<=>(PauliString const &, PauliString const &) = default;

  //
  /**
   * @brief Return the number of qubits in the PauliString.
   *
   * @return  size_t
   */
  size_t n_qubits() const noexcept { return paulis.size(); }

  /**
   * @brief Return the dimension (2^n_qubits) of the PauliString.
   *
   * @return  size_t
   */
  size_t dims() const noexcept { return 1UL << paulis.size(); }

  /**
   * @brief Get the sparse representation of the pauli string matrix.
   *
   * PauliStrings are always sparse and have only single non-zero element per
   * row. It's N non-zero elements for NxN matrix where N is 2^n_qubits.
   * Therefore j, k, and m will always have N elements.
   *
   * TODO remove j because it's unused (and redundant).
   * See Algorithm 1 in https://arxiv.org/pdf/2301.00560.pdf for details about
   * the algorithm.
   *
   * @tparam T The floating point type (i.e. std::complex<T> for the values of
   * the PauliString matrix)
   * @param j (unused)
   * @param k The column index of the matrix
   * @param m The values of the matrix
   */
  template <std::floating_point T>
  void get_sparse_repr(std::vector<size_t> &j, std::vector<size_t> &k,
                       std::vector<std::complex<T>> &m) const {
    // We reverse the view here because the tensor product is taken from right
    // to left
    auto ps = paulis | std::views::reverse;
    size_t const n = paulis.size();
    size_t const nY =
        std::count_if(ps.begin(), ps.end(),
                      [](fast_pauli::Pauli const &p) { return p.code == 2; });
    size_t const dim = 1 << n;

    // Safe, but expensive, we overwrite the vectors
    j = std::vector<size_t>(dim);
    // j.clear();
    k = std::vector<size_t>(dim);
    m = std::vector<std::complex<T>>(dim);

    // Helper function that let's us know if a pauli matrix has diagonal (or
    // conversely off-diagonal) elements
    auto diag = [](Pauli const &p) {
      if (p.code == 0 || p.code == 3) {
        return 0UL;
      } else {
        return 1UL;
      }
    };

    // Populate the initial values of our output
    k[0] = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
      k[0] += (1UL << i) * diag(ps[i]);
    }
    m[0] = std::pow(-1i, nY % 4);

    // Populate the rest of the values in a recursive-like manner
    for (size_t l = 0; l < n; ++l) {
      Pauli const &po = ps[l];

      T eps = 1.0;
      if (po.code == 2 || po.code == 3) {
        eps = -1;
      }

      T sign = std::pow(-1., diag(po)); // Keep this outside the loop

      for (size_t li = (1UL << l); li < (1UL << (l + 1)); li++) {
        k[li] = k[li - (1UL << l)] + (1UL << l) * sign;
        m[li] = m[li - (1UL << l)] * eps;
      }
    }
  }

  /**
   * @brief Apply the PauliString (using the sparse representation) to a vector.
   * This performs following matrix-vector multiplication \f$ \mathcal{\hat{P}}
   * \ket{\psi} \f$
   *
   * @tparam T The floating point base to use for all the complex numbers
   * @param v The input vector to apply the PauliString to. Must be the same
   * size as PauliString.dims().
   * @return  std::vector<std::complex<T>> The output state after
   * applying the PauliString.
   */
  template <std::floating_point T>
  std::vector<std::complex<T>>
  apply(std::vector<std::complex<T>> const &v) const {
    // Input check
    if (v.size() != dims()) {
      throw std::invalid_argument(
          "Input vector size must match the number of qubits");
    }

    std::vector<size_t> j, k; // TODO reminder that j is unused
    std::vector<std::complex<T>> m;
    get_sparse_repr(j, k, m);

    std::vector<std::complex<T>> result(v.size(), 0);
    for (size_t i = 0; i < k.size(); ++i) {
      result[i] += m[i] * v[k[i]];
    }

    return result;
  }

  /**
   * @brief @copybrief PauliString::apply(std::vector<std::complex<T>>)
   *
   * @tparam T The floating point base to use for all the complex numbers
   * @param v The input vector to apply the PauliString to. Must be the same
   * size as PauliString.dims().
   * @return  std::vector<std::complex<T>> The output state after
   * applying the PauliString.
   */
  template <std::floating_point T>
  std::vector<std::complex<T>>
  apply(std::mdspan<std::complex<T>, std::dextents<size_t, 1>> v) const {
    // Input check
    if (v.size() != dims()) {
      throw std::invalid_argument(
          "Input vector size must match the number of qubits");
    }

    std::vector<size_t> j, k; // TODO reminder that j is unused
    std::vector<std::complex<T>> m;
    get_sparse_repr(j, k, m);

    std::vector<std::complex<T>> result(v.size(), 0);
    for (size_t i = 0; i < k.size(); ++i) {
      result[i] += m[i] * v(k[i]);
    }

    return result;
  }

  /**
   * @brief Apply the PauliString to a batch of states. This function takes a
   * different shape of the states than the other apply functions. here all the
   * states (new and old) are transposed so their shape is (n_dims x n_states).
   * All the new_stats are overwritten, no need to initialize.
   *
   * This performs following matrix-matrix multiplication \f$ \mathcal{\hat{P}}
   * \hat{\Psi} \f$ where matrix \f$ \hat{\Psi} \f$ has \f$ \ket{\psi_t} \f$ as
   * columns
   *
   * @tparam T The floating point base to use for all the complex numbers
   * @param new_states_T The output states after applying the PauliString
   * (n_data x n_dim)
   * @param states_T THe original states to apply the PauliString to (n_data x
   * n_dim)
   * @param c Multiplication factor to apply to the PauliString
   */
  template <std::floating_point T>
  void apply_batch(std::mdspan<std::complex<T>, std::dextents<size_t, 2>>
                       new_states_T, // extent(0) = dims, extent(1) = n_states
                   std::mdspan<std::complex<T>, std::dextents<size_t, 2>> const
                       states_T, // extent(0) = dims, extent(1) = n_states
                   std::complex<T> const c) const {
    // Input check
    if (states_T.extent(0) != dims()) {
      auto error_msg =
          fmt::format("[PauliString] states shape ({}) must match the "
                      "dimension of the operators ({})",
                      states_T.extent(0), dims());
      throw std::invalid_argument(error_msg);
    }

    if ((states_T.extent(0) != new_states_T.extent(0)) ||
        states_T.extent(1) != new_states_T.extent(1)) {
      throw std::invalid_argument(
          "[PauliString] new_states must have the same dimensions as states");
    }

    std::vector<size_t> j, k; // TODO reminder that j is unused
    std::vector<std::complex<T>> m;
    get_sparse_repr(j, k, m);

    // TODO we have bad memory access patterns
    // std::vector<std::complex<T>> col(states_T.extent(1), 0);
    for (size_t i = 0; i < states_T.extent(0); ++i) {
      std::complex<T> c_m_i = c * m[i];
      size_t k_i = k[i];
      std::memcpy(&new_states_T(i, 0), &states_T(k_i, 0),
                  states_T.extent(1) * sizeof(std::complex<T>));

      for (size_t t = 0; t < states_T.extent(1); ++t) {
        new_states_T(i, t) *= c_m_i; // * states_T(k_i, t);
      }
    }
  }

  //
  // Debugging Helpers
  //
  /**
   * @brief Get the dense representation of the object as a 2D-std::vector
   *
   * @tparam T The floating point base to use for all the complex numbers
   * @return  std::vector<std::vector<std::complex<T>>>
   */
  template <std::floating_point T>
  std::vector<std::vector<std::complex<T>>> get_dense_repr() const {
    //
    std::vector<size_t> j, k;
    std::vector<std::complex<T>> m;
    get_sparse_repr(j, k, m);

    // Convert to dense representation
    size_t const dim = 1UL << paulis.size();
    std::vector<std::vector<std::complex<T>>> result(
        dim, std::vector<std::complex<T>>(dim, 0));

    for (size_t i = 0; i < k.size(); ++i) {
      result[i][k[i]] = m[i];
    }

    return result;
  }
};

//
// Helper
//

/**
 * @brief Get the nontrivial sets of pauli matrices given a weight.
 *
 * @param weight
 * @return std::vector<std::string>
 */
std::vector<std::string> get_nontrivial_paulis(size_t const weight) {
  // We want to return no paulis for weight 0
  if (weight == 0) {
    return {};
  }

  // For Weight >= 1
  std::vector<std::string> set_of_nontrivial_paulis{"X", "Y", "Z"};

  for (size_t i = 1; i < weight; i++) {
    std::vector<std::string> updated_set_of_nontrivial_paulis;
    for (auto str : set_of_nontrivial_paulis) {
      for (auto pauli : {"X", "Y", "Z"}) {
        updated_set_of_nontrivial_paulis.push_back(str + pauli);
      }
    }
    set_of_nontrivial_paulis = updated_set_of_nontrivial_paulis;
  }
  return set_of_nontrivial_paulis;
}

/**
 * @brief Get all the combinations of k indices for a given array of size n.
 *
 * @param n
 * @param k
 * @return std::vector<std::vector<size_t>>
 */
std::vector<std::vector<size_t>> idx_combinations(size_t const n,
                                                  size_t const k) {

  // TODO this is a very inefficient way to do this
  std::vector<std::vector<size_t>> result;
  std::vector<size_t> bitmask(k, 1); // K leading 1's
  bitmask.resize(n, 0);              // N-K trailing 0's

  do {
    std::vector<size_t> combo;
    for (size_t i = 0; i < n; ++i) {
      if (bitmask[i]) {
        combo.push_back(i);
      }
    }
    result.push_back(combo);
  } while (std::ranges::prev_permutation(bitmask).found);
  return result;
}

/**
 * @brief Calculate all possible PauliStrings for a given number of qubits and
 * weight and return them in lexicographical order.
 *
 * @param n_qubits
 * @param weight
 * @return std::vector<PauliString>
 */
std::vector<PauliString> calcutate_pauli_strings(size_t const n_qubits,
                                                 size_t const weight) {

  // base case
  if (weight == 0) {
    return {PauliString(std::string(n_qubits, 'I'))};
  }

  // for weight >= 1
  std::string base_str(n_qubits, 'I');

  auto nontrivial_paulis = get_nontrivial_paulis(weight);
  auto idx_combos = idx_combinations(n_qubits, weight);
  size_t n_pauli_strings = nontrivial_paulis.size() * idx_combos.size();
  std::vector<PauliString> result(n_pauli_strings);

  fmt::println(
      "n_qubits = {}  weight = {}  n_nontrivial_paulis = {}  n_combos = {}",
      n_qubits, weight, nontrivial_paulis.size(), idx_combos.size());

  // Iterate through all the nontrivial paulis and all the combinations
  for (size_t i = 0; i < nontrivial_paulis.size(); ++i) {
    for (size_t j = 0; j < idx_combos.size(); ++j) {
      // Creating new pauli string at index i*idx_combos.size() + j
      // Overwriting the base string with the appropriate nontrivial paulis
      // at the specified indices
      std::string str = base_str;
      for (size_t k = 0; k < idx_combos[j].size(); ++k) {
        size_t idx = idx_combos[j][k];
        str[idx] = nontrivial_paulis[i][k];
      }
      result[i * idx_combos.size() + j] = PauliString(str);
    }
  }

  return result;
}

/**
 * @brief Calculate all possible PauliStrings for a given number of qubits and
 * all weights less than or equal to a given weight.
 *
 * @param n_qubits
 * @param weight
 * @return std::vector<PauliString>
 */
std::vector<PauliString> calculate_pauli_strings_max_weight(size_t n_qubits,
                                                            size_t weight) {
  std::vector<PauliString> result;
  for (size_t i = 0; i <= weight; ++i) {
    auto ps = calcutate_pauli_strings(n_qubits, i);
    result.insert(result.end(), ps.begin(), ps.end());
  }

  fmt::println("n_qubits = {}  weight = {}  n_pauli_strings = {}", n_qubits,
               weight, result.size());
  return result;
}

} // namespace fast_pauli

//
// fmt::formatter specialization
//

//
template <> struct fmt::formatter<fast_pauli::PauliString> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(fast_pauli::PauliString const &ps, FormatContext &ctx) const {
    std::vector<fast_pauli::Pauli> paulis = ps.paulis;
    return fmt::format_to(ctx.out(), "{}", fmt::join(paulis, "x"));
  }
};

#endif // __PAULI_STRING_HPP
