#ifndef __PAULI_STRING_HPP
#define __PAULI_STRING_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
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
        throw std::invalid_argument(std::string("Invalid Pauli character ") +
                                    c);
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
   * @note this returns 0 if the PauliString is empty.
   *
   * @return  size_t
   */
  size_t dims() const noexcept {
    return paulis.size() ? 1UL << paulis.size() : 0;
  }

  /**
   * @brief @copybrief PauliString::apply(std::mdspan)
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
    // route this to implementation we have for mdspan specialization
    return this->apply(std::mdspan(v.data(), v.size()));
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
  apply(std::mdspan<std::complex<T> const, std::dextents<size_t, 1>> v) const {
    // Input check
    if (v.size() != dims()) {
      throw std::invalid_argument(
          "Input vector size must match the number of qubits");
    }

    auto [k, m] = get_sparse_repr<T>(paulis);

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
   * (n_dim x n_states)
   * @param states_T THe original states to apply the PauliString to
   * (n_dim x n_states)
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

    auto [k, m] = get_sparse_repr<T>(paulis);

    for (size_t i = 0; i < states_T.extent(0); ++i) {
      std::copy_n(&states_T(k[i], 0), states_T.extent(1), &new_states_T(i, 0));
      const std::complex<T> c_m_i = c * m[i];
      for (size_t t = 0; t < states_T.extent(1); ++t) {
        new_states_T(i, t) *= c_m_i;
      }
    }
  }

  /**
   * @brief Calculate expected values for a given batch of states.
   * This function takes in transposed states with (n_dims x n_states) shape
   *
   * It computes following inner product
   * \f$ \bra{\psi_t} \mathcal{\hat{P_i}} \ket{\psi_t} \f$
   * for each state \f$ \ket{\psi_t} \f$ from provided batch.
   *
   * @note The expected values are added to corresponding coordinates
   * in the expected_vals_out vector.
   *
   * @tparam T The floating point base to use for all the complex numbers
   * @param expected_vals_out accumulator for expected values for each state in
   * the batch
   * @param states THe original states to apply the PauliString to
   * (n_dim x n_states)
   * @param c Multiplication factor to apply to the PauliString
   */
  template <std::floating_point T>
  void expected_value(
      std::mdspan<std::complex<T>, std::dextents<size_t, 1>> expected_vals_out,
      std::mdspan<std::complex<T> const, std::dextents<size_t, 2>> states,
      std::complex<T> const c = 1.0) const {
    // Input check
    if (states.extent(0) != dims())
      throw std::invalid_argument(
          fmt::format("[PauliString] states shape ({}) must match the dimension"
                      " of the operators ({})",
                      states.extent(0), dims()));
    if (expected_vals_out.extent(0) != states.extent(1))
      throw std::invalid_argument("[PauliString] expected_vals_out shape must "
                                  "match the number of states");

    auto [k, m] = get_sparse_repr<T>(paulis);

    for (size_t i = 0; i < states.extent(0); ++i) {
      const std::complex<T> c_m_i = c * m[i];
      for (size_t t = 0; t < states.extent(1); ++t) {
        expected_vals_out[t] +=
            std::conj(states(i, t)) * c_m_i * states(k[i], t);
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
    // Convert to dense representation
    auto [k, m] = get_sparse_repr<T>(paulis);

    std::vector<std::vector<std::complex<T>>> result(
        dims(), std::vector<std::complex<T>>(dims(), 0));

    for (size_t i = 0; i < k.size(); ++i) {
      result[i][k[i]] = m[i];
    }

    return result;
  }

  /**
   * @brief Get the sparse representation of the pauli string matrix.
   *
   * PauliStrings are always sparse and have only single non-zero element per
   * row. It's N non-zero elements for NxN matrix where N is 2^n_qubits.
   * Therefore k and m will always have N elements.
   *
   * See Algorithm 1 in https://arxiv.org/pdf/2301.00560.pdf for details about
   * the algorithm.
   *
   * @tparam T The floating point type (i.e. std::complex<T> for the values of
   * the PauliString matrix)
   * @param k The column index of the matrix
   * @param m The values of the matrix
   */
  template <std::floating_point T>
  static std::tuple<std::vector<size_t>, std::vector<std::complex<T>>>
  get_sparse_repr(std::vector<Pauli> const &paulis) {
    // We reverse the view here because the tensor product is taken from right
    // to left
    auto ps = paulis | std::views::reverse;
    size_t const n = paulis.size();
    size_t const nY =
        std::count_if(ps.begin(), ps.end(),
                      [](fast_pauli::Pauli const &p) { return p.code == 2; });
    size_t const dim = n ? 1 << n : 0;

    if (dim == 0)
      return {};

    std::vector<size_t> k(dim);
    std::vector<std::complex<T>> m(dim);

    // Helper function that let's us know if a pauli matrix has diagonal (or
    // conversely off-diagonal) elements
    auto diag = [](Pauli const &p) {
      if (p.code == 0 || p.code == 3) {
        return 0UL;
      } else {
        return 1UL;
      }
    };
    // Helper function that resolves first value of pauli string
    auto initial_value = [&nY]() -> std::complex<T> {
      switch (nY % 4) {
      case 0:
        return 1.0;
      case 1:
        return {0.0, -1.0};
      case 2:
        return -1.0;
      case 3:
        return {0.0, 1.0};
      }
      return {};
    };

    // Populate the initial values of our output
    k[0] = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
      k[0] += (1UL << i) * diag(ps[i]);
    }
    m[0] = initial_value();

    // Populate the rest of the values in a recursive-like manner
    for (size_t l = 0; l < n; ++l) {
      Pauli const &po = ps[l];

      T eps = 1.0;
      if (po.code == 2 || po.code == 3) {
        eps = -1;
      }

      T sign = diag(po) ? -1.0 : 1.0;

      auto const lower_bound = 1UL << l;
      for (size_t li = lower_bound; li < (lower_bound << 1); li++) {
        k[li] = k[li - lower_bound] + lower_bound * sign;
        m[li] = m[li - lower_bound] * eps;
      }
    }

    return std::make_tuple(std::move(k), std::move(m));
  }
};

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
    return fmt::format_to(ctx.out(), "{}", fmt::join(paulis, ""));
  }
};

#endif // __PAULI_STRING_HPP
