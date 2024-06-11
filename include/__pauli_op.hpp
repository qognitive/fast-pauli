#ifndef __PAULI_OP_HPP
#define __PAULI_OP_HPP

#include <omp.h>

#include "__pauli_string.hpp"
#include <algorithm>
#include <experimental/mdspan> // From Kokkos

using namespace std::experimental;

namespace fast_pauli {

template <std::floating_point T> struct PauliOp {
  std::vector<std::complex<T>> coeffs;

  // TODO NEED TO THINK ABOUT THE ORDER HERE
  // DO WE ASSUME PEOPLE WANT IT COMPLETE? (if weight 3, do we include all
  // possible combinations up to and including weight 3 strings?)
  std::vector<PauliString> pauli_strings;

  PauliOp() = default;

  //
  PauliOp(std::vector<std::complex<T>> const &coeffs,
          std::vector<PauliString> const &pauli_strings)
      : coeffs(coeffs), pauli_strings(pauli_strings) {
    // TODO may want to wrap this in a #IFDEF DEBUG block to avoid the overhead
    // input check
    if (coeffs.size() != pauli_strings.size()) {
      throw std::invalid_argument(
          "coeffs and pauli_strings must have the same size");
    }

    // Check that the dims are all the same
    size_t const n_qubits = pauli_strings[0].n_qubits();
    bool const qubits_match =
        std::all_of(pauli_strings.begin(), pauli_strings.end(),
                    [n_qubits](PauliString const &ps) {
                      return ps.n_qubits() == n_qubits;
                    });
    if (!qubits_match) {
      throw std::invalid_argument("All PauliStrings must have the same size");
    }
  }

  size_t dims() const {
    if (pauli_strings.size() > 0) {
      return pauli_strings[0].dims();
    } else {
      return 0;
    }
  }

  std::vector<std::complex<T>>
  apply(std::vector<std::complex<T>> const &state) const {
    // input check
    if (state.size() != dims()) {
      throw std::invalid_argument(
          "state size must match the dimension of the operators");
    }

    //
    std::vector<std::complex<T>> res(state.size(), 0);

    for (size_t i = 0; i < pauli_strings.size(); ++i) {
      PauliString const &ps = pauli_strings[i];
      std::complex<T> c = coeffs[i];
      std::vector<std::complex<T>> tmp = ps.apply(state);
      for (size_t j = 0; j < state.size(); ++j) {
        res[j] += c * tmp[j];
      }
    }
    return res;
  }

  void apply_naive(
      mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states,
      mdspan<std::complex<T>, std::dextents<size_t, 2>> const states) const {
    // input check
    if (states.extent(1) != dims()) {
      throw std::invalid_argument(
          "state size must match the dimension of the operators");
    }
    // TODO add checks for new_state dimensions compared to old state dimensions
    if (states.extent(0) != new_states.extent(0) ||
        states.extent(1) != new_states.extent(1)) {
      throw std::invalid_argument(
          "new_states must have the same dimensions as states");
    }

    //
    size_t const n_states = states.extent(0);
#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < n_states; ++t) {
      for (size_t i = 0; i < pauli_strings.size(); ++i) {
        PauliString const &ps = pauli_strings[i];
        std::complex<T> c = coeffs[i];

        std::mdspan state_t = std::submdspan(states, t, std::full_extent);
        std::vector<std::complex<T>> tmp = ps.apply(state_t);
        for (size_t j = 0; j < states.extent(1); ++j) {
          new_states(t, j) += c * tmp[j];
        }
      }
    }
    return;
  }

  void
  apply(mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states,
        mdspan<std::complex<T>, std::dextents<size_t, 2>> const states) const {

    fmt::println(
        "[WARNING] Apply function causes problems on CI, use with caution.");

    // input check
    if (states.extent(0) != this->dims()) {
      throw std::invalid_argument(
          "[PauliOp] state size must match the dimension of the operators");
    }
    // TODO add checks for new_state dimensions compared to old state dimensions
    if (states.extent(0) != new_states.extent(0) ||
        states.extent(1) != new_states.extent(1)) {
      throw std::invalid_argument(
          "[PauliOp] new_states must have the same dimensions as states");
    }

    // Create tmp obj for reduction
    size_t const n_threads = omp_get_max_threads();
    size_t const n_data = states.extent(1);
    size_t const n_dim = states.extent(0);

    //     // transpose states
    //     std::vector<std::complex<T>> states_T_raw(n_data * n_dim);
    //     std::mdspan<std::complex<T>, std::dextents<size_t, 2>> states_T(
    //         states_T_raw.data(), n_dim, n_data);

    // #pragma omp parallel for schedule(static) collapse(2)
    //     for (size_t i = 0; i < n_data; ++i) {
    //       for (size_t j = 0; j < n_dim; ++j) {
    //         states_T(j, i) = states(i, j);
    //       }
    //     }

    std::vector<std::complex<T>> new_states_thr_raw(n_threads * n_dim * n_data);
    std::mdspan<std::complex<T>, std::dextents<size_t, 3>> new_states_thr(
        new_states_thr_raw.data(), n_threads, n_dim, n_data);

//
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < pauli_strings.size(); ++i) {
      size_t const tid = omp_get_thread_num();

      PauliString const &ps = pauli_strings[i];
      std::complex<T> c = coeffs[i];
      std::mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states_local =
          std::submdspan(new_states_thr, tid, std::full_extent,
                         std::full_extent);
      ps.apply_batch(new_states_local, states, c);
    }

    // Do the reduction and tranpose back
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < new_states.extent(0); ++i) {
      for (size_t t = 0; t < new_states.extent(1); ++t) {
        for (size_t th = 0; th < n_threads; ++th) {
          new_states(i, t) += new_states_thr(th, i, t);
        }
      }
    }
  }

  //
  // Helpers (mostly for debugging)
  //
  std::vector<std::vector<std::complex<T>>> get_dense_repr() const {
    size_t const dim = 1UL << pauli_strings[0].paulis.size();

    std::vector<std::vector<std::complex<T>>> res(
        dim, std::vector<std::complex<T>>(dim, 0));

    for (size_t i = 0; i < pauli_strings.size(); ++i) {
      PauliString const &ps = pauli_strings[i];
      std::complex<T> c = coeffs[i];

      auto ps_dense = ps.get_dense_repr<T>();

      for (size_t j = 0; j < dim; ++j) {
        for (size_t k = 0; k < dim; ++k) {
          res[j][k] += c * ps_dense[j][k];
        }
      }
    }
    return res;
  }
};

} // namespace fast_pauli

#endif // __PAULI_OP_HPP
