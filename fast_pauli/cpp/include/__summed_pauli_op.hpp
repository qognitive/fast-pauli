#ifndef __SUMMED_PAULI_OP_HPP
#define __SUMMED_PAULI_OP_HPP

#include "__pauli_string.hpp"
#include <omp.h>
#include <stdexcept>

namespace fast_pauli {

template <std::floating_point T> struct SummedPauliOp {

  // Short hand for complex, dynamic extent tensor with N dimension
  template <size_t N>
  using Tensor = std::mdspan<std::complex<T>, std::dextents<size_t, N>>;

  // std::vector<PauliOp<T>> ops;
  std::vector<PauliString> pauli_strings;
  std::vector<std::complex<T>> coeffs_raw;
  Tensor<2> coeffs;

  // TODO dangerous
  size_t _dim;
  size_t _n_operators;

  SummedPauliOp() noexcept = default;

  //
  SummedPauliOp(std::vector<PauliString> const &pauli_strings,
                std::vector<std::complex<T>> const &coeffs_raw)
      : pauli_strings(pauli_strings), coeffs_raw(coeffs_raw) {

    // TODO add more checks
    size_t const n_pauli_strings = pauli_strings.size();
    _dim = pauli_strings[0].dims();
    _n_operators = coeffs_raw.size() / n_pauli_strings;
    coeffs = Tensor<2>(this->coeffs_raw.data(), n_pauli_strings, _n_operators);

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

  SummedPauliOp(std::vector<PauliString> const &pauli_strings,
                Tensor<2> const coeffs)
      : pauli_strings(pauli_strings) {

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

    _dim = pauli_strings[0].dims();
    _n_operators = coeffs.extent(1);

    // Copy over the coeffs so our std::mdspan points at the memory owned by
    // this object
    coeffs_raw = std::vector<std::complex<T>>(coeffs.size());
    std::memcpy(this->coeffs_raw.data(), coeffs.data_handle(),
                coeffs.size() * sizeof(std::complex<T>));
    this->coeffs = std::mdspan<std::complex<T>, std::dextents<size_t, 2>>(
        this->coeffs_raw.data(), coeffs.extent(0), coeffs.extent(1));
  }

  SummedPauliOp(std::vector<std::string> const &pauli_strings,
                Tensor<2> const coeffs) {

    // Init the pauli strings
    this->pauli_strings.reserve(pauli_strings.size());
    for (auto const &ps : pauli_strings) {
      this->pauli_strings.emplace_back(ps);
    }

    // Check that the dims are all the same
    size_t const n_qubits = this->pauli_strings[0].n_qubits();
    bool const qubits_match =
        std::all_of(pauli_strings.begin(), pauli_strings.end(),
                    [n_qubits](PauliString const &ps) {
                      return ps.n_qubits() == n_qubits;
                    });
    if (!qubits_match) {
      throw std::invalid_argument("All PauliStrings must have the same size");
    }

    _dim = this->pauli_strings[0].dims();
    _n_operators = coeffs.extent(1);

    // Copy over the coeffs so our std::mdspan points at the memory owned by
    // this object
    coeffs_raw = std::vector<std::complex<T>>(coeffs.size());
    std::memcpy(this->coeffs_raw.data(), coeffs.data_handle(),
                coeffs.size() * sizeof(std::complex<T>));
    this->coeffs = std::mdspan<std::complex<T>, std::dextents<size_t, 2>>(
        this->coeffs_raw.data(), coeffs.extent(0), coeffs.extent(1));
  }

  //
  // Accessors/helpers
  //
  size_t n_dimensions() const noexcept { return _dim; }
  size_t n_operators() const noexcept { return _n_operators; }
  size_t n_pauli_strings() const noexcept { return pauli_strings.size(); }

  //
  // Primary (TODO "primary" is vague here) functions
  //
  void apply(Tensor<2> new_states, Tensor<2> states,
             std::mdspan<double, std::dextents<size_t, 2>> data) const {
    // TODO MAKE IT CLEAR THAT THE NEW_STATES NEED TO BE ZEROED

    // input checking
    if (states.extent(0) != new_states.extent(0) ||
        states.extent(1) != new_states.extent(1)) {
      throw std::invalid_argument(
          "new_states must have the same dimensions as states");
    }

    if (data.extent(0) != n_operators() || data.extent(1) != states.extent(1)) {
      throw std::invalid_argument(
          "data(k,t) must have the same number of operators as the "
          "SummedPauliOp "
          "and the same number of states as the input states");
    }

    if (states.extent(0) != n_dimensions()) {
      throw std::invalid_argument(
          "state size must match the dimension of the operators");
    }

    size_t const n_ps = n_pauli_strings();
    size_t const n_ops = n_operators();
    size_t const n_data = new_states.extent(1);
    size_t const n_dim = n_dimensions();

    std::vector<std::complex<T>> states_j_raw(n_data * n_dim);
    Tensor<2> states_j(states_j_raw.data(), n_dim, n_data);

    std::vector<std::complex<T>> weighted_coeffs_raw(n_data * n_dim);
    Tensor<2> weighted_coeffs(weighted_coeffs_raw.data(), n_ps, n_data);

    for (size_t j = 0; j < n_ps; ++j) {
      for (size_t t = 0; t < n_data; ++t) {
        for (size_t k = 0; k < n_ops; ++k) {
          weighted_coeffs(j, t) += coeffs(j, k) * data(k, t);
        }
      }
    }

    for (size_t j = 0; j < n_ps; ++j) {
      // new psi_prime
      std::fill(states_j_raw.begin(), states_j_raw.end(), std::complex<T>{0.0});
      pauli_strings[j].apply_batch(states_j, states, std::complex<T>(1.));
      for (size_t l = 0; l < n_dim; ++l) {
        for (size_t t = 0; t < n_data; ++t) {
          new_states(l, t) += states_j(l, t) * weighted_coeffs(j, t);
        }
      }
    }
  }

  template <std::floating_point data_dtype>
  void
  apply_parallel(Tensor<2> new_states, Tensor<2> states,
                 std::mdspan<data_dtype, std::dextents<size_t, 2>> data) const {
    // TODO MAKE IT CLEAR THAT THE NEW_STATES NEED TO BE ZEROED

    // input checking
    if (states.extent(0) != new_states.extent(0) ||
        states.extent(1) != new_states.extent(1)) {
      throw std::invalid_argument(
          "new_states must have the same dimensions as states");
    }

    if (data.extent(0) != n_operators() || data.extent(1) != states.extent(1)) {
      throw std::invalid_argument(
          "data(k,t) must have the same number of operators as the "
          "SummedPauliOp "
          "and the same number of states as the input states");
    }

    if (states.extent(0) != n_dimensions()) {
      throw std::invalid_argument(
          "state size must match the dimension of the operators");
    }

    size_t const n_ps = n_pauli_strings();
    size_t const n_ops = n_operators();
    size_t const n_data = new_states.extent(1);
    size_t const n_dim = n_dimensions();

    size_t const n_threads = omp_get_max_threads();
    std::vector<std::complex<T>> states_th_raw(n_threads * n_dim * n_data);
    Tensor<3> states_th(states_th_raw.data(), n_threads, n_dim, n_data);

    //
    std::vector<std::complex<T>> weighted_coeffs_raw(n_data * n_dim);
    Tensor<2> weighted_coeffs(weighted_coeffs_raw.data(), n_ps, n_data);

#pragma omp parallel
    {

      // Contract the coeffs with the data since we can reuse this below
#pragma omp for collapse(2)
      for (size_t j = 0; j < n_ps; ++j) {
        for (size_t t = 0; t < n_data; ++t) {
          for (size_t k = 0; k < n_ops; ++k) {
            weighted_coeffs(j, t) += coeffs(j, k) * data(k, t);
          }
        }
      }

      // Thread local temporaries and aliases
      std::vector<std::complex<T>> states_j_raw(n_data * n_dim);
      Tensor<2> states_j(states_j_raw.data(), n_dim, n_data);

      // std::vector<std::complex<T>> states_j_T_raw(n_data * n_dim);
      // Tensor<2> states_j_T(states_j_T_raw.data(), n_data, n_dim);

      std::mdspan states_th_local = std::submdspan(
          states_th, omp_get_thread_num(), std::full_extent, std::full_extent);

#pragma omp for schedule(dynamic)
      for (size_t j = 0; j < n_ps; ++j) {
        // new psi_prime
        pauli_strings[j].apply_batch(states_j, states, std::complex<T>(1.));

        for (size_t l = 0; l < n_dim; ++l) {
          for (size_t t = 0; t < n_data; ++t) {
            states_th_local(l, t) += states_j(l, t) * weighted_coeffs(j, t);
          }
        }
      }

      // Reduce
#pragma omp for collapse(2)
      for (size_t l = 0; l < n_dim; ++l) {
        for (size_t t = 0; t < n_data; ++t) {
          for (size_t i = 0; i < n_threads; ++i) {
            new_states(l, t) += states_th(i, l, t);
          }
        }
      }
    }
  }

  // TODO IMPLEMENT
  template <std::floating_point data_dtype>
  void apply_parallel_weighted_data(
      Tensor<2> new_states, Tensor<2> states,
      std::mdspan<data_dtype, std::dextents<size_t, 2>> data);
};

} // namespace fast_pauli

#endif
