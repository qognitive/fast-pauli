#include <algorithm>
#include <experimental/mdspan>
#include <random>

#include "__pauli.hpp"
#include "__pauli_string.hpp"
#include "fast_pauli.hpp"

using namespace fast_pauli;

int main() {

  //
  // User settings
  //
  size_t const n_operators = 10000;
  //   size_t const n_paulis_per_operator = 631;
  size_t const n_qubits = 14;
  size_t const weight = 2;
  size_t const n_states = 1000;
  //   std::string fake_pauli_string = "XYZXYZXYZXYZ";
  using fp_type = double;

  //
  // Setup the summed pauli operator
  //
  //   std::vector<PauliString> pauli_strings(n_paulis_per_operator,
  //                                          PauliString(fake_pauli_string));
  std::vector<PauliString> pauli_strings =
      calculate_pauli_strings_max_weight(n_qubits, weight);

  size_t const n_paulis_per_operator = pauli_strings.size();
  std::vector<std::complex<fp_type>> coeff_raw(
      n_paulis_per_operator * n_operators, 1);
  SummedPauliOp<fp_type> summed_op{pauli_strings, coeff_raw};

  //
  // Setup states
  //
  size_t const dim = summed_op.n_dimensions();
  size_t const n_ops = summed_op.n_operators();

  std::vector<std::complex<fp_type>> states_raw(dim * n_states, 1);
  std::mdspan<std::complex<fp_type>, std::dextents<size_t, 2>> states(
      states_raw.data(), dim, n_states);

  auto new_states_raw = std::vector<std::complex<fp_type>>(dim * n_states, 0);
  std::mdspan<std::complex<fp_type>, std::dextents<size_t, 2>> new_states(
      new_states_raw.data(), dim, n_states);

  //
  // Init weights (aka data)
  //
  std::vector<fp_type> weights_raw(n_ops * n_states, 1);
  std::mdspan<fp_type, std::dextents<size_t, 2>> weights(weights_raw.data(),
                                                         n_ops, n_states);

  //
  // Apply the states
  //
  //   summed_op.apply(new_states, states, weights);
  summed_op.apply_parallel(new_states, states, weights);

  return 0;
}