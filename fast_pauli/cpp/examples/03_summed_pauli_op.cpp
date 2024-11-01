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

#include <algorithm>
#include <chrono>
#include <experimental/mdspan>
#include <random>

#include "__pauli.hpp"
#include "__pauli_string.hpp"
#include "fast_pauli.hpp"

namespace fp = fast_pauli;

int main()
{
    //
    // User settings
    //
    size_t const n_operators = 1000;
    size_t const n_qubits = 12;
    size_t const weight = 2;
    size_t const n_states = 1000;
    using fp_type = float;

    //
    // Setup the summed pauli operator
    //
    std::vector<fp::PauliString> pauli_strings = fp::calculate_pauli_strings_max_weight(n_qubits, weight);

    size_t const n_paulis_per_operator = pauli_strings.size();
    std::vector<std::complex<fp_type>> coeff_raw(n_paulis_per_operator * n_operators, 1);
    fp::SummedPauliOp<fp_type> summed_op{pauli_strings, coeff_raw};

    //
    // Setup states
    //
    size_t const dim = summed_op.dim();
    size_t const n_ops = summed_op.n_operators();

    std::vector<std::complex<fp_type>> states_raw(dim * n_states, 1);
    std::mdspan<std::complex<fp_type>, std::dextents<size_t, 2>> states(states_raw.data(), dim, n_states);

    auto new_states_raw = std::vector<std::complex<fp_type>>(dim * n_states, 0);
    std::mdspan<std::complex<fp_type>, std::dextents<size_t, 2>> new_states(new_states_raw.data(), dim, n_states);

    //
    // Init weights (aka data)
    //
    std::vector<fp_type> weights_raw(n_ops * n_states, 1);
    std::mdspan<fp_type, std::dextents<size_t, 2>> weights(weights_raw.data(), n_ops, n_states);

    //
    // Apply the states
    //
    // auto start_seq = std::chrono::high_resolution_clock::now();
    // summed_op.apply_weighted(std::execution::seq, new_states, states, weights);
    // auto end_seq = std::chrono::high_resolution_clock::now();
    // fmt::println("Time taken for sequential execution: {} seconds",
    //              std::chrono::duration_cast<std::chrono::seconds>(end_seq - start_seq).count());

    auto start_par = std::chrono::high_resolution_clock::now();
    summed_op.apply_weighted(std::execution::par, new_states, states, weights);
    auto end_par = std::chrono::high_resolution_clock::now();
    fmt::println("Time taken for parallel execution:   {} seconds",
                 std::chrono::duration_cast<std::chrono::seconds>(end_par - start_par).count());

    return 0;
}
