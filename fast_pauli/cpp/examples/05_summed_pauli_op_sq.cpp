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
    using fp_type = float;

    //
    // Setup the summed pauli operator
    //
    std::vector<fp::PauliString> pauli_strings = fp::calculate_pauli_strings_max_weight(n_qubits, weight);

    size_t const n_paulis_per_operator = pauli_strings.size();
    std::vector<std::complex<fp_type>> coeff_raw(n_paulis_per_operator * n_operators, 1);
    fp::SummedPauliOp<fp_type> summed_op{pauli_strings, coeff_raw};

    //
    // Apply the states
    //

    auto start_par = std::chrono::high_resolution_clock::now();
    auto sq_op = summed_op.square();
    auto end_par = std::chrono::high_resolution_clock::now();
    fmt::println("Time taken for parallel execution:   {} seconds",
                 std::chrono::duration_cast<std::chrono::seconds>(end_par - start_par).count());

    return 0;
}
