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
#include <experimental/mdspan>
#include <random>

#include "fast_pauli.hpp"

using namespace fast_pauli;

int main()
{
    using mytype = float;

    // Setup PauliOp
    std::vector<PauliString> pauli_strings(631, "XYZXYZXYZXYZ");
    std::vector<std::complex<mytype>> coeffs(pauli_strings.size(), 1);
    PauliOp<mytype> pauli_op(coeffs, pauli_strings);

    size_t const dims = pauli_strings[0].dim();

    // Set up random states
    size_t const n_states = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    std::vector<std::complex<mytype>> states_raw(dims * n_states, 0);
    std::generate(states_raw.begin(), states_raw.end(), [&]() { return std::complex<mytype>(dis(gen), dis(gen)); });
    std::mdspan<std::complex<mytype>, std::dextents<size_t, 2>> states(states_raw.data(), dims, n_states);

    // Apply the PauliOp
    std::vector<std::complex<mytype>> new_states_raw(dims * n_states, 0);
    std::mdspan<std::complex<mytype>, std::dextents<size_t, 2>> new_states(new_states_raw.data(), dims, n_states);

    // pauli_op.apply_naive(new_states, states);
    pauli_op.apply(new_states, states);
    return 0;
}