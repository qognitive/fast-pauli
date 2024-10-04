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
#include <random>

#include "__factory.hpp"
#include "fast_pauli.hpp"

using namespace fast_pauli;

int main()
{
    std::vector<PauliString> pauli_strings(100000, "XYZXYZXYZXYZ");

    std::vector<std::complex<double>> coeffs(pauli_strings.size(), 1);
    PauliOp<double> pauli_op(coeffs, pauli_strings);

    size_t const dims = pauli_strings[0].dim();

    // Set up random state
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    std::vector<std::complex<double>> state(dims, 0);
    std::generate(state.begin(), state.end(), [&]() { return std::complex<double>(dis(gen), dis(gen)); });

    // Apply the PauliOp
    std::vector<std::complex<double>> result;
    auto span_result = empty(result, state.size());
    auto span_state = std::mdspan(state.data(), state.size());
    pauli_op.apply(span_result, span_state);

    return 0;
}