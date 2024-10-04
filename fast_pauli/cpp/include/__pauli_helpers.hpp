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

#ifndef __PAULI_HELPERS_HPP
#define __PAULI_HELPERS_HPP

#include <fmt/format.h>

#include <ranges>

#include "__pauli_string.hpp"

namespace fast_pauli
{
//
// Helper
//

/**
 * @brief Get the nontrivial sets of pauli matrices given a weight.
 *
 * @param weight
 * @return std::vector<std::string>
 */
std::vector<std::string> get_nontrivial_paulis(size_t const weight)
{
    // We want to return no paulis for weight 0
    if (weight == 0)
    {
        return {};
    }

    // For Weight >= 1
    std::vector<std::string> set_of_nontrivial_paulis{"X", "Y", "Z"};

    for (size_t i = 1; i < weight; i++)
    {
        std::vector<std::string> updated_set_of_nontrivial_paulis;
        for (auto const &str : set_of_nontrivial_paulis)
        {
            for (auto pauli : {"X", "Y", "Z"})
            {
                updated_set_of_nontrivial_paulis.push_back(str + pauli);
            }
        }
        set_of_nontrivial_paulis = std::move(updated_set_of_nontrivial_paulis);
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
std::vector<std::vector<size_t>> idx_combinations(size_t const n, size_t const k)
{
    // TODO this is a very inefficient way to do this
    std::vector<std::vector<size_t>> result;
    std::vector<size_t> bitmask(k, 1); // K leading 1's
    bitmask.resize(n, 0);              // N-K trailing 0's

    do
    {
        std::vector<size_t> combo;
        for (size_t i = 0; i < n; ++i)
        {
            if (bitmask[i])
            {
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
std::vector<PauliString> calcutate_pauli_strings(size_t const n_qubits, size_t const weight)
{
    // base case
    if (weight == 0)
    {
        return {PauliString(std::string(n_qubits, 'I'))};
    }

    // for weight >= 1
    std::string base_str(n_qubits, 'I');

    auto nontrivial_paulis = get_nontrivial_paulis(weight);
    auto idx_combos = idx_combinations(n_qubits, weight);
    size_t n_pauli_strings = nontrivial_paulis.size() * idx_combos.size();
    std::vector<PauliString> result(n_pauli_strings);

    fmt::println("n_qubits = {}  weight = {}  n_nontrivial_paulis = {}  n_combos = {}", n_qubits, weight,
                 nontrivial_paulis.size(), idx_combos.size());

    // Iterate through all the nontrivial paulis and all the combinations
    for (size_t i = 0; i < nontrivial_paulis.size(); ++i)
    {
        for (size_t j = 0; j < idx_combos.size(); ++j)
        {
            // Creating new pauli string at index i*idx_combos.size() + j
            // Overwriting the base string with the appropriate nontrivial paulis
            // at the specified indices
            std::string str = base_str;
            for (size_t k = 0; k < idx_combos[j].size(); ++k)
            {
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
std::vector<PauliString> calculate_pauli_strings_max_weight(size_t n_qubits, size_t weight)
{
    std::vector<PauliString> result;
    for (size_t i = 0; i <= weight; ++i)
    {
        auto ps = calcutate_pauli_strings(n_qubits, i);
        result.insert(result.end(), ps.begin(), ps.end());
    }

    fmt::println("n_qubits = {}  weight = {}  n_pauli_strings = {}", n_qubits, weight, result.size());
    return result;
}

} // namespace fast_pauli

#endif // __PAULI_HELPERS_HPP
