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

#ifndef __SUMMED_PAULI_OP_HPP
#define __SUMMED_PAULI_OP_HPP

#include <chrono>
#include <fmt/core.h>
#include <omp.h>

#include <cstddef>
#include <stdexcept>
#include <unordered_map>

#include "__pauli_string.hpp"
#include "__type_traits.hpp"

namespace fast_pauli
{

/**
 * @brief A class representing a sum of Pauli operators \f$ A = \sum_k A_k  = \sum_{ik} h_{ik} \mathcal{\hat{P}}_i \f$.
 * Where \f$ \mathcal{\hat{P}}_i \f$ are Pauli strings and \f$ h_{ik} \f$ are complex-valued coefficients.
 *
 */
template <std::floating_point T> struct SummedPauliOp
{
    // Short hand for complex, dynamic extent tensor with N dimension
    template <size_t N> using Tensor = std::mdspan<std::complex<T>, std::dextents<size_t, N>>;

    // std::vector<PauliOp<T>> ops;
    std::vector<PauliString> pauli_strings;
    std::vector<std::complex<T>> coeffs_raw;
    Tensor<2> coeffs; // (n_pauli_strings, n_operators)

    void __check_ctor_inputs(std::vector<fast_pauli::PauliString> const &pauli_strings, Tensor<2> const coeffs)
    {
        // Check the PauliStrings to make sure they're all the same size
        size_t const n_qubits = pauli_strings.front().n_qubits();
        bool const qubits_match =
            std::all_of(pauli_strings.begin(), pauli_strings.end(),
                        [n_qubits](fast_pauli::PauliString const &ps) { return ps.n_qubits() == n_qubits; });
        if (!qubits_match)
        {
            throw std::invalid_argument("All PauliStrings must have the same size");
        }

        // Check the shape of the coeffs
        if (coeffs.extent(0) != pauli_strings.size())
        {
            throw std::invalid_argument("The number of PauliStrings must match the number of rows in the "
                                        "coeffs matrix");
        }
    }

    /**
     * @brief Default constructor
     *
     */
    SummedPauliOp() noexcept = default;

    /**
     * @brief Construct a new Summed Pauli Op object from a vector of PauliStrings
     * and a blob of coefficients.
     *
     * @param pauli_strings The PauliStrings that define the set of PauliStrings used by all operators (n_pauli_strings)
     * @param coeffs_raw A vector of coefficients that define the weights of each PauliString in each operator.
     * The coefficients here are a flattened version of \f$ h_{ik} \f$ in \f$ A_k = \sum_i h_{ik} \mathcal{\hat{P}}_i
     * \f$ (n_pauli_strings * n_operators,)
     *
     */
    SummedPauliOp(std::vector<PauliString> const &pauli_strings, std::vector<std::complex<T>> const &coeffs_raw)
        : pauli_strings(pauli_strings), coeffs_raw(coeffs_raw)
    {
        // TODO add more checks
        size_t const n_pauli_strings = pauli_strings.size();
        size_t const n_operators = coeffs_raw.size() / n_pauli_strings;
        coeffs = Tensor<2>(this->coeffs_raw.data(), n_pauli_strings, n_operators);

        this->__check_ctor_inputs(pauli_strings, coeffs);
    }

    /**
     * @brief Construct a new Summed Pauli Op object from a vector of PauliStrings
     * and an std::mdspan of coefficients.
     *
     * @param pauli_strings The PauliStrings that define the set of PauliStrings used by all operators
     * (n_pauli_strings,)
     * @param coeffs A 2D std::mdspan of coefficients that define the weights of each
     * PauliString in each operator. The coefficients here are \f$ h_{ik} \f$ in
     * \f$ A_k = \sum_i h_{ik} \mathcal{\hat{P}}_i \f$. (n_pauli_strings, n_operators)
     */
    SummedPauliOp(std::vector<PauliString> const &pauli_strings, Tensor<2> const coeffs) : pauli_strings(pauli_strings)
    {
        //
        this->__check_ctor_inputs(pauli_strings, coeffs);

        // Copy over the coeffs so our std::mdspan points at the memory owned by
        // this object
        coeffs_raw = std::vector<std::complex<T>>(coeffs.size());
        std::memcpy(this->coeffs_raw.data(), coeffs.data_handle(), coeffs.size() * sizeof(std::complex<T>));
        this->coeffs = std::mdspan<std::complex<T>, std::dextents<size_t, 2>>(this->coeffs_raw.data(), coeffs.extent(0),
                                                                              coeffs.extent(1));
    }

    /**
     * @brief Construct a new Summed Pauli Op object from a vector of strings and
     * a std::mdspan of coefficients.
     *
     * @param pauli_strings A vector of strings that define the set of PauliStrings used by all operators
     * (n_pauli_strings)
     * @param coeffs A 2D std::mdspan of coefficients that define the weights of each
     * PauliString in each operator. The coefficients here are \f$ h_{ik} \f$ in
     * \f$ A_k = \sum_i h_{ik} \mathcal{\hat{P}}_i \f$. (n_pauli_strings, n_operators)
     */
    SummedPauliOp(std::vector<std::string> const &pauli_strings, Tensor<2> const coeffs)
    {
        // Init the pauli strings
        // this->pauli_strings.reserve(pauli_strings.size());
        for (auto const &ps : pauli_strings)
        {
            // this->pauli_strings.emplace_back(ps);
            this->pauli_strings.push_back(PauliString(ps));
        }

        this->__check_ctor_inputs(this->pauli_strings, coeffs);

        // Copy over the coeffs so our std::mdspan points at the memory owned by
        // this object
        coeffs_raw = std::vector<std::complex<T>>(coeffs.size());
        std::memcpy(this->coeffs_raw.data(), coeffs.data_handle(), coeffs.size() * sizeof(std::complex<T>));
        this->coeffs = std::mdspan<std::complex<T>, std::dextents<size_t, 2>>(this->coeffs_raw.data(), coeffs.extent(0),
                                                                              coeffs.extent(1));
    }

    //
    // Accessors/helpers
    //
    /**
     * @brief Return the number of dimensions of the SummedPauliOp
     *
     * @return size_t
     */
    size_t dim() const noexcept
    {
        return pauli_strings.size() ? pauli_strings[0].dim() : 0;
    }

    /**
     * @brief Return the number of qubits in the SummedPauliOp
     *
     * @return size_t
     */
    size_t n_qubits() const noexcept
    {
        return pauli_strings.size() ? pauli_strings[0].n_qubits() : 0;
    }

    /**
     * @brief Return the number of Pauli operators in the SummedPauliOp
     *
     * @return s
     */
    size_t n_operators() const noexcept
    {
        return coeffs.extent(1);
    }

    /**
     * @brief Return the number of PauliStrings in the SummedPauliOp
     *
     * @return size_t
     */
    size_t n_pauli_strings() const noexcept
    {
        return pauli_strings.size();
    }

    /**
     * @brief Square the SummedPauliOp, mathematically \f$ A_k \rightarrow A_k^2 = \sum_{a,b} T_{ab} A_a A_b \f$.
     *
     * We use the sprarse structure of A_k operators and PauliString products to calculate the new coefficients.
     *
     * @return SummedPauliOp<T>
     */
    fast_pauli::SummedPauliOp<T> square() const
    {
        // Part 1: Get the squared pauli strings
        size_t weight = std::transform_reduce(
            pauli_strings.begin(), pauli_strings.end(), size_t(0),
            // Reduce
            [](size_t e1, size_t e2) { return std::max(e1, e2); },
            // Transform
            [](fast_pauli::PauliString const &ps) { return static_cast<size_t>(ps.weight); });

        std::vector<PauliString> pauli_strings_sq =
            fast_pauli::calculate_pauli_strings_max_weight(n_qubits(), std::min(n_qubits(), 2UL * weight));

        std::unordered_map<fast_pauli::PauliString, size_t> sq_idx_map;
        for (size_t i = 0; i < pauli_strings_sq.size(); ++i)
        {
            sq_idx_map[pauli_strings_sq[i]] = i;
        }

        // Part 2: Create the T_aij tensor that maps the coeffiencts from h_i * h_j to h'_a
        // Create t_aij sparse tensor
        std::vector<std::vector<std::tuple<size_t, size_t, std::complex<T>>>> t_aij(
            pauli_strings_sq.size(), std::vector<std::tuple<size_t, size_t, std::complex<T>>>());

        // Serial because we'll have a race condition on the t_aij[a] vector
        for (size_t i = 0; i < pauli_strings.size(); ++i)
        {
            for (size_t j = 0; j < pauli_strings.size(); ++j)
            {
                std::complex<T> phase;
                fast_pauli::PauliString prod;
                std::tie(phase, prod) = pauli_strings[i] * pauli_strings[j];
                size_t a = sq_idx_map[prod];
                t_aij[a].emplace_back(i, j, phase);
            }
        }

        // Part 3: Create the new coeffs tensor (transpose of the old one for better memory access)
        std::vector<std::complex<T>> coeffs_t_raw(coeffs.size());
        Tensor<2> coeffs_t(coeffs_t_raw.data(), coeffs.extent(1), coeffs.extent(0));

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < coeffs.extent(0); ++i)
        {
            for (size_t j = 0; j < coeffs.extent(1); ++j)
            {
                coeffs_t(j, i) = coeffs(i, j);
            }
        }

        // Part 4: Contract the T_aij tensor with the coeffs to get the new coeffs
        std::vector<std::complex<T>> coeffs_sq_raw(pauli_strings_sq.size() * n_operators());
        Tensor<2> coeffs_sq(coeffs_sq_raw.data(), pauli_strings_sq.size(), n_operators());

        size_t i, j;
        std::complex<T> phase;

#pragma omp parallel for schedule(dynamic) collapse(2) private(i, j, phase)
        for (size_t a = 0; a < pauli_strings_sq.size(); ++a)
        {
            for (size_t k = 0; k < n_operators(); ++k)
            {
                for (size_t x = 0; x < t_aij[a].size(); ++x)
                {
                    std::tie(i, j, phase) = t_aij[a][x];
                    coeffs_sq(a, k) += phase * coeffs_t(k, i) * coeffs_t(k, j);
                }
            }
        }

        return SummedPauliOp<T>(pauli_strings_sq, coeffs_sq);
    }

    /**
     * @brief Apply the SummedPauliOp to a set of states, mathematically
     * \f$ \big(\sum_k \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t} \f$
     *
     * @param new_states The output states after applying the SummedPauliOp (n_dim, n_states)
     * @param states The input states to apply the SummedPauliOp to (n_dim, n_states)
     */
    void apply(Tensor<2> new_states, Tensor<2> states) const
    {
        apply(std::execution::seq, new_states, states);
    }

    /**
     * @brief \copydoc SummedPauliOp::apply(Tensor<2>, Tensor<2>) const
     *
     * @tparam ExecutionPolicy
     */
    template <execution_policy ExecutionPolicy>
    void apply(ExecutionPolicy &&, Tensor<2> new_states, Tensor<2> states) const
    {
        if (states.extent(0) != new_states.extent(0) || states.extent(1) != new_states.extent(1))
        {
            throw std::invalid_argument("new_states must have the same dimensions as states");
        }
        size_t const n_threads = omp_get_max_threads();
        size_t const n_ps = n_pauli_strings();
        size_t const n_ops = n_operators();
        size_t const n_dim = dim();
        size_t const n_data = states.extent(1);

        std::vector<std::complex<T>> new_states_thr_raw(n_threads * n_dim * n_data);
        std::mdspan<std::complex<T>, std::dextents<size_t, 3>> new_states_thr(new_states_thr_raw.data(), n_threads,
                                                                              n_dim, n_data);

        if constexpr (is_parallel_execution_policy_v<ExecutionPolicy>)
        {

#pragma omp parallel
            {
#pragma omp for schedule(static)
                for (size_t j = 0; j < n_ps; ++j)
                {
                    std::complex<T> summed_coeff(0, 0);

                    for (size_t k = 0; k < n_ops; ++k)
                    {
                        summed_coeff += coeffs(j, k);
                    }

                    std::mdspan new_states_j =
                        std::submdspan(new_states_thr, omp_get_thread_num(), std::full_extent, std::full_extent);
                    pauli_strings[j].apply_batch(new_states_j, states, summed_coeff);
                }

#pragma omp for schedule(static) collapse(2)
                for (size_t i = 0; i < n_dim; ++i)
                {
                    for (size_t t = 0; t < n_data; ++t)
                    {
                        for (size_t j = 0; j < n_threads; ++j)
                        {
                            new_states(i, t) += new_states_thr(j, i, t);
                        }
                    }
                }
            }
        }
        else
        {
            for (size_t j = 0; j < n_ps; ++j)
            {
                std::complex<T> c(0, 0);
                for (size_t k = 0; k < n_ops; ++k)
                {
                    c += coeffs(j, k);
                }
                pauli_strings[j].apply_batch(new_states, states, c);
            }
        }
    }

    /**
     * @brief Apply the SummedPauliOp to a set of weighted states.
     *
     * Calculates \f$
     * \big(\sum_k x_{tk} \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}
     * \f$
     *
     * @tparam data_dtype The floating point type of the weights \f$ x_{kt} \f$ (n_operators, n_states)
     * @param new_states The output states after applying the SummedPauliOp (n_dim, n_states)
     * @param states The input states to apply the SummedPauliOp to (n_dim, n_states)
     * @param data A 2D std::mdspan of the data \f$ x_{tk} \f$ that weights the operators in the expression above
     * (n_operators, n_states)
     */
    template <std::floating_point data_dtype>
    void apply_weighted(Tensor<2> new_states, Tensor<2> states,
                        std::mdspan<data_dtype, std::dextents<size_t, 2>> data) const
    {
        apply_weighted(std::execution::seq, new_states, states, data);
    }

    /**
     * @brief \copydoc SummedPauliOp::apply_weighted(Tensor<2>, Tensor<2>,
     * std::mdspan<double, std::dextents<size_t, 2>>) const
     *
     * @tparam ExecutionPolicy Execution policy for parallelization
     * @tparam data_dtype The floating point type of the weights \f$ x_{kt} \f$ (n_operators, n_states)
     */
    template <execution_policy ExecutionPolicy, std::floating_point data_dtype>
    void apply_weighted(ExecutionPolicy &&, Tensor<2> new_states, Tensor<2> states,
                        std::mdspan<data_dtype, std::dextents<size_t, 2>> data) const
    {
        // TODO MAKE IT CLEAR THAT THE NEW_STATES NEED TO BE ZEROED
        // input checking
        if (states.extent(0) != new_states.extent(0) || states.extent(1) != new_states.extent(1))
        {
            throw std::invalid_argument("new_states must have the same dimensions as states");
        }

        if (data.extent(0) != n_operators() || data.extent(1) != states.extent(1))
        {
            throw std::invalid_argument("data(k,t) must have the same number of operators as the "
                                        "SummedPauliOp "
                                        "and the same number of states as the input states");
        }

        if (states.extent(0) != dim())
        {
            throw std::invalid_argument("state size must match the dimension of the operators");
        }

        size_t const n_ps = n_pauli_strings();
        size_t const n_ops = n_operators();
        size_t const n_data = new_states.extent(1);
        size_t const n_dim = dim();

        if constexpr (is_parallel_execution_policy_v<ExecutionPolicy>)
        {
            size_t const n_threads = omp_get_max_threads();
            std::vector<std::complex<T>> new_states_th_raw(n_threads * n_dim * n_data);
            Tensor<3> new_states_th(new_states_th_raw.data(), n_threads, n_dim, n_data);

            //
            std::vector<std::complex<T>> weighted_coeffs_raw(n_ps * n_data);
            Tensor<2> weighted_coeffs(weighted_coeffs_raw.data(), n_ps, n_data);

#pragma omp parallel
            {

                // Contract the coeffs with the data since we can reuse this below
#pragma omp for schedule(dynamic) collapse(2)
                for (size_t j = 0; j < n_ps; ++j)
                {
                    for (size_t t = 0; t < n_data; ++t)
                    {
                        for (size_t k = 0; k < n_ops; ++k)
                        {
                            // TODO we should transpose the data here for better memory access
                            // initial tests didn't show much improvement, but something to try later
                            weighted_coeffs(j, t) += coeffs(j, k) * data(k, t);
                        }
                    }
                }

                // Thread local temporaries and aliases
                std::vector<std::complex<T>> new_states_j_raw(n_data * n_dim);
                Tensor<2> new_states_j(new_states_j_raw.data(), n_dim, n_data);

                std::mdspan new_states_th_local =
                    std::submdspan(new_states_th, omp_get_thread_num(), std::full_extent, std::full_extent);

#pragma omp for schedule(dynamic)
                for (size_t j = 0; j < n_ps; ++j)
                {
                    // new psi_prime
                    std::fill(new_states_j_raw.begin(), new_states_j_raw.end(), std::complex<T>{0.0});
                    pauli_strings[j].apply_batch(new_states_j, states, std::complex<T>(1.));

                    for (size_t l = 0; l < n_dim; ++l)
                    {
                        for (size_t t = 0; t < n_data; ++t)
                        {
                            new_states_th_local(l, t) += new_states_j(l, t) * weighted_coeffs(j, t);
                        }
                    }
                }
                // Reduce
#pragma omp for schedule(dynamic) collapse(2)
                for (size_t l = 0; l < n_dim; ++l)
                {
                    for (size_t t = 0; t < n_data; ++t)
                    {
                        for (size_t i = 0; i < n_threads; ++i)
                        {
                            new_states(l, t) += new_states_th(i, l, t);
                        }
                    }
                }
            }
        }
        else
        {
            std::vector<std::complex<T>> states_j_raw(n_data * n_dim);
            Tensor<2> states_j(states_j_raw.data(), n_dim, n_data);

            std::vector<std::complex<T>> weighted_coeffs_raw(n_ps * n_data);
            Tensor<2> weighted_coeffs(weighted_coeffs_raw.data(), n_ps, n_data);

            for (size_t j = 0; j < n_ps; ++j)
            {
                for (size_t t = 0; t < n_data; ++t)
                {
                    for (size_t k = 0; k < n_ops; ++k)
                    {
                        weighted_coeffs(j, t) += coeffs(j, k) * data(k, t);
                    }
                }
            }

            for (size_t j = 0; j < n_ps; ++j)
            {
                // new psi_prime
                std::fill(states_j_raw.begin(), states_j_raw.end(), std::complex<T>{0.0});
                pauli_strings[j].apply_batch(states_j, states, std::complex<T>(1.));
                for (size_t l = 0; l < n_dim; ++l)
                {
                    for (size_t t = 0; t < n_data; ++t)
                    {
                        new_states(l, t) += states_j(l, t) * weighted_coeffs(j, t);
                    }
                }
            }
        }
    }

    /**
     * @brief Calculate the expectation value of the SummedPauliOp on a batch of
     * states. This function returns the expectation values of each operator for
     * each input states, so the output tensor will have shape (n_operators x
     * n_states).
     *
     * \f$
     * \bra{\psi_t} \big(\sum_k \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}
     * \f$
     *
     * @param expectation_vals_out Output tensor for the expectation values
     * (n_operators x n_states)
     * @param states The states used to calculate the expectation values (n_dim,
     * n_states)
     */
    void expectation_value(Tensor<2> expectation_vals_out, Tensor<2> states) const
    {
        expectation_value(std::execution::seq, expectation_vals_out, states);
    }

    /**
     * @brief \copydoc SummedPauliOp::expectation_value(Tensor<2>, Tensor<2>) const
     *
     * @tparam ExecutionPolicy Execution policy for parallelization
     */
    template <execution_policy ExecutionPolicy>
    void expectation_value(ExecutionPolicy &&, Tensor<2> expectation_vals_out, Tensor<2> states) const
    {
        size_t const n_data = states.extent(1);
        size_t const n_ops = n_operators();

        //
        // Input checking
        //
        if (expectation_vals_out.extent(0) != n_ops)
        {
            throw std::invalid_argument(fmt::format("expectation_vals_out must have the same number of "
                                                    "operators ({}) as the SummedPauliOp ({})",
                                                    expectation_vals_out.extent(0), n_ops));
        }

        if (states.extent(0) != dim())
        {
            throw std::invalid_argument(fmt::format("states must have the same dimension ({}) as the "
                                                    "SummedPauliOp ({})",
                                                    states.extent(0), dim()));
        }

        if (expectation_vals_out.extent(1) != n_data)
        {
            throw std::invalid_argument(fmt::format("expectation_vals_out must have the same number of "
                                                    "states ({}) as the input states ({})",
                                                    expectation_vals_out.extent(1), n_data));
        }

        //
        // Calculate the expectation values
        //
        // Expectation value of paulis (n_pauli_strings, n_data)
        std::vector<std::complex<T>> expectation_vals_raw(n_pauli_strings() * n_data);
        Tensor<2> expectation_vals(expectation_vals_raw.data(), n_pauli_strings(), n_data);

        if constexpr (is_parallel_execution_policy_v<ExecutionPolicy>)
        {

#pragma omp parallel
            {
#pragma omp for schedule(static)
                for (size_t j = 0; j < n_pauli_strings(); ++j)
                {
                    std::mdspan expectation_vals_j = std::submdspan(expectation_vals, j, std::full_extent);
                    pauli_strings[j].expectation_value(expectation_vals_j, states);
                }

#pragma omp for collapse(2) schedule(static)
                for (size_t k = 0; k < n_ops; ++k)
                {
                    for (size_t t = 0; t < n_data; ++t)
                    {
                        for (size_t j = 0; j < n_pauli_strings(); ++j)
                        {
                            // Could be better about memory access patterns here, but I don't think the transposes are
                            // worth it
                            expectation_vals_out(k, t) += coeffs(j, k) * expectation_vals(j, t);
                        }
                    }
                }
            }
        }
        else
        {
            for (size_t j = 0; j < n_pauli_strings(); ++j)
            {
                std::mdspan expectation_vals_j = std::submdspan(expectation_vals, j, std::full_extent);
                pauli_strings[j].expectation_value(expectation_vals_j, states);
            }

            // einsum("jk,jt->kt", coeffs, expectation_vals)
            for (size_t t = 0; t < n_data; ++t)
            {
                for (size_t k = 0; k < n_ops; ++k)
                {
                    for (size_t j = 0; j < n_pauli_strings(); ++j)
                    {
                        expectation_vals_out(k, t) += coeffs(j, k) * expectation_vals(j, t);
                    }
                }
            }
        }
    }

    /**
     * @brief Get the dense representation of the SummedPauliOp as a 3D tensor
     *
     * @param A_k_out The output tensor to fill with the dense representation
     */
    void to_tensor(Tensor<3> A_k_out) const
    {
        // TODO need to add input checks to this
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < n_operators(); ++k)
        {
            // This is a lot of wasted work, but in some quick tests with n_ops = 1000, n_qubits = 10, weight = 2 this
            // strategy was slightly faster than have the loop over i being the outer-most loop.
            for (size_t i = 0; i < pauli_strings.size(); ++i)
            {
                PauliString const &ps = pauli_strings[i];
                std::vector<size_t> cols;
                std::vector<std::complex<T>> vals;
                std::tie(cols, vals) = get_sparse_repr<T>(ps.paulis);

                for (size_t j = 0; j < dim(); ++j)
                {
                    A_k_out(k, j, cols[j]) += coeffs(i, k) * vals[j];
                }
            }
        }
    }
};

} // namespace fast_pauli

#endif
