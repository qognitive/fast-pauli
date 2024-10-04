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

#include <fmt/core.h>
#include <omp.h>

#include <cstddef>
#include <stdexcept>

#include "__pauli_string.hpp"

namespace fast_pauli
{

template <std::floating_point T> struct SummedPauliOp
{
    // Short hand for complex, dynamic extent tensor with N dimension
    template <size_t N> using Tensor = std::mdspan<std::complex<T>, std::dextents<size_t, N>>;

    // std::vector<PauliOp<T>> ops;
    std::vector<PauliString> pauli_strings;
    std::vector<std::complex<T>> coeffs_raw;
    Tensor<2> coeffs;

    // TODO dangerous
    size_t _dim;
    size_t _n_operators;

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
     * @param pauli_strings
     * @param coeffs_raw
     */
    SummedPauliOp(std::vector<PauliString> const &pauli_strings, std::vector<std::complex<T>> const &coeffs_raw)
        : pauli_strings(pauli_strings), coeffs_raw(coeffs_raw)
    {
        // TODO add more checks
        size_t const n_pauli_strings = pauli_strings.size();
        _dim = pauli_strings[0].dim();
        _n_operators = coeffs_raw.size() / n_pauli_strings;
        coeffs = Tensor<2>(this->coeffs_raw.data(), n_pauli_strings, _n_operators);

        this->__check_ctor_inputs(pauli_strings, coeffs);
    }

    /**
     * @brief Construct a new Summed Pauli Op object from a vector of PauliStrings
     * and an std::mdspan of coefficients.
     *
     * @param pauli_strings
     * @param coeffs
     */
    SummedPauliOp(std::vector<PauliString> const &pauli_strings, Tensor<2> const coeffs) : pauli_strings(pauli_strings)
    {
        //
        this->__check_ctor_inputs(pauli_strings, coeffs);

        //
        _dim = pauli_strings[0].dim();
        _n_operators = coeffs.extent(1);

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
     * @param pauli_strings
     * @param coeffs
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

        //
        _dim = this->pauli_strings.front().dim();
        _n_operators = coeffs.extent(1);

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
        return _dim;
    }

    /**
     * @brief Return the number of operators in the SummedPauliOp
     *
     * @return s
     */
    size_t n_operators() const noexcept
    {
        return _n_operators;
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
     * @brief Apply the SummedPauliOp to a set of weighted states.
     *
     * Calculates \f$
     * \bra{\psi_t} \big(\sum_k x_{tk} \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}
     * \f$
     *
     * @param new_states
     * @param new_states
     * @param states
     * @param data
     */
    void apply_weighted(Tensor<2> new_states, Tensor<2> states,
                        std::mdspan<double, std::dextents<size_t, 2>> data) const
    {
        apply_weighted(std::execution::seq, new_states, states, data);
    }

    /**
     * @brief \copydoc SummedPauliOp::apply_weighted(Tensor<2>, Tensor<2>,
     * std::mdspan<double, std::dextents<size_t, 2>>) const
     *
     * @tparam data_dtype
     * @tparam ExecutionPolicy
     * @param new_states
     * @param states
     * @param data
     */
    template <std::floating_point data_dtype, execution_policy ExecutionPolicy>
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

        if constexpr (std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
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
#pragma omp for schedule(static) collapse(2)
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

                // Thread local temporaries and aliases
                std::vector<std::complex<T>> new_states_j_raw(n_data * n_dim);
                Tensor<2> new_states_j(new_states_j_raw.data(), n_dim, n_data);

                // std::vector<std::complex<T>> states_j_T_raw(n_data * n_dim);
                // Tensor<2> states_j_T(states_j_T_raw.data(), n_data, n_dim);

                std::mdspan new_states_th_local =
                    std::submdspan(new_states_th, omp_get_thread_num(), std::full_extent, std::full_extent);

#pragma omp for schedule(static)
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
#pragma omp for schedule(static) collapse(2)
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
     * @param expectation_vals_out Output tensor for the expectation values
     * (n_operators x n_states)
     * @param states The states used to calculate the expectation values (n_dim x
     * n_states)
     */
    void expectation_value(Tensor<2> expectation_vals_out, Tensor<2> states) const
    {
        expectation_value(std::execution::seq, expectation_vals_out, states);
    }

    /**
     * @brief \copydoc SummedPauliOp::expectation_value(Tensor<2>, Tensor<2>) const
     *
     * @tparam ExecutionPolicy
     * @param expectation_vals_out
     * @param states
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

        if constexpr (std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
        {

#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < n_pauli_strings(); ++j)
            {
                std::mdspan expectation_vals_j = std::submdspan(expectation_vals, j, std::full_extent);
                pauli_strings[j].expectation_value(expectation_vals_j, states);
            }

            // einsum("jk,jt->kt", coeffs, expectation_vals)
#pragma omp parallel for collapse(2)
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
};

} // namespace fast_pauli

#endif
