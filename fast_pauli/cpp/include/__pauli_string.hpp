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

#ifndef __PAULI_STRING_HPP
#define __PAULI_STRING_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <experimental/mdspan>
#include <functional>
#include <omp.h>
#include <ranges>
#include <string>

#include "__pauli.hpp"
#include "__type_traits.hpp"

namespace fast_pauli
{

/**
 * @brief Get the sparse representation of the pauli string matrix.
 *
 * PauliStrings are always sparse and have only single non-zero element per
 * row. It's N non-zero elements for NxN matrix where N is 2^n_qubits.
 * Therefore k and m will always have N elements.
 *
 * See Algorithm 1 in https://arxiv.org/pdf/2301.00560.pdf for details about
 * the algorithm.
 *
 * @tparam T The floating point type (i.e. std::complex<T> for the values of
 * the PauliString matrix)
 * @param k The column index of the matrix
 * @param m The values of the matrix
 */
template <std::floating_point T>
std::tuple<std::vector<size_t>, std::vector<std::complex<T>>> get_sparse_repr(std::vector<Pauli> const &paulis)
{
    // We reverse the view here because the tensor product is taken from right
    // to left
    auto ps = paulis | std::views::reverse;
    size_t const n = paulis.size();
    size_t const nY = std::count_if(ps.begin(), ps.end(), [](fast_pauli::Pauli const &p) { return p.code == 2; });
    size_t const dim = n ? 1 << n : 0;

    if (dim == 0)
        return {};

    std::vector<size_t> k(dim);
    std::vector<std::complex<T>> m(dim);

    // Helper function that let's us know if a pauli matrix has diagonal (or
    // conversely off-diagonal) elements
    auto diag = [](Pauli const &p) {
        if (p.code == 0 || p.code == 3)
        {
            return 0UL;
        }
        else
        {
            return 1UL;
        }
    };
    // Helper function that resolves first value of pauli string
    auto initial_value = [&nY]() -> std::complex<T> {
        switch (nY % 4)
        {
        case 0:
            return 1.0;
        case 1:
            return {0.0, -1.0};
        case 2:
            return -1.0;
        case 3:
            return {0.0, 1.0};
        }
        return {};
    };

    // Populate the initial values of our output
    k[0] = 0;
    for (size_t i = 0; i < ps.size(); ++i)
    {
        k[0] += (1UL << i) * diag(ps[i]);
    }
    m[0] = initial_value();

    // Populate the rest of the values in a recursive-like manner
    for (size_t l = 0; l < n; ++l)
    {
        Pauli const &po = ps[l];

        T eps = 1.0;
        if (po.code == 2 || po.code == 3)
        {
            eps = -1;
        }

        T sign = diag(po) ? -1.0 : 1.0;

        auto const lower_bound = 1UL << l;
        for (size_t li = lower_bound; li < (lower_bound << 1); li++)
        {
            k[li] = k[li - lower_bound] + lower_bound * sign;
            m[li] = m[li - lower_bound] * eps;
        }
    }

    return std::make_tuple(std::move(k), std::move(m));
}

/**
 * @brief A class representation of a Pauli string (i.e. a tensor product of 2x2
 * pauli matrices) \f$ $\mathcal{\hat{P}} = \bigotimes_i \sigma_i \f$
 * where \f$ \sigma_i \in \{ I,X,Y,Z \} \f$
 *
 */
struct PauliString
{
    uint8_t weight;
    std::vector<Pauli> paulis;

    //
    // Constructors
    //

    /**
     * @brief Default constructor, initialize weight and empty vector for paulis.
     *
     */
    PauliString() noexcept = default;

    /**
     * @brief Constructs a PauliString from a vector of pauli matrices and
     * calculates the weight.
     *
     */
    PauliString(std::vector<Pauli> paulis) : weight(0), paulis(std::move(paulis))
    {
        for (auto const &pauli : this->paulis)
        {
            weight += pauli.code > 0;
        }
    }

    /**
     * @brief Constructs a PauliString from a span of pauli matrices and
     * calculates the weight.
     *
     */
    PauliString(std::span<fast_pauli::Pauli> paulis) : weight(0), paulis(paulis.begin(), paulis.end())
    {
        for (auto const &pauli : paulis)
        {
            weight += pauli.code > 0;
        }
    }

    /**
     * @brief Constructs a PauliString from a string and calculates the weight.
     * This is often the most compact way to initialize a PauliString.
     *
     */
    PauliString(std::string const &str) : weight(0)
    {
        for (auto const &c : str)
        {
            switch (c)
            {
            case 'I':
                paulis.push_back(fast_pauli::Pauli{0});
                break;
            case 'X':
                paulis.push_back(fast_pauli::Pauli{1});
                weight += 1;
                break;
            case 'Y':
                paulis.push_back(fast_pauli::Pauli{2});
                weight += 1;
                break;
            case 'Z':
                paulis.push_back(fast_pauli::Pauli{3});
                weight += 1;
                break;
            default:
                throw std::invalid_argument(std::string("Invalid Pauli character ") + c);
            }
        }
    }

    /**
     * @brief Allows implicit conversion of string literals to PauliStrings.
     * Ex: std::vector<PauliString> pauli_strings = {"IXYZ", "IIIII"};
     *
     */
    PauliString(char const *str) : PauliString(std::string(str))
    {
    }

    PauliString(PauliString const &other) : weight(other.weight), paulis(other.paulis) {};
    PauliString &operator=(PauliString const &other)
    {
        this->weight = other.weight;
        this->paulis = other.paulis;
        return *this;
    };

    // TODO should we separately define <,> operators based just on pualis vector
    friend auto operator<=>(PauliString const &, PauliString const &) = default;

    /**
     * @brief Returns the result of matrix multiplication of two PauliStrings and
     * their phase as a pair.
     *
     * @param lhs left hand side PauliString
     * @param rhs right hand side PauliString
     * @return  std::pair<std::complex<double>, PauliString> phase and resulting
     * PauliString
     */
    friend std::pair<std::complex<double>, PauliString> operator*(PauliString const &lhs, PauliString const &rhs)
    {
        if (lhs.dim() != rhs.dim())
        {
            throw std::invalid_argument("PauliStrings must have the same size");
        }

        std::complex<double> new_phase = 1;
        std::vector<Pauli> new_paulis;
        new_paulis.reserve(lhs.n_qubits());

        for (size_t i = 0; i < lhs.n_qubits(); ++i)
        {
            auto [phase, new_pauli] = lhs.paulis[i] * rhs.paulis[i];
            new_phase *= phase;
            new_paulis.push_back(new_pauli);
        }

        return {new_phase, PauliString(std::move(new_paulis))};
    }

    //
    /**
     * @brief Return the number of qubits in the PauliString.
     *
     * @return  size_t
     */
    size_t n_qubits() const noexcept
    {
        return paulis.size();
    }

    /**
     * @brief Return the dimension (2^n_qubits) of the PauliString.
     * @note this returns 0 if the PauliString is empty.
     *
     * @return  size_t
     */
    size_t dim() const noexcept
    {
        return paulis.size() ? 1UL << paulis.size() : 0;
    }

    template <std::floating_point T>
    void __input_checks_apply_1d(std::mdspan<std::complex<T>, std::dextents<size_t, 1>> new_states,
                                 std::mdspan<std::complex<T>, std::dextents<size_t, 1>> states) const
    {
        if (states.size() != dim())
        {
            throw std::invalid_argument("Input vector size must match the number of qubits");
        }
        if (states.size() != new_states.size())
        {
            throw std::invalid_argument("new_states must have the same dimensions as states");
        }
    }

    /**
     * @brief Apply a pauli string (using the sparse representation) to a vector.
     * This performs following matrix-vector multiplication \f$ \mathcal{\hat{P}}
     * \ket{\psi} \f$
     *
     * @tparam T The floating point base to use for all the complex numbers
     * @param new_states Output state
     * @param states The input vector to apply the PauliString to. Must be the
     * @param c Multiplication factor to apply to the PauliString
     * same size as PauliString.dim().
     */
    template <std::floating_point T>
    void apply(std::mdspan<std::complex<T>, std::dextents<size_t, 1>> new_states,
               std::mdspan<std::complex<T>, std::dextents<size_t, 1>> states, std::complex<T> const c = 1.0) const
    {
        apply(std::execution::seq, new_states, states, c);
    }

    /**
     * @brief \copydoc PauliString::apply(std::mdspan<std::complex<T>, std::dextents<size_t, 1>>,
     * std::mdspan<std::complex<T>, std::dextents<size_t, 1>>) const
     *
     * @tparam T
     * @tparam ExecutionPolicy
     * @param new_states
     * @param states
     */
    template <std::floating_point T, execution_policy ExecutionPolicy>
    void apply(ExecutionPolicy &&, std::mdspan<std::complex<T>, std::dextents<size_t, 1>> new_states,
               std::mdspan<std::complex<T>, std::dextents<size_t, 1>> states, std::complex<T> const c = 1.0) const
    {

        __input_checks_apply_1d(new_states, states);

        // WARNING: can't use structured bindings here because of a bug in LLVM
        // https://github.com/llvm/llvm-project/issues/63152
        std::vector<size_t> k;
        std::vector<std::complex<T>> m;
        std::tie(k, m) = get_sparse_repr<T>(paulis);

        if constexpr (is_parallel_execution_policy_v<ExecutionPolicy>)
        {

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < k.size(); ++i)
            {
                new_states[i] += c * m[i] * states[k[i]];
            }
        }
        else
        {
            for (size_t i = 0; i < k.size(); ++i)
            {
                new_states[i] += c * m[i] * states[k[i]];
            }
        }
    }

    template <std::floating_point T>
    void __input_checks_apply_batch(std::mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states_T,
                                    std::mdspan<std::complex<T>, std::dextents<size_t, 2>> const states_T) const
    {
        if (states_T.extent(0) != dim())
        {
            auto error_msg = fmt::format("[PauliString] states shape ({}) must match the "
                                         "dimension of the operators ({})",
                                         states_T.extent(0), dim());
            throw std::invalid_argument(error_msg);
        }

        if ((states_T.extent(0) != new_states_T.extent(0)) || states_T.extent(1) != new_states_T.extent(1))
        {
            throw std::invalid_argument("[PauliString] new_states must have the same dimensions as states");
        }
    }
    /**
     * @brief Apply the PauliString to a batch of states. This function takes a
     * different shape of the states than the other apply functions. here all the
     * states (new and old) are transposed so their shape is (n_dims x n_states).
     * All the new_stats are overwritten, no need to initialize.
     *
     * This performs following matrix-matrix multiplication \f$ \mathcal{\hat{P}}
     * \hat{\Psi} \f$ where matrix \f$ \hat{\Psi} \f$ has \f$ \ket{\psi_t} \f$ as
     * columns
     *
     * @tparam T The floating point base to use for all the complex numbers
     * @param new_states_T The output states after applying the PauliString
     * (n_dim x n_states)
     * @param states_T THe original states to apply the PauliString to
     * (n_dim x n_states)
     * @param c Multiplication factor to apply to the PauliString
     */
    template <std::floating_point T>
    void apply_batch(
        std::mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states_T,   // extent(0) = dims, extent(1) = n_states
        std::mdspan<std::complex<T>, std::dextents<size_t, 2>> const states_T, // extent(0) = dims, extent(1) = n_states
        std::complex<T> const c) const
    {
        apply_batch(std::execution::seq, new_states_T, states_T, c);
    }

    /**
     * @brief \copydoc PauliString::apply_batch(std::mdspan<std::complex<T>, std::dextents<size_t, 2>>,
     * std::mdspan<std::complex<T>, std::dextents<size_t, 2>>, std::complex<T> const) const
     *
     * @tparam T
     * @tparam ExecutionPolicy
     * @param new_states_T
     * @param states_T
     * @param c
     */
    template <std::floating_point T, execution_policy ExecutionPolicy>
    void apply_batch(ExecutionPolicy &&, std::mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states_T,
                     std::mdspan<std::complex<T>, std::dextents<size_t, 2>> const states_T,
                     std::complex<T> const c) const
    {

        __input_checks_apply_batch(new_states_T, states_T);

        // WARNING: can't use structugred bindings here because of a bug in LLVM
        // https://github.com/llvm/llvm-project/issues/63152
        std::vector<size_t> k;
        std::vector<std::complex<T>> m;
        std::tie(k, m) = get_sparse_repr<T>(paulis);

        if constexpr (is_parallel_execution_policy_v<ExecutionPolicy>)
        {

#pragma omp parallel for schedule(static) collapse(2)
            for (size_t i = 0; i < states_T.extent(0); ++i)
            {
                for (size_t t = 0; t < states_T.extent(1); ++t)
                {
                    std::complex<T> const c_m_i = c * m[i];
                    new_states_T(i, t) += c_m_i * states_T(k[i], t);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < states_T.extent(0); ++i)
            {
                std::complex<T> const c_m_i = c * m[i];
                std::mdspan<std::complex<T>, std::dextents<size_t, 1>> states_row =
                    std::submdspan(states_T, k[i], std::full_extent);
                for (size_t t = 0; t < states_T.extent(1); ++t)
                {
                    new_states_T(i, t) += c_m_i * states_row[t];
                }
            }
        }
    }

    template <std::floating_point T>
    void __input_checks_expectation_value(std::mdspan<std::complex<T>, std::dextents<size_t, 1>> expectation_vals_out,
                                          std::mdspan<std::complex<T>, std::dextents<size_t, 2>> states) const
    {
        // Input check
        if (states.extent(0) != dim())
            throw std::invalid_argument(fmt::format("[PauliString] states shape ({}) must match the dimension"
                                                    " of the operators ({})",
                                                    states.extent(0), dim()));
        if (expectation_vals_out.extent(0) != states.extent(1))
            throw std::invalid_argument("[PauliString] expectation_vals_out shape must "
                                        "match the number of states");
    }

    /**
     * @brief Calculate expectation values for a given batch of states.
     * This function takes in transposed states with (n_dims x n_states) shape
     *
     * It computes following inner product
     * \f$ \bra{\psi_t} \mathcal{\hat{P_i}} \ket{\psi_t} \f$
     * for each state \f$ \ket{\psi_t} \f$ from provided batch.
     *
     * @note The expectation values are added to corresponding coordinates
     * in the expectation_vals_out vector.
     *
     * @tparam T The floating point base to use for all the complex numbers
     * @param expectation_vals_out accumulator for expectation values for each
     * state in the batch
     * @param states THe original states to apply the PauliString to
     * (n_dim x n_states)
     * @param c Multiplication factor to apply to the PauliString
     */
    template <std::floating_point T>
    void expectation_value(std::mdspan<std::complex<T>, std::dextents<size_t, 1>> expectation_vals_out,
                           std::mdspan<std::complex<T>, std::dextents<size_t, 2>> states,
                           std::complex<T> const c = 1.0) const
    {
        expectation_value(std::execution::seq, expectation_vals_out, states, c);
    }

    /**
     * @brief \copydoc PauliString::expectation_value(std::mdspan<std::complex<T>, std::dextents<size_t, 1>>,
     * std::mdspan<std::complex<T>, std::dextents<size_t, 2>>, std::complex<T> const) const
     *
     * @tparam T
     * @tparam ExecutionPolicy
     * @param expectation_vals_out
     * @param states
     * @param c
     */
    template <std::floating_point T, execution_policy ExecutionPolicy>
    void expectation_value(ExecutionPolicy &&,
                           std::mdspan<std::complex<T>, std::dextents<size_t, 1>> expectation_vals_out,
                           std::mdspan<std::complex<T>, std::dextents<size_t, 2>> states,
                           std::complex<T> const c = 1.0) const
    {

        __input_checks_expectation_value(expectation_vals_out, states);

        std::vector<size_t> k;
        std::vector<std::complex<T>> m;
        std::tie(k, m) = get_sparse_repr<T>(paulis);

        if constexpr (is_parallel_execution_policy_v<ExecutionPolicy>)
        {
            size_t const n_threads = omp_get_max_threads();
            std::vector<std::complex<T>> expectation_vals_out_thread_raw(states.extent(1) * n_threads);
            std::mdspan expectation_vals_out_thread(expectation_vals_out_thread_raw.data(), states.extent(1),
                                                    n_threads);

#pragma omp parallel for schedule(static) collapse(2)
            for (size_t t = 0; t < states.extent(1); ++t)
            {
                for (size_t i = 0; i < states.extent(0); ++i)
                {
                    std::complex<T> const c_m_i = c * m[i];
                    size_t const thread_id = omp_get_thread_num();
                    expectation_vals_out_thread(t, thread_id) += std::conj(states(i, t)) * c_m_i * states(k[i], t);
                }
            }

            for (size_t t = 0; t < states.extent(1); ++t)
            {
                for (size_t thread_id = 0; thread_id < n_threads; ++thread_id)
                {
                    expectation_vals_out[t] += expectation_vals_out_thread(t, thread_id);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < states.extent(0); ++i)
            {
                std::complex<T> const c_m_i = c * m[i];
                for (size_t t = 0; t < states.extent(1); ++t)
                {
                    expectation_vals_out[t] += std::conj(states(i, t)) * c_m_i * states(k[i], t);
                }
            }
        }
    }

    //
    // Debugging Helpers
    //
    /**
     * @brief Get the dense representation of the object as a 2D-array
     *
     * @tparam T The floating point base to use for all the complex numbers
     * @param output The output tensor to fill with the dense representation
     */
    template <std::floating_point T> void to_tensor(std::mdspan<std::complex<T>, std::dextents<size_t, 2>> output) const
    {
        if (output.extent(0) != dim() or output.extent(1) != dim())
            throw std::invalid_argument("Output tensor must have the same dimensions as the PauliString");

        // TODO on the calling side: based on the output shape and num of cores we should decide if we invoke parallel
        // or serial version
        auto [k, m] = get_sparse_repr<T>(paulis);

        for (size_t i = 0; i < k.size(); ++i)
            output(i, k[i]) = m[i];
    }
};

} // namespace fast_pauli

//
// fmt::formatter specialization
//

//
template <> struct fmt::formatter<fast_pauli::PauliString>
{
    constexpr auto parse(format_parse_context &ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext> auto format(fast_pauli::PauliString const &ps, FormatContext &ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", fmt::join(ps.paulis, ""));
    }
};

//
// std::hash specialization
//
template <> struct std::hash<fast_pauli::PauliString>
{
    std::size_t operator()(fast_pauli::PauliString const &key) const
    {
        // this is pretty slow way to hash our PauliString. we might want to come
        // up with something more effiecient based on internal vector of uints
        return std::hash<std::string>()(fmt::format("{}", key));
    }
};

#endif // __PAULI_STRING_HPP
