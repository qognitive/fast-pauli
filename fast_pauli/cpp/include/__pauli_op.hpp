#ifndef __PAULI_OP_HPP
#define __PAULI_OP_HPP

#include <omp.h>

#include <algorithm>
#include <experimental/mdspan> // From Kokkos
#include <unordered_map>

#include "__pauli_string.hpp"

using namespace std::experimental;

namespace fast_pauli
{

/**
 * @brief A class representation for a Pauli Operator (i.e. a weighted sum of
 * Pauli Strings) \f$ \big( \sum_i h_i \mathcal{\hat{P}}_i \big) \f$ where \f$
 * \mathcal{\hat{P}}_i \f$ are composed using \f$ \sigma_i \in \{ I,X,Y,Z \} \f$
 * and \f$ h_i \f$ are the coefficients.
 *
 */
template <std::floating_point T, typename H = std::complex<T>> struct PauliOp
{
    std::vector<H> coeffs;
    std::vector<PauliString> pauli_strings;
    // TODO NEED TO THINK ABOUT THE ORDER HERE
    // DO WE ASSUME PEOPLE WANT IT COMPLETE? (if weight 3, do we include all
    // possible combinations up to and including weight 3 strings?)

    /**
     * @brief Default constructor, initialize empty vectors for paulis and
     * coefficients.
     *
     */
    PauliOp() = default;

    /**
     * @brief Construct a PauliOp from a vector of strings and default
     * corresponding coeffs to ones.
     *
     * @param strings vector of strings
     */
    PauliOp(std::vector<std::string> const &strings)
    {
        for (auto const &s : strings)
        {
            pauli_strings.push_back(PauliString(s));
        }
        coeffs.resize(pauli_strings.size(), 1.0);
        validate_pauli_strings(this->pauli_strings);
    }

    /**
     * @brief Construct a PauliOp from a vector of PauliStrings and
     * default corresponding coeffs to ones.
     *
     * @param strings vector of PauliString objects
     */
    PauliOp(std::vector<PauliString> strings) : coeffs(strings.size(), 1.0), pauli_strings(std::move(strings))
    // note that strings are moved after coeffs initialization
    // according to the order of data member declarations in the class
    {
        validate_pauli_strings(this->pauli_strings);
    }

    /**
     * @brief Construct a PauliOp from a vector of PauliStrings and
     * corresponding coefficients.
     *
     * @param coefficients vector of coefficients
     * @param strings vector of PauliString objects
     */
    PauliOp(std::vector<H> coefficients, std::vector<PauliString> strings)
        : coeffs(std::move(coefficients)), pauli_strings(std::move(strings))
    {
        if (coeffs.size() != pauli_strings.size())
        {
            throw std::invalid_argument("coeffs and pauli_strings must have the same size");
        }
        validate_pauli_strings(this->pauli_strings);
    }

    /**
     * @brief Return the dimension (2^n_qubits) of the PauliStrings
     * used to compose PauliOp.
     *
     * @return  size_t
     */
    size_t dim() const
    {
        if (pauli_strings.size() > 0)
        {
            return pauli_strings[0].dim();
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Return the number of qubits in PauliOp.
     *
     * @return  size_t
     */
    size_t n_qubits() const
    {
        return pauli_strings.size() ? pauli_strings[0].n_qubits() : 0;
    }

    /**
     * @brief Return the number of PauliStrings in PauliOp.
     *
     * @return  size_t
     */
    size_t n_pauli_strings() const
    {
        return pauli_strings.size();
    }

    /**
     * @brief Return the copy of PauliOp with the coefficients negated.
     *
     * @return PauliOp<T, H>
     */
    PauliOp<T, H> operator-() const
    {
        std::vector<H> neg_coeffs;
        neg_coeffs.reserve(coeffs.size());
        std::transform(coeffs.begin(), coeffs.end(), std::back_inserter(neg_coeffs), [](auto const &c) { return -c; });
        return PauliOp<T, H>(std::move(neg_coeffs), pauli_strings);
    }

    /**
     * @brief Matrix multiplication of PauliOp with a PauliString on the right.
     *
     * @param pauli_op_left left hand side PauliOp
     * @param pauli_str_right right hand side PauliString
     * @return PauliOp<T, H> new PauliOp instance containing the result of the
     * multiplication
     */
    friend PauliOp<T, H> operator*(PauliOp<T, H> const &pauli_op_left, PauliString const &pauli_str_right)
    {
        // TODO figure out if it's possible to end up with duplicate strings and
        // need to dedupe
        if (pauli_op_left.dim() != pauli_str_right.dim())
            throw std::invalid_argument("PauliStrings must have same size as PauliOp");

        std::vector<H> coefficients(pauli_op_left.coeffs);
        std::vector<PauliString> strings;
        strings.reserve(pauli_op_left.n_pauli_strings());

        for (size_t i = 0; i < pauli_op_left.n_pauli_strings(); ++i)
        {
            auto [phase, pauli_str] = pauli_op_left.pauli_strings[i] * pauli_str_right;
            coefficients[i] *= phase;
            strings.push_back(std::move(pauli_str));
        }

        return PauliOp<T, H>(std::move(coefficients), std::move(strings));
    }

    /**
     * @brief Matrix multiplication of PauliOp with a PauliString on the left.
     *
     * @param pauli_str_left left hand side PauliString
     * @param pauli_op_right right hand side PauliOp
     * @return PauliOp<T, H> new PauliOp instance containing the result of the
     * multiplication
     */
    friend PauliOp<T, H> operator*(PauliString const &pauli_str_left, PauliOp<T, H> const &pauli_op_right)
    {
        if (pauli_str_left.dim() != pauli_op_right.dim())
            throw std::invalid_argument("PauliStrings must have same size as PauliOp");

        std::vector<H> coefficients(pauli_op_right.coeffs);
        std::vector<PauliString> strings;
        strings.reserve(pauli_op_right.n_pauli_strings());

        for (size_t i = 0; i < pauli_op_right.n_pauli_strings(); ++i)
        {
            auto [phase, pauli_str] = pauli_str_left * pauli_op_right.pauli_strings[i];
            coefficients[i] *= phase;
            strings.push_back(std::move(pauli_str));
        }

        return PauliOp<T, H>(std::move(coefficients), std::move(strings));
    }

    /**
     * @brief Matrix multiplication of two Pauli operators.
     *
     * @param lhs left hand side PauliOp
     * @param rhs right hand side PauliOp
     * @return PauliOp<T, H> new PauliOp instance containing the result of the
     * multiplication
     */
    friend PauliOp<T, H> operator*(PauliOp<T, H> const &lhs, PauliOp<T, H> const &rhs)
    {
        if (lhs.dim() != rhs.dim())
            throw std::invalid_argument("Mismatched dimensions for provided PauliOp");

        size_t init_capacity = std::max(lhs.n_pauli_strings(), rhs.n_pauli_strings());
        std::vector<H> coefficients;
        std::vector<PauliString> strings;
        coefficients.reserve(init_capacity);
        strings.reserve(init_capacity);

        std::unordered_map<PauliString, size_t> dedupe_strings;

        for (size_t i = 0; i < lhs.n_pauli_strings(); ++i)
        {
            for (size_t j = 0; j < rhs.n_pauli_strings(); ++j)
            {
                auto [phase, pauli_str] = lhs.pauli_strings[i] * rhs.pauli_strings[j];
                auto coeff_ij = phase * lhs.coeffs[i] * rhs.coeffs[j];

                if (dedupe_strings.contains(pauli_str))
                {
                    size_t idx = dedupe_strings[pauli_str];
                    coefficients[idx] += coeff_ij;
                }
                else
                {
                    dedupe_strings[pauli_str] = coefficients.size();
                    coefficients.push_back(coeff_ij);
                    strings.push_back(std::move(pauli_str));
                }
            }
        }

        return PauliOp<T, H>(std::move(coefficients), std::move(strings));
    }

    /**
     * @brief Add a PauliString term with appropriate coefficient
     * to the summation inside PauliOp.
     *
     * @param pauli_str PauliString to add to the summation
     * @param coeff coefficient to apply to the PauliString
     * @param dedupe whether to deduplicate provided PauliString
     */
    void extend(PauliString pauli_str, std::complex<T> coeff, bool dedupe = true)
    {
        if (pauli_str.dim() != dim())
        {
            throw std::invalid_argument("PauliStrings must have same size as PauliOp");
        }

        if (dedupe)
        {
            for (size_t i = 0; i < pauli_strings.size(); ++i)
            {
                if (pauli_strings[i] == pauli_str)
                {
                    coeffs[i] += coeff;
                    // finish immediately because we don't want to add same string twice
                    return;
                }
            }
        }

        coeffs.push_back(coeff);
        pauli_strings.push_back(std::move(pauli_str));
    }

    /**
     * @brief Add another PauliOp to the current one
     * by extending the internal summation with new terms.
     *
     * @note: for now it's very sloppy implementation just to have this
     * functionality
     * @param other_op PauliOp to add to the current one
     */
    void extend(PauliOp<T, H> const &other_op)
    {
        // TODO add dedupe capabilities once we have pauli_strings stored in
        // lexicographic order
        // With current implementation it would take O(N*M) to dedupe
        // TODO handle the case when other_op is *this object
        if (other_op.dim() != dim())
            throw std::invalid_argument("Mismatched dimensions for provided PauliOp");
        size_t naive_size = n_pauli_strings() + other_op.n_pauli_strings();

        pauli_strings.reserve(naive_size);
        std::copy(other_op.pauli_strings.begin(), other_op.pauli_strings.end(), std::back_inserter(pauli_strings));

        coeffs.reserve(naive_size);
        std::copy(other_op.coeffs.begin(), other_op.coeffs.end(), std::back_inserter(coeffs));
    }

    std::vector<std::complex<T>> apply(std::vector<std::complex<T>> const &state) const
    {
        // input check
        if (state.size() != dim())
        {
            throw std::invalid_argument("state size must match the dimension of the operators");
        }

        std::vector<std::complex<T>> res(state.size(), 0);

        for (size_t i = 0; i < pauli_strings.size(); ++i)
        {
            PauliString const &ps = pauli_strings[i];
            std::complex<T> c = coeffs[i];
            std::vector<std::complex<T>> tmp = ps.apply(state);
            for (size_t j = 0; j < state.size(); ++j)
            {
                res[j] += c * tmp[j];
            }
        }
        return res;
    }

    /**
     * @brief Apply the PauliOp to a batch of states. This function takes a
     * different shape of the states than the other apply functions. Here all the
     * states (new and old) have shape of (n_states x n_dim).
     *
     * @note: the states are expected to be in row-major order for this method
     *
     * @param new_states The output states after applying the PauliOp
     * (n_states x n_dim)
     * @param states The original states to apply the PauliOp to
     * (n_states x n_dim)
     */
    void apply_naive(mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states,
                     mdspan<std::complex<T>, std::dextents<size_t, 2>> const states) const
    {
        // input check
        if (states.extent(1) != dim())
        {
            throw std::invalid_argument("state size must match the dimension of the operators");
        }
        if (states.extent(0) != new_states.extent(0) || states.extent(1) != new_states.extent(1))
        {
            throw std::invalid_argument("new_states must have the same dimensions as states");
        }

        size_t const n_states = states.extent(0);
#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < n_states; ++t)
        {
            for (size_t i = 0; i < pauli_strings.size(); ++i)
            {
                PauliString const &ps = pauli_strings[i];
                std::complex<T> c = coeffs[i];

                std::mdspan state_t = std::submdspan(states, t, std::full_extent);
                std::vector<std::complex<T>> tmp = ps.apply(state_t);
                // we intentionally do T*N sparse decompositions here instead of just N
                for (size_t j = 0; j < states.extent(1); ++j)
                {
                    new_states(t, j) += c * tmp[j];
                }
            }
        }
        return;
    }

    void __check_apply_inputs(mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states,
                              mdspan<std::complex<T>, std::dextents<size_t, 2>> states) const
    {
        if (states.extent(0) != this->dim())
        {
            throw std::invalid_argument("[PauliOp] state size must match the dimension of the operators");
        }
        if (states.extent(0) != new_states.extent(0) || states.extent(1) != new_states.extent(1))
        {
            throw std::invalid_argument("[PauliOp] new_states must have the same dimensions as states");
        }
    }

    /**
     * @brief Apply the PauliOp to a batch of states. Here all the
     * states (new and old) are transposed so their shape is (n_dims x n_states).
     * All the new_stats are overwritten, no need to initialize.
     *
     * This performs following matrix-matrix multiplication
     * \f$ \big( \sum_i h_i \mathcal{\hat{P}}_i \big) \hat{\Psi} \f$
     * where matrix \f$ \hat{\Psi} \f$ has \f$ \ket{\psi_t} \f$ as columns
     *
     * @param new_states The output states after applying the PauliOp
     * (n_dim x n_states)
     * @param states THe original states to apply the PauliOp to
     * (n_dim x n_states)
     */
    void apply(mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states,
               mdspan<std::complex<T>, std::dextents<size_t, 2>> const states) const
    {
        __check_apply_inputs(new_states, states);

        // Create tmp obj for reduction
        size_t const n_threads = omp_get_max_threads();
        size_t const n_data = states.extent(1);
        size_t const n_dim = states.extent(0);

        std::vector<std::complex<T>> new_states_thr_raw(n_threads * n_dim * n_data);
        std::mdspan<std::complex<T>, std::dextents<size_t, 3>> new_states_thr(new_states_thr_raw.data(), n_threads,
                                                                              n_dim, n_data);

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (size_t i = 0; i < pauli_strings.size(); ++i)
            {
                size_t const tid = omp_get_thread_num();

                PauliString const &ps = pauli_strings[i];
                std::complex<T> c = coeffs[i];
                std::mdspan<std::complex<T>, std::dextents<size_t, 2>> new_states_local =
                    std::submdspan(new_states_thr, tid, std::full_extent, std::full_extent);
                ps.apply_batch(new_states_local, states, c);
            }

            // Do the reduction
#pragma omp for schedule(static) collapse(2)
            for (size_t i = 0; i < new_states.extent(0); ++i)
            {
                for (size_t t = 0; t < new_states.extent(1); ++t)
                {
                    for (size_t th = 0; th < n_threads; ++th)
                    {
                        new_states(i, t) += new_states_thr(th, i, t);
                    }
                }
            }
        }
    }

    /**
     * @brief Calculate the expectation value of the PauliOp on a batch of states.
     *
     * It computes following inner product
     * \f$ \bra{\psi_t} ( \sum_i h_{ik} \mathcal{\hat{P}}_i ) \ket{\psi_t} \f$
     * for each state \f$ \ket{\psi_t} \f$ from provided batch.
     *
     * @param expectation_vals_out expectation values for each state in
     * the batch
     * @param states The states we want to use in our expectation value
     * calculation (n_dim x n_states)
     */
    void expectation_value(std::mdspan<std::complex<T>, std::dextents<size_t, 1>> expectation_vals_out,
                           mdspan<std::complex<T>, std::dextents<size_t, 2>> states) const
    {
        // input check
        if (states.extent(0) != this->dim())
        {
            throw std::invalid_argument("[PauliOp] state size must match the dimension of the operators");
        }

        size_t const n_data = states.extent(1);
        size_t const n_threads = omp_get_max_threads();

        // no need to default initialize with 0 since std::complex constructor
        // handles that
        std::vector<std::complex<T>> expected_vals_per_thread_storage(n_threads * n_data);
        std::mdspan<std::complex<T>, std::dextents<size_t, 2>> exp_vals_accum_per_thread(
            expected_vals_per_thread_storage.data(), n_threads, n_data);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pauli_strings.size(); ++i)
        {
            size_t const tid = omp_get_thread_num();

            PauliString const &ps = pauli_strings[i];
            std::complex<T> c = coeffs[i];
            std::mdspan<std::complex<T>, std::dextents<size_t, 1>> exp_vals_accum_local =
                std::submdspan(exp_vals_accum_per_thread, tid, std::full_extent);
            ps.expectation_value(exp_vals_accum_local, states, c);
        }

#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < states.extent(1); ++t)
        {
            for (size_t th = 0; th < n_threads; ++th)
            {
                expectation_vals_out[t] += exp_vals_accum_per_thread(th, t);
            }
        }
    }

    //
    // Helpers (mostly for debugging)
    //
    /**
     * @brief Get dense representation of PauliOp as a 2D-std::vector
     *
     * @return std::vector<std::vector<std::complex<T>>>
     */
    std::vector<std::vector<std::complex<T>>> get_dense_repr() const
    {
        std::vector<std::vector<std::complex<T>>> res(dim(), std::vector<std::complex<T>>(dim(), 0));

        for (size_t i = 0; i < pauli_strings.size(); ++i)
        {
            PauliString const &ps = pauli_strings[i];
            std::complex<T> c = coeffs[i];

            auto [cols, vals] = get_sparse_repr<T>(ps.paulis);
            for (size_t j = 0; j < dim(); ++j)
            {
                res[j][cols[j]] += c * vals[j];
            }
        }
        return res;
    }

    /**
     * @brief Check that the dims of pauli strings are all the same
     *
     * @param pauli_strings
     */
    static inline void validate_pauli_strings(std::vector<PauliString> const &pauli_strings)
    {
        if (pauli_strings.size() > 0)
        {
            size_t const n_qubits = pauli_strings[0].n_qubits();
            bool const qubits_match =
                std::all_of(pauli_strings.begin(), pauli_strings.end(),
                            [n_qubits](PauliString const &ps) { return ps.n_qubits() == n_qubits; });
            if (!qubits_match)
            {
                throw std::invalid_argument("All PauliStrings must have the same size");
            }
        }
    }
};

} // namespace fast_pauli

#endif // __PAULI_OP_HPP
