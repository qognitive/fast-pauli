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


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "fast_pauli.hpp"

using namespace std::literals;
using namespace fast_pauli;

//
// Helpers
//

template <std::floating_point T> struct ComplexMatrix
{
    size_t n_rows;
    size_t n_cols;
    std::vector<std::complex<T>> data;

    ComplexMatrix(size_t n_rows, size_t n_cols) : n_rows(n_rows), n_cols(n_cols), data(n_rows * n_cols)
    {
    }

    ComplexMatrix(std::vector<std::vector<std::complex<T>>> const &matrix)
        : n_rows(matrix.size()), n_cols(matrix.at(0).size()), data(n_rows * n_cols)
    {
        for (size_t i = 0; i < n_rows; ++i)
            for (size_t j = 0; j < n_cols; ++j)
                data[i * n_cols + j] = matrix[i][j];
    }

    friend auto operator<=>(ComplexMatrix const &, ComplexMatrix const &) = default;

    std::complex<T> &operator()(size_t i, size_t j)
    {
        return data.at(i * n_cols + j);
    }

    std::span<std::complex<T>> operator[](size_t i)
    {
        return std::span(data.data() + i * n_cols, n_cols);
    }
    std::span<std::complex<T> const> operator[](size_t i) const
    {
        return std::span(data.data() + i * n_cols, n_cols);
    }

    size_t size() const
    {
        return n_rows;
    }

    auto mdspan()
    {
        return std::mdspan(data.data(), n_rows, n_cols);
    }
};

template <std::floating_point T> ComplexMatrix<T> tensor_prod(ComplexMatrix<T> const &A, ComplexMatrix<T> const &B)
{
    size_t const n = A.size();
    size_t const m = A[0].size();
    size_t const p = B.size();
    size_t const q = B[0].size();

    ComplexMatrix<T> C(n * p, (m * q));

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            for (size_t k = 0; k < p; ++k)
            {
                for (size_t l = 0; l < q; ++l)
                {
                    C[i * p + k][j * q + l] = A[i][j] * B[k][l];
                }
            }
        }
    }

    return C;
}

template <std::floating_point T> ComplexMatrix<T> paulistring_to_dense(PauliString const &ps)
{
    ComplexMatrix<double> res(1, 1);
    res(0, 0) = 1.0;

    for (auto const &p : ps.paulis | std::views::reverse)
    {
        ComplexMatrix<double> tmp(2, 2);
        p.to_tensor(tmp.mdspan());
        res = tensor_prod(tmp, res);
    }
    return res;
}

std::vector<std::complex<double>> dense_apply(PauliString const &ps, std::vector<std::complex<double>> const &state)
{
    ComplexMatrix<double> pauli_mat = paulistring_to_dense<double>(ps);
    std::vector<std::complex<double>> res(state.size(), 0);
    for (size_t i = 0; i < pauli_mat.size(); ++i)
    {
        for (size_t j = 0; j < pauli_mat[0].size(); ++j)
        {
            res[i] += pauli_mat[i][j] * state[j];
        }
    }
    return res;
}

TEST_CASE("check tensor_prod helper")
{
    ComplexMatrix<double> A({{1, 2}, {3, 4}});
    ComplexMatrix<double> B({{5, 6}, {7, 8}});
    ComplexMatrix<double> C = tensor_prod(A, B);

    // fmt::print("C: \n{}\n", fmt::join(C, "\n"));

    CHECK(C == ComplexMatrix<double>({{5, 6, 10, 12}, {7, 8, 14, 16}, {15, 18, 20, 24}, {21, 24, 28, 32}}));
}

//
// Tests
//

TEST_CASE("test pauli default init")
{
    std::vector<Pauli> paulis(10);
    PauliString ps(paulis);
    CHECK(ps.weight == 0);
    fmt::print("PauliString: {}\n", ps);
}

TEST_CASE("pauli string manual init")
{
    std::vector paulis{Pauli{0}, Pauli{1}, Pauli{2}, Pauli{3}};
    PauliString ps{paulis};
    CHECK(ps.weight == 3);
    fmt::print("PauliString: {}\n", ps);
}

TEST_CASE("string init")
{
    PauliString ps{"IXYZ"};
    CHECK(ps.weight == 3);
    fmt::print("PauliString: {}\n", ps);
}

TEST_CASE("getters")
{
    {
        PauliString ps{"I"};
        CHECK(ps.n_qubits() == 1);
        CHECK(ps.dim() == 2);
    }

    {
        PauliString ps{"IXYZI"};
        CHECK(ps.n_qubits() == 5);
        CHECK(ps.dim() == 32);
    }

    {
        PauliString ps{"IXYZ"};
        CHECK(ps.n_qubits() == 4);
        CHECK(ps.dim() == 16);
    }

    {
        PauliString ps;
        CHECK(ps.n_qubits() == 0);
        CHECK(ps.dim() == 0);
    }
}

TEST_CASE("test sparse repr")
{
    for (PauliString ps : {"IXYZ", "IX", "XXYIYZI"})
    {
        ComplexMatrix<double> res = paulistring_to_dense<double>(ps);
        ComplexMatrix<double> myres(ps.dim(), ps.dim());
        ps.to_tensor(myres.mdspan());

        // fmt::print("res: \n{}\n", fmt::join(res, "\n"));
        // fmt::print("myres: \n{}\n", fmt::join(myres, "\n"));

        // Check
        for (size_t i = 0; i < res.size(); ++i)
        {
            for (size_t j = 0; j < res[0].size(); ++j)
            {
                // CHECK(res[i][j] == myres[i][j]);
                CHECK(abs(res[i][j] - myres[i][j]) < 1e-6);
            }
        }
    }
}

TEST_CASE("test apply trivial")
{
    PauliString ps{"IIII"};

    std::vector<std::complex<double>> state(1 << ps.paulis.size(), 1.);

    std::vector<std::complex<double>> new_state;
    auto span_result = empty(new_state, state.size());
    ps.apply(span_result, std::mdspan(state.data(), state.size()));

    fmt::print("New state: \n[{}]\n", fmt::join(new_state, ",\n "));
    CHECK(new_state == state);
}

TEST_CASE("test apply simple")
{
    PauliString ps{"IXI"};

    std::vector<std::complex<double>> state(1 << ps.paulis.size(), 0);
    state[6] = 1.;
    state[7] = 1.;

    std::vector<std::complex<double>> new_state;
    auto span_result = empty(new_state, state.size());
    ps.apply(span_result, std::mdspan(state.data(), state.size()));

    fmt::print("New state: \n[{}]\n", fmt::join(new_state, ",\n "));

    auto expected = dense_apply(ps, state);
    CHECK(new_state == expected);
}

template <execution_policy ExecutionPolicy> void __test_apply_impl(ExecutionPolicy &&policy)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 2.0);

    for (PauliString ps : {"IXYZ", "YYIX", "XXYIYZ", "IZIXYYZ", "IZIXYYZIXYZ"})
    {
        std::vector<std::complex<double>> state(1 << ps.paulis.size());
        std::generate(state.begin(), state.end(), [&]() { return std::complex<double>(dis(gen), dis(gen)); });

        std::vector<std::complex<double>> new_state;
        auto span_result = empty(new_state, state.size());
        ps.apply(policy, span_result, std::mdspan(state.data(), state.size()));

        auto expected = dense_apply(ps, state);

        //
        for (size_t i = 0; i < new_state.size(); ++i)
        {
            CHECK(abs(new_state[i] - expected[i]) < 1e-6);
        }
    }
}

TEST_CASE("test apply")
{
    __test_apply_impl(std::execution::seq);
    __test_apply_impl(std::execution::par);
}

template <execution_policy ExecutionPolicy> void __test_apply_batch_impl(ExecutionPolicy &&policy)
{
    // For random states
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 2.0);

    size_t const n_states = 10;

    // Testing each of these pauli strings individually
    // NOTE: apply_batch takes the transpose of the states and new_states
    for (PauliString ps : {"IXYZ", "YYIX", "XXYIYZ", "IZIXYYZ"})
    {
        size_t const dims = ps.dim();

        std::vector<std::complex<double>> states_raw;
        std::mdspan states_T = rand<std::complex<double>, 2>(states_raw, {dims, n_states});

        std::vector<std::complex<double>> new_states_raw;
        std::mdspan new_states_T = zeros<std::complex<double>, 2>(new_states_raw, {dims, n_states});

        ps.apply_batch(policy, new_states_T, states_T, std::complex<double>(1.));

        // Calculate expected
        ComplexMatrix<double> ps_dense(ps.dim(), ps.dim());
        ps.to_tensor(ps_dense.mdspan()); // d x d
        std::vector<std::complex<double>> expected_raw;
        std::mdspan expected = zeros<std::complex<double>, 2>(expected_raw, {dims, n_states});

        for (size_t i = 0; i < (dims); ++i)
        {
            for (size_t t = 0; t < n_states; ++t)
            {
                for (size_t j = 0; j < (dims); ++j)
                {
                    expected(i, t) += ps_dense[i][j] * states_T(j, t);
                }
            }
        }

        // Check
        for (size_t t = 0; t < n_states; ++t)
        {
            for (size_t i = 0; i < dims; ++i)
            {
                CHECK(abs(new_states_T(i, t) - expected(i, t)) < 1e-6);
            }
        }
    }
}

TEST_CASE("test apply batch")
{
    __test_apply_batch_impl(std::execution::seq);
    __test_apply_batch_impl(std::execution::par);
}

template <execution_policy ExecutionPolicy> void _test_expectation_value_impl(ExecutionPolicy &&policy)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 2.0);

    size_t const n_states = 10;

    for (PauliString ps : {"IXYZ", "YYIX", "XXYIYZ", "IZIXYYZ"})
    {
        size_t const dims = ps.dim();

        std::vector<std::complex<double>> states_raw;
        std::mdspan states_T = rand<std::complex<double>, 2>(states_raw, {dims, n_states});

        std::vector<std::complex<double>> expectation_vals_raw;
        std::mdspan expectation_vals = zeros<std::complex<double>, 1>(expectation_vals_raw, {n_states});

        ps.expectation_value(policy, expectation_vals, states_T);

        // Calculate true expectation values
        ComplexMatrix<double> ps_dense(ps.dim(), ps.dim());
        ps.to_tensor(ps_dense.mdspan()); // d x d
        std::vector<std::complex<double>> expected_vals_true_raw;
        std::mdspan expected_vals_true = zeros<std::complex<double>, 1>(expected_vals_true_raw, {n_states});

        for (size_t t = 0; t < n_states; ++t)
        {
            //
            for (size_t i = 0; i < dims; ++i)
            {
                for (size_t j = 0; j < dims; ++j)
                {
                    expected_vals_true(t) += std::conj(states_T(i, t)) * ps_dense(i, j) * states_T(j, t);
                }
            }
        }

        // Check
        for (size_t t = 0; t < n_states; ++t)
        {
            CHECK(abs(expectation_vals(t) - expected_vals_true(t)) < 1e-6);
        }
    }
}

TEST_CASE("test expectation value")
{
    _test_expectation_value_impl(std::execution::seq);
    _test_expectation_value_impl(std::execution::par);
}