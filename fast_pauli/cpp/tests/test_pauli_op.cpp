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

void __check_apply(PauliOp<double> &pauli_op, size_t n_states)
{
    size_t const dims = pauli_op.dim();

    // Set up random states
    std::vector<std::complex<double>> states_raw;
    std::mdspan states = fast_pauli::rand<std::complex<double>, 2>(states_raw, {dims, n_states});
    // fmt::println("states: \n{}\n", fmt::join(states_raw, ",\n "));

    // Apply the PauliOp to a batch of states
    std::vector<std::complex<double>> new_states_raw(dims * n_states, 0);
    std::mdspan<std::complex<double>, std::dextents<size_t, 2>> new_states(new_states_raw.data(), dims, n_states);
    pauli_op.apply(new_states, states);
    // fmt::println("new_states: \n{}\n", fmt::join(new_states_raw, ",\n "));

    //
    // Calculate the expected new states
    //
    std::vector<std::complex<double>> dense_raw;
    auto pop_dense = empty(dense_raw, dims, dims);
    pauli_op.to_tensor(pop_dense);

    // pop_dense : d x d
    // states : n x d
    // expected_states : n x d

    std::vector<std::complex<double>> expected(dims * n_states, 0);
    std::mdspan<std::complex<double>, std::dextents<size_t, 2>> expected_span(expected.data(), dims, n_states);

    // Calculate the expected new states
    for (size_t t = 0; t < n_states; ++t)
    {
        for (size_t i = 0; i < dims; ++i)
        {
            for (size_t j = 0; j < dims; ++j)
            {
                expected_span(i, t) += pop_dense(i, j) * states(j, t);
            }
        }
    }

    // Check
    for (size_t t = 0; t < n_states; ++t)
    {
        for (size_t i = 0; i < dims; ++i)
        {
            CHECK(abs(new_states(i, t) - expected_span(i, t)) < 1e-6);
        }
    }
}

//
// Tests
//

//
// Constructors
//

TEST_CASE("test pauli op")
{
    std::vector<PauliString> pauli_strings = {"IXYZ", "XIII", "XXYZ"};
    std::vector<std::complex<double>> coeffs = {1i, 2., -1i};
    PauliOp<double> pauli_op(coeffs, pauli_strings);

    std::vector<std::complex<double>> dense_raw;
    auto pop_dense = empty(dense_raw, pauli_op.dim(), pauli_op.dim());
    pauli_op.to_tensor(pop_dense);

    fmt::print("pop_dense: \n{}\n", fmt::join(dense_raw, "\n"));
}

TEST_CASE("test bad init")
{
    CHECK_THROWS(PauliOp<double>({1i, 2.}, {"IXYZ", "XII", "XXYZ"}));
}

TEST_CASE("test bad apply")
{
    std::vector<PauliString> pauli_strings = {"IXYZ", "XIII", "XXYZ"};
    std::vector<std::complex<double>> coeffs = {1i, 2., -1i};
    std::vector<std::complex<double>> state(5, 0);
    PauliOp<double> pauli_op(coeffs, pauli_strings);

    std::vector<std::complex<double>> result;
    auto span_result = empty(result, state.size());

    CHECK_THROWS(pauli_op.apply(span_result, std::mdspan(state.data(), state.size())));
}

//
// Member functions
//

TEST_CASE("test get_dense_repr")
{
    std::vector<PauliString> pauli_strings = {"III", "III", "III"};
    std::vector<std::complex<double>> coeffs = {1, 2., 1};
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    size_t const dim = pauli_op.dim();

    std::vector<std::complex<double>> dense_raw;
    auto pop_dense = empty(dense_raw, dim, dim);
    pauli_op.to_tensor(pop_dense);

    for (size_t i = 0; i < dim; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            if (i == j)
            {
                CHECK(abs(pop_dense(i, j) - std::complex<double>(4)) < 1e-6);
            }
            else
            {
                CHECK(abs(pop_dense(i, j)) < 1e-6);
            }
        }
    }
}

// TODO add tests for multiple pauli strings
TEST_CASE("test apply simple")
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings = {"IXYZ"};
    std::vector<std::complex<double>> coeffs = {1};
    PauliOp<double> pauli_op(coeffs, pauli_strings);

    // Set up random state
    std::vector<std::complex<double>> state = {0.6514 + 0.8887i, 0.0903 + 0.8848i, 0.8130 + 0.6729i, 0.9854 + 0.9497i,
                                               0.0410 + 0.5132i, 0.7920 + 0.6519i, 0.4683 + 0.5708i, 0.2148 + 0.4561i,
                                               0.5400 + 0.9932i, 0.9736 + 0.0449i, 0.2979 + 0.9834i, 0.4346 + 0.5635i,
                                               0.0767 + 0.3516i, 0.4873 + 0.4097i, 0.1543 + 0.7329i, 0.4121 + 0.5059i};

    fmt::print("state: \n[{}]\n", fmt::join(state, ",\n "));

    // Apply the PauliOp and the PauliString
    std::vector<std::complex<double>> res, expected;
    auto span_result = empty(res, state.size());
    auto span_expected = empty(expected, state.size());

    pauli_op.apply(span_result, std::mdspan(state.data(), state.size()));
    pauli_strings[0].apply(span_expected, std::mdspan(state.data(), state.size()));

    fmt::print("res: \n[{}]\n", fmt::join(res, ",\n "));
    fmt::print("expected: \n[{}]\n", fmt::join(expected, ",\n "));

    for (size_t i = 0; i < res.size(); ++i)
    {
        CHECK(abs(res[i] - expected[i]) < 1e-6);
    }
}

TEST_CASE("test apply single state")
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings = {"IXYZ"};
    std::vector<std::complex<double>> coeffs = {1i};
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    __check_apply(pauli_op, 1);
}

TEST_CASE("test apply two states")
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings = {"IXYZ"};
    std::vector<std::complex<double>> coeffs = {1i};
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    __check_apply(pauli_op, 2);
}

TEST_CASE("test apply multistate")
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings = {"IXYZ"};
    std::vector<std::complex<double>> coeffs = {1i};
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    __check_apply(pauli_op, 10);
}

TEST_CASE("test apply single state multistring")
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings = {"XXY", "YYZ"};
    std::vector<std::complex<double>> coeffs = {1i, -1.23};
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    __check_apply(pauli_op, 1);
}

TEST_CASE("test apply multistate multistring")
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings = {"IXXZYZ", "XXXXZX", "YZZXZZ", "ZXZIII", "IIIXZI", "ZZXZYY"};
    std::vector<std::complex<double>> coeffs = {1i, -2., 42i, 0.5, 1, 0.1};
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    __check_apply(pauli_op, 10);
}

template <execution_policy ExecutionPolicy> void __test_apply_impl(ExecutionPolicy &&policy)
{
    // Set up PauliOp
    std::vector<PauliString> pauli_strings(16, "IIII");
    std::vector<std::complex<double>> coeffs(pauli_strings.size(), 1. / pauli_strings.size());
    PauliOp<double> pauli_op(coeffs, pauli_strings);
    size_t const dims = pauli_strings[0].dim();

    // Set up random states
    size_t const n_states = 10;
    std::vector<std::complex<double>> states_raw;
    std::mdspan states = fast_pauli::rand<std::complex<double>, 2>(states_raw, {dims, n_states});

    // Apply the PauliOp to a batch of states
    std::vector<std::complex<double>> new_states_raw(dims * n_states, 0);
    std::mdspan<std::complex<double>, std::dextents<size_t, 2>> new_states(new_states_raw.data(), dims, n_states);
    pauli_op.apply(policy, new_states, states);

    // States should be unchanged
    // Check
    for (size_t t = 0; t < n_states; ++t)
    {
        for (size_t i = 0; i < dims; ++i)
        {
            CHECK(abs(new_states(i, t) - states(i, t)) < 1e-6);
            if (abs(new_states(i, t) - states(i, t)) > 1e-6)
            {
                fmt::print("new_states(i, t): {}, states(i, t): {}", new_states(i, t), states(i, t));
                fmt::println(" ratio {}", new_states(i, t) / states(i, t));
            }
        }
    }
}

TEST_CASE("test apply multistate multistring identity")
{
    fmt::println("test apply multistate multistring identity");
    __test_apply_impl(std::execution::seq);
    __test_apply_impl(std::execution::par);
}

template <execution_policy ExecutionPolicy>
void __check_exp_vals_batch(ExecutionPolicy &&policy, size_t const n_qubits, size_t const n_states)
{
    std::vector<PauliString> pauli_strings = fast_pauli::calculate_pauli_strings_max_weight(n_qubits, 2);

    std::vector<std::complex<double>> coeff_raw;
    fast_pauli::rand<std::complex<double>, 1>(coeff_raw, {pauli_strings.size()});

    PauliOp<double> op(coeff_raw, pauli_strings);

    std::vector<std::complex<double>> states_raw;
    std::mdspan states = fast_pauli::rand<std::complex<double>, 2>(states_raw, {1UL << n_qubits, n_states});

    std::vector<std::complex<double>> expected_vals_raw;
    std::mdspan expected_vals = fast_pauli::zeros<std::complex<double>, 1>(expected_vals_raw, {n_states});

    op.expectation_value(policy, expected_vals, states);

    //
    // Construct "trusted answer"
    //

    std::vector<std::complex<double>> op_dense_raw;
    std::mdspan op_dense = fast_pauli::zeros<std::complex<double>, 2>(op_dense_raw, {1UL << n_qubits, 1UL << n_qubits});
    op.to_tensor(op_dense);

    std::vector<std::complex<double>> expected_vals_check_raw;

    std::mdspan expected_vals_check = fast_pauli::zeros<std::complex<double>, 1>(expected_vals_check_raw, {n_states});

    for (size_t t = 0; t < n_states; ++t)
    {
        for (size_t i = 0; i < (1UL << n_qubits); ++i)
        {
            for (size_t j = 0; j < (1UL << n_qubits); ++j)
            {
                expected_vals_check(t) += std::conj(states(i, t)) * op_dense(i, j) * states(j, t);
            }
        }
    }

    // Check
    for (size_t t = 0; t < n_states; ++t)
    {
        CHECK(abs(expected_vals(t) - expected_vals_check(t)) < 1e-6);
    }
}

TEST_CASE("Test expectation values 1 qubit")
{
    __check_exp_vals_batch(std::execution::seq, 1, 10);
    __check_exp_vals_batch(std::execution::par, 1, 10);
}

TEST_CASE("Test expectation values 10 qubits")
{
    __check_exp_vals_batch(std::execution::seq, 10, 10);
    __check_exp_vals_batch(std::execution::par, 10, 10);
}
