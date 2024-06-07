#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "fast_pauli.hpp"

using namespace std::literals;
using namespace fast_pauli;

TEST_CASE("test pauli op") {
  std::vector<PauliString> pauli_strings = {"IXYZ", "XIII", "XXYZ"};
  std::vector<std::complex<double>> coeffs = {1i, 2., -1i};

  PauliOp<double> pauli_op(coeffs, pauli_strings);
  auto pop_dense = pauli_op.get_dense_repr();

  fmt::print("pop_dense: \n{}\n", fmt::join(pop_dense, "\n"));
}

TEST_CASE("test bad init") {
  CHECK_THROWS(PauliOp<double>({1i, 2.}, {"IXYZ", "XII", "XXYZ"}));
}

TEST_CASE("test bad apply") {
  std::vector<PauliString> pauli_strings = {"IXYZ", "XIII", "XXYZ"};
  std::vector<std::complex<double>> coeffs = {1i, 2., -1i};
  std::vector<std::complex<double>> state(5, 0);
  PauliOp<double> pauli_op(coeffs, pauli_strings);
  CHECK_THROWS(pauli_op.apply(state));
}

// TODO add tests for multiple pauli strings
TEST_CASE("test apply simple") {
  // Set up PauliOp
  std::vector<PauliString> pauli_strings = {"IXYZ"};
  std::vector<std::complex<double>> coeffs = {1};
  PauliOp<double> pauli_op(coeffs, pauli_strings);

  // Set up random state
  std::vector<std::complex<double>> state = {
      0.6514 + 0.8887i, 0.0903 + 0.8848i, 0.8130 + 0.6729i, 0.9854 + 0.9497i,
      0.0410 + 0.5132i, 0.7920 + 0.6519i, 0.4683 + 0.5708i, 0.2148 + 0.4561i,
      0.5400 + 0.9932i, 0.9736 + 0.0449i, 0.2979 + 0.9834i, 0.4346 + 0.5635i,
      0.0767 + 0.3516i, 0.4873 + 0.4097i, 0.1543 + 0.7329i, 0.4121 + 0.5059i};

  fmt::print("state: \n[{}]\n", fmt::join(state, ",\n "));

  // Apply the PauliOp and the PauliString
  std::vector<std::complex<double>> res = pauli_op.apply(state);
  std::vector<std::complex<double>> expected = pauli_strings[0].apply(state);

  fmt::print("res: \n[{}]\n", fmt::join(res, ",\n "));
  fmt::print("expected: \n[{}]\n", fmt::join(expected, ",\n "));

  for (size_t i = 0; i < res.size(); ++i) {
    CHECK(abs(res[i] - expected[i]) < 1e-6);
  }
}

TEST_CASE("test apply multistate") {
  // Set up PauliOp
  std::vector<PauliString> pauli_strings = {"IXYZ"};
  std::vector<std::complex<double>> coeffs = {1i};
  PauliOp<double> pauli_op(coeffs, pauli_strings);

  size_t const dims = pauli_strings[0].dims();

  // Set up random states
  size_t const n_states = 10; // TODO change back to more than 1
  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dims, n_states});

  // Apply the PauliOp to a batch of states
  std::vector<std::complex<double>> new_states_raw(dims * n_states, 0);
  std::mdspan<std::complex<double>, std::dextents<size_t, 2>> new_states(
      new_states_raw.data(), dims, n_states);
  pauli_op.apply(new_states, states);

  //
  // Calculate the expected new states
  //
  auto pop_dense = pauli_op.get_dense_repr();

  // pop_dense : d x d
  // states : n x d
  // expected_states : n x d

  std::vector<std::complex<double>> expected(dims * n_states, 0);
  std::mdspan<std::complex<double>, std::dextents<size_t, 2>> expected_span(
      expected.data(), dims, n_states);

  // Calculate the expected new states
  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dims; ++i) {
      for (size_t j = 0; j < dims; ++j) {
        expected_span(i, t) += pop_dense[i][j] * states(j, t);
      }
    }
  }

  // Check
  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dims; ++i) {
      CHECK(abs(new_states(i, t) - expected_span(i, t)) < 1e-6);
    }
  }
}

TEST_CASE("test apply multistate multistring") {
  // Set up PauliOp
  std::vector<PauliString> pauli_strings = {"IXXZYZ", "XXXXZX", "YZZXZZ",
                                            "ZXZIII", "IIIXZI", "ZZXZYY"};
  std::vector<std::complex<double>> coeffs = {1i, -2., 42i, 0.5, 1, 0.1};
  // std::vector<PauliString> pauli_strings = {"IXYZ", "YYIZ"};
  // std::vector<std::complex<double>> coeffs = {1, 1};
  PauliOp<double> pauli_op(coeffs, pauli_strings);

  size_t const dims = pauli_strings[0].dims();

  // Set up random states
  size_t const n_states = 10;
  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dims, n_states});

  // Apply the PauliOp to a batch of states
  std::vector<std::complex<double>> new_states_raw(dims * n_states, 0);
  std::mdspan<std::complex<double>, std::dextents<size_t, 2>> new_states(
      new_states_raw.data(), dims, n_states);
  pauli_op.apply(new_states, states);

  //
  // Calculate the expected new states
  //
  auto pop_dense = pauli_op.get_dense_repr();

  // pop_dense : d x d
  // states : n x d
  // expected_states : n x d

  std::vector<std::complex<double>> expected(dims * n_states, 0);
  std::mdspan<std::complex<double>, std::dextents<size_t, 2>> expected_span(
      expected.data(), dims, n_states);

  // Calculate the expected new states
  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dims; ++i) {
      for (size_t j = 0; j < dims; ++j) {
        expected_span(i, t) += pop_dense[i][j] * states(j, t);
      }
    }
  }

  // Check
  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dims; ++i) {
      fmt::println("new_states(i, t): {}, expected_span(i, t): {}",
                   new_states(i, t), expected_span(i, t));
      CHECK(abs(new_states(i, t) - expected_span(i, t)) < 1e-6);
    }
  }
}

TEST_CASE("test apply multistate multistring identity") {
  // Set up PauliOp
  std::vector<PauliString> pauli_strings = {"IIIIII", "IIIIII", "IIIIII",
                                            "IIIIII", "IIIIII", "IIIIII"};
  std::vector<std::complex<double>> coeffs(pauli_strings.size(),
                                           1. / pauli_strings.size());
  PauliOp<double> pauli_op(coeffs, pauli_strings);

  size_t const dims = pauli_strings[0].dims();

  // Set up random states
  size_t const n_states = 10;
  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dims, n_states});

  // Apply the PauliOp to a batch of states
  std::vector<std::complex<double>> new_states_raw(dims * n_states, 0);
  std::mdspan<std::complex<double>, std::dextents<size_t, 2>> new_states(
      new_states_raw.data(), dims, n_states);
  pauli_op.apply(new_states, states);

  //
  // States should be unchanged
  //

  // Check
  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dims; ++i) {
      CHECK(abs(new_states(i, t) - states(i, t)) < 1e-6);
    }
  }
}
