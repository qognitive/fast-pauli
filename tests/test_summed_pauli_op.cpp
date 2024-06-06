#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

// #include "__summed_pauli_op.hpp"
#include <doctest/doctest.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "fast_pauli.hpp"

using namespace std::literals;
using namespace fast_pauli;

TEST_CASE("ctors") {
  {
    // Default
    SummedPauliOp<double> op;
    CHECK(op.pauli_strings.size() == 0);
  }

  {
    // SummedPauliOp(std::vector<PauliString> const &pauli_strings,
    // std::vector<std::complex<T>> const &coeffs_raw)
    SummedPauliOp<double> summed_op{{"XYZ", "ZZY", "YYI"}, {1i, 1i, 1i}};
  }

  {
    // Bad init
    // TODO probably a bad way to do this
    try {
      SummedPauliOp<double> summed_op{{"XYZ", "XXXX", "III"}, {1, 1i, -0.5i}};
      CHECK(false); // We shouldn't get here
    } catch (std::invalid_argument const &) {
      CHECK(true); // We should throw an invalid_argument and get here
    }
  }
}

TEST_CASE("accessors") {
  // Basic case
  {
    SummedPauliOp<double> summed_op{{"XYZ", "ZZY", "YYI"}, {1i, 1i, 1i}};
    CHECK(summed_op.n_dimensions() == 8);
    CHECK(summed_op.n_operators() == 1);
    CHECK(summed_op.n_pauli_strings() == 3);
  }

  // Multiple operators
  {
    SummedPauliOp<double> summed_op{{"XYZ", "ZZY", "YYI"},
                                    {1i, 1i, 1i, 1, 1, 1, 0, 0, 0}};
    CHECK(summed_op.n_dimensions() == 8);
    CHECK(summed_op.n_operators() == 3);
    CHECK(summed_op.n_pauli_strings() == 3);
  }
}

TEST_CASE("apply 1 operator 1 PauliString") {
  fmt::print("\n\napply 1 operator 1 PauliString\n");
  // Setup operator
  SummedPauliOp<double> summed_op{{"XYZ"}, {std::complex<double>(1.)}};

  // Setup states
  size_t const n_states = 1;
  size_t const dim = summed_op.n_dimensions();
  size_t const n_ops = summed_op.n_operators();

  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dim, n_states});

  std::vector<std::complex<double>> new_states_raw;
  std::mdspan new_states = fast_pauli::zeros<std::complex<double>, 2>(
      new_states_raw, {dim, n_states});

  // Init data
  std::vector<double> data_raw(n_ops * n_states, 1);
  std::mdspan<double, std::dextents<size_t, 2>> data(data_raw.data(), n_ops,
                                                     n_states);
  summed_op.apply(new_states, states, data);

  // Calculated the expected answer and check
  std::vector<std::complex<double>> expected_raw;
  std::mdspan expected =
      fast_pauli::zeros<std::complex<double>, 2>(expected_raw, {dim, n_states});

  PauliOp<double> pop{{std::complex<double>(1.)}, {"XYZ"}};
  pop.apply(expected, states);

  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dim; ++i) {
      CHECK(abs(new_states(i, t) - expected(i, t)) < 1e-6);
    }
  }
}

TEST_CASE("apply 2 operators 1 PauliString") {
  fmt::print("\n\napply 2 operators 1 PauliString\n");

  // Setup operator
  SummedPauliOp<double> summed_op{{"XYZ"}, {std::complex<double>(1.), 1i}};

  // Setup states
  size_t const n_states = 1;
  size_t const dim = summed_op.n_dimensions();
  size_t const n_ops = summed_op.n_operators();

  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dim, n_states});

  std::vector<std::complex<double>> new_states_raw;
  std::mdspan new_states = fast_pauli::zeros<std::complex<double>, 2>(
      new_states_raw, {dim, n_states});

  // Init data (aka data)
  std::vector<double> data_raw(n_ops * n_states, 1);
  std::mdspan<double, std::dextents<size_t, 2>> data(data_raw.data(), n_ops,
                                                     n_states);
  summed_op.apply(new_states, states, data);

  // Calculated the expected answer and check
  std::vector<std::complex<double>> expected_raw;
  std::mdspan expected =
      fast_pauli::zeros<std::complex<double>, 2>(expected_raw, {dim, n_states});

  // Manually calculate the sum over the different pauli operators
  for (size_t i = 0; i < dim; ++i) {
    for (size_t t = 0; t < n_states; ++t) {
      expected(i, t) += 1i * expected(i, t);
    }
  }

  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dim; ++i) {
      CHECK(abs(new_states(i, t) - expected(i, t)) < 1e-6);
    }
  }
}

TEST_CASE("apply 2 operators 2 PauliString") {
  fmt::print("\n\napply 2 operators 2 PauliString\n");
  // Setup operator
  SummedPauliOp<double> summed_op{{"XYZ", "YYZ"}, {1i, 1, 0.5i, -0.99}};

  // Setup states
  size_t const n_states = 10;
  size_t const dim = summed_op.n_dimensions();
  size_t const n_ops = summed_op.n_operators();

  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dim, n_states});

  std::vector<std::complex<double>> new_states_raw;
  std::mdspan new_states = fast_pauli::zeros<std::complex<double>, 2>(
      new_states_raw, {dim, n_states});

  // Init data
  std::vector<double> data_raw(n_ops * n_states, 1);
  std::mdspan<double, std::dextents<size_t, 2>> data(data_raw.data(), n_ops,
                                                     n_states);
  summed_op.apply(new_states, states, data);

  // Calculated expected answer
  std::vector<std::complex<double>> expected_raw;
  std::mdspan expected =
      fast_pauli::zeros<std::complex<double>, 2>(expected_raw, {dim, n_states});

  for (size_t j = 0; j < summed_op.n_pauli_strings(); ++j) {

    auto ps = summed_op.pauli_strings[j];

    PauliOp<double> pop{{std::complex<double>(1.)}, {ps}};
    std::vector<std::complex<double>> tmp_raw;
    std::mdspan tmp =
        fast_pauli::zeros<std::complex<double>, 2>(tmp_raw, {dim, n_states});

    pop.apply(tmp, states);

    // Manually calculate the sum over the different pauli operators
    // This is specific to the coefficients we've chosen above
    for (size_t i = 0; i < dim; ++i) {
      for (size_t t = 0; t < n_states; ++t) {
        // expected(i, t) += 1i * tmp(i, t);
        for (size_t k = 0; k < n_ops; ++k) {
          expected(i, t) += data(k, t) * summed_op.coeffs(j, k) * tmp(i, t);
        }
        // expected(i, t) += std::complex<double>(2) * tmp(i, t);
      }
    }
  }

  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dim; ++i) {
      CHECK(abs(new_states(i, t) - expected(i, t)) < 1e-6);
    }
  }
}

TEST_CASE("apply many operators many PauliString") {
  fmt::print("\n\napply many operators many PauliString\n");
  // Setup operator
  std::vector<PauliString> pauli_strings{"XIXXX", "IIXII", "ZYYZI",
                                         "ZYIIZ", "YXZZY", "IZYII"};

  std::vector<std::complex<double>> coeff_raw;
  std::mdspan coeff = fast_pauli::rand<std::complex<double>, 2>(
      coeff_raw, {pauli_strings.size(), 100});

  SummedPauliOp<double> summed_op{pauli_strings, coeff};

  // Setup states
  size_t const n_states = 10;
  size_t const dim = summed_op.n_dimensions();
  size_t const n_ops = summed_op.n_operators();

  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dim, n_states});

  std::vector<std::complex<double>> new_states_raw;
  std::mdspan new_states = fast_pauli::zeros<std::complex<double>, 2>(
      new_states_raw, {dim, n_states});

  // Init data (aka data)
  std::vector<double> data_raw;
  std::mdspan data = fast_pauli::rand<double, 2>(data_raw, {n_ops, n_states});

  // Apply the summed operator
  summed_op.apply(new_states, states, data);

  // Check the check
  std::vector<std::complex<double>> expected_raw;
  std::mdspan expected =
      fast_pauli::zeros<std::complex<double>, 2>(expected_raw, {dim, n_states});

  for (size_t j = 0; j < summed_op.n_pauli_strings(); ++j) {
    auto ps = summed_op.pauli_strings[j];

    PauliOp<double> pop{{std::complex<double>(1.)}, {ps}};
    std::vector<std::complex<double>> tmp_raw;
    std::mdspan tmp =
        fast_pauli::zeros<std::complex<double>, 2>(tmp_raw, {dim, n_states});

    pop.apply(tmp, states);

    // Manually calculate the sum over the different pauli operators
    // This is specific to the coefficients we've chosen above
    for (size_t i = 0; i < dim; ++i) {
      for (size_t t = 0; t < n_states; ++t) {
        for (size_t k = 0; k < n_ops; ++k) {
          expected(i, t) += data(k, t) * summed_op.coeffs(j, k) * tmp(i, t);
        }
      }
    }
  }

  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dim; ++i) {
      CHECK(abs(new_states(i, t) - expected(i, t)) < 1e-6);
    }
  }
}

TEST_CASE("apply many operators many PauliString") {
  fmt::print("\n\napply many operators many PauliString\n");
  // Setup operator
  std::vector<PauliString> pauli_strings{"XIXXX", "IIXII", "ZYYZI",
                                         "ZYIIZ", "YXZZY", "IZYII"};

  std::vector<std::complex<double>> coeff_raw;
  std::mdspan coeff = fast_pauli::rand<std::complex<double>, 2>(
      coeff_raw, {pauli_strings.size(), 100});

  SummedPauliOp<double> summed_op{pauli_strings, coeff};

  // Setup states
  size_t const n_states = 10;
  size_t const dim = summed_op.n_dimensions();
  size_t const n_ops = summed_op.n_operators();

  std::vector<std::complex<double>> states_raw;
  std::mdspan states =
      fast_pauli::rand<std::complex<double>, 2>(states_raw, {dim, n_states});

  std::vector<std::complex<double>> new_states_raw;
  std::mdspan new_states = fast_pauli::zeros<std::complex<double>, 2>(
      new_states_raw, {dim, n_states});

  // Init data
  std::vector<double> data_raw;
  std::mdspan data = fast_pauli::rand<double, 2>(data_raw, {n_ops, n_states});

  // Apply the summed operator
  summed_op.apply_parallel(new_states, states, data);

  // Calculate the expected answer
  std::vector<std::complex<double>> expected_raw;
  std::mdspan expected =
      fast_pauli::zeros<std::complex<double>, 2>(expected_raw, {dim, n_states});

  for (size_t j = 0; j < summed_op.n_pauli_strings(); ++j) {
    auto ps = summed_op.pauli_strings[j];

    PauliOp<double> pop{{std::complex<double>(1.)}, {ps}};
    std::vector<std::complex<double>> tmp_raw;
    std::mdspan tmp =
        fast_pauli::zeros<std::complex<double>, 2>(tmp_raw, {dim, n_states});

    pop.apply(tmp, states);

    // Manually calculate the sum over the different pauli operators
    // This is specific to the coefficients we've chosen above
    for (size_t i = 0; i < dim; ++i) {
      for (size_t t = 0; t < n_states; ++t) {
        // expected(i, t) += 1i * tmp(i, t);
        for (size_t k = 0; k < n_ops; ++k) {
          expected(i, t) += data(k, t) * summed_op.coeffs(j, k) * tmp(i, t);
        }
      }
    }
  }

  for (size_t t = 0; t < n_states; ++t) {
    for (size_t i = 0; i < dim; ++i) {
      CHECK(abs(new_states(i, t) - expected(i, t)) < 1e-6);
    }
  }
}