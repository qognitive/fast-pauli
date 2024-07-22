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

template <std::floating_point T>
using ComplexMatrix = std::vector<std::vector<std::complex<T>>>;

template <std::floating_point T>
ComplexMatrix<T> tensor_prod(ComplexMatrix<T> const &A,
                             ComplexMatrix<T> const &B) {
  size_t const n = A.size();
  size_t const m = A[0].size();
  size_t const p = B.size();
  size_t const q = B[0].size();

  ComplexMatrix<T> C(n * p, std::vector<std::complex<T>>(m * q));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (size_t k = 0; k < p; ++k) {
        for (size_t l = 0; l < q; ++l) {
          C[i * p + k][j * q + l] = A[i][j] * B[k][l];
        }
      }
    }
  }

  return C;
}

template <std::floating_point T>
ComplexMatrix<T> paulistring_to_dense(PauliString const &ps) {
  ComplexMatrix<double> res, tmp;
  res = {{1.}};

  for (auto const &p : ps.paulis | std::views::reverse) {
    tmp = p.to_tensor<double>();
    res = tensor_prod(tmp, res);
  }
  return res;
}

std::vector<std::complex<double>>
dense_apply(PauliString const &ps,
            std::vector<std::complex<double>> const &state) {
  ComplexMatrix<double> pauli_mat = paulistring_to_dense<double>(ps);
  std::vector<std::complex<double>> res(state.size(), 0);
  for (size_t i = 0; i < pauli_mat.size(); ++i) {
    for (size_t j = 0; j < pauli_mat[0].size(); ++j) {
      res[i] += pauli_mat[i][j] * state[j];
    }
  }
  return res;
}

TEST_CASE("check tensor_prod helper") {
  ComplexMatrix<double> A{{1, 2}, {3, 4}};
  ComplexMatrix<double> B{{5, 6}, {7, 8}};
  ComplexMatrix<double> C = tensor_prod(A, B);

  fmt::print("C: \n{}\n", fmt::join(C, "\n"));

  CHECK(C == ComplexMatrix<double>{{5, 6, 10, 12},
                                   {7, 8, 14, 16},
                                   {15, 18, 20, 24},
                                   {21, 24, 28, 32}});
}

//
// Tests
//

TEST_CASE("test pauli default init") {
  std::vector<Pauli> paulis(10);
  PauliString ps(paulis);
  CHECK(ps.weight == 0);
  fmt::print("PauliString: {}\n", ps);
}

TEST_CASE("pauli string manual init") {
  std::vector paulis{Pauli{0}, Pauli{1}, Pauli{2}, Pauli{3}};
  PauliString ps{paulis};
  CHECK(ps.weight == 3);
  fmt::print("PauliString: {}\n", ps);
}

TEST_CASE("string init") {
  PauliString ps{"IXYZ"};
  CHECK(ps.weight == 3);
  fmt::print("PauliString: {}\n", ps);
}

TEST_CASE("getters") {

  {
    PauliString ps{"I"};
    CHECK(ps.n_qubits() == 1);
    CHECK(ps.dims() == 2);
  }

  {
    PauliString ps{"IXYZI"};
    CHECK(ps.n_qubits() == 5);
    CHECK(ps.dims() == 32);
  }

  {
    PauliString ps{"IXYZ"};
    CHECK(ps.n_qubits() == 4);
    CHECK(ps.dims() == 16);
  }

  {
    PauliString ps;
    CHECK(ps.n_qubits() == 0);
    CHECK(ps.dims() == 1);
  }
}

TEST_CASE("test sparse repr") {
  for (PauliString ps : {"IXYZ", "IX", "XXYIYZI"}) {
    ComplexMatrix<double> res = paulistring_to_dense<double>(ps);
    ComplexMatrix<double> myres = ps.get_dense_repr<double>();

    // fmt::print("res: \n{}\n", fmt::join(res, "\n"));
    // fmt::print("myres: \n{}\n", fmt::join(myres, "\n"));

    // Check
    for (size_t i = 0; i < res.size(); ++i) {
      for (size_t j = 0; j < res[0].size(); ++j) {
        // CHECK(res[i][j] == myres[i][j]);
        CHECK(abs(res[i][j] - myres[i][j]) < 1e-6);
      }
    }
  }
}

TEST_CASE("test apply trivial") {
  PauliString ps{"IIII"};

  std::vector<std::complex<double>> state(1 << ps.paulis.size(), 1.);

  auto new_state = ps.apply(state);
  fmt::print("New state: \n[{}]\n", fmt::join(new_state, ",\n "));
  CHECK(new_state == state);
}

TEST_CASE("test apply simple") {
  PauliString ps{"IXI"};

  std::vector<std::complex<double>> state(1 << ps.paulis.size(), 0);
  state[6] = 1.;
  state[7] = 1.;
  auto new_state = ps.apply(state);
  fmt::print("New state: \n[{}]\n", fmt::join(new_state, ",\n "));

  auto expected = dense_apply(ps, state);
  CHECK(new_state == expected);
}

TEST_CASE("test apply") {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1.0, 2.0);

  for (PauliString ps : {"IXYZ", "YYIX", "XXYIYZ", "IZIXYYZ", "IZIXYYZIXYZ"}) {
    std::vector<std::complex<double>> state(1 << ps.paulis.size());
    std::generate(state.begin(), state.end(),
                  [&]() { return std::complex<double>(dis(gen), dis(gen)); });

    auto new_state = ps.apply(state);
    auto expected = dense_apply(ps, state);

    // fmt::print("New state: \n[{}]\n", fmt::join(new_state, ",\n "));
    // fmt::print("Expected: \n[{}]\n", fmt::join(expected, ",\n "));

    //
    for (size_t i = 0; i < new_state.size(); ++i) {
      CHECK(abs(new_state[i] - expected[i]) < 1e-6);
    }
  }
}

TEST_CASE("test apply batch") {
  // For random states
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1.0, 2.0);

  size_t const n_states = 10;

  // Testing each of these pauli strings individually
  // NOTE: apply_batch takes the transpose of the states and new_states
  for (PauliString ps : {"IXYZ", "YYIX", "XXYIYZ", "IZIXYYZ"}) {
    size_t const dims = ps.dims();

    std::vector<std::complex<double>> states_raw(dims * n_states);
    std::generate(states_raw.begin(), states_raw.end(),
                  [&]() { return std::complex<double>(dis(gen), dis(gen)); });
    std::mdspan<std::complex<double>, std::dextents<size_t, 2>> states_T(
        states_raw.data(), dims, n_states);

    auto new_states_raw = states_raw;
    std::mdspan<std::complex<double>, std::dextents<size_t, 2>> new_states_T(
        new_states_raw.data(), dims, n_states);

    ps.apply_batch(new_states_T, states_T, std::complex<double>(1.));

    // Calculate expected
    auto ps_dense = ps.get_dense_repr<double>(); // d x d
    std::vector<std::complex<double>> expected_raw(dims * n_states, 0);
    std::mdspan<std::complex<double>, std::dextents<size_t, 2>> expected(
        expected_raw.data(), dims, n_states);

    for (size_t i = 0; i < (dims); ++i) {
      for (size_t t = 0; t < n_states; ++t) {
        for (size_t j = 0; j < (dims); ++j) {
          expected(i, t) += ps_dense[i][j] * states_T(j, t);
        }
      }
    }

    // Check
    for (size_t t = 0; t < n_states; ++t) {
      for (size_t i = 0; i < dims; ++i) {
        CHECK(abs(new_states_T(i, t) - expected(i, t)) < 1e-6);
      }
    }
  }
}

TEST_CASE("get nontrivial paulis") {
  {
    auto res = get_nontrivial_paulis(0);
    CHECK(res.size() == 0);
  }

  {
    auto res = get_nontrivial_paulis(1);
    CHECK(res.size() == 3);
    CHECK(res[0] == "X");
    CHECK(res[1] == "Y");
    CHECK(res[2] == "Z");
  }

  {
    auto res = get_nontrivial_paulis(2);
    CHECK(res.size() == 9);
    CHECK(res[0] == "XX");
    CHECK(res[1] == "XY");
    CHECK(res[2] == "XZ");
    CHECK(res[3] == "YX");
    CHECK(res[4] == "YY");
    CHECK(res[5] == "YZ");
    CHECK(res[6] == "ZX");
    CHECK(res[7] == "ZY");
    CHECK(res[8] == "ZZ");
  }

  {
    auto res = get_nontrivial_paulis(3);
    CHECK(res.size() == 27);
    CHECK(res[0] == "XXX");
    CHECK(res[1] == "XXY");
    CHECK(res[2] == "XXZ");
    CHECK(res[3] == "XYX");
    CHECK(res[4] == "XYY");
    CHECK(res[5] == "XYZ");
    CHECK(res[6] == "XZX");
    CHECK(res[7] == "XZY");
    CHECK(res[8] == "XZZ");
    CHECK(res[9] == "YXX");
    CHECK(res[10] == "YXY");
    CHECK(res[11] == "YXZ");
    CHECK(res[12] == "YYX");
    CHECK(res[13] == "YYY");
    CHECK(res[14] == "YYZ");
    CHECK(res[15] == "YZX");
    CHECK(res[16] == "YZY");
    CHECK(res[17] == "YZZ");
    CHECK(res[18] == "ZXX");
    CHECK(res[19] == "ZXY");
    CHECK(res[20] == "ZXZ");
    CHECK(res[21] == "ZYX");
    CHECK(res[22] == "ZYY");
    CHECK(res[23] == "ZYZ");
    CHECK(res[24] == "ZZX");
    CHECK(res[25] == "ZZY");
    CHECK(res[26] == "ZZZ");
  }
}

TEST_CASE("idx combinations") {
  {
    auto res = idx_combinations(4, 1);
    CHECK(res.size() == 4);
    CHECK(res[0] == std::vector<size_t>{0});
    CHECK(res[1] == std::vector<size_t>{1});
    CHECK(res[2] == std::vector<size_t>{2});
    CHECK(res[3] == std::vector<size_t>{3});
  }

  {
    auto res = idx_combinations(4, 2);
    CHECK(res.size() == 6);
    CHECK(res[0] == std::vector<size_t>{0, 1});
    CHECK(res[1] == std::vector<size_t>{0, 2});
    CHECK(res[2] == std::vector<size_t>{0, 3});
    CHECK(res[3] == std::vector<size_t>{1, 2});
    CHECK(res[4] == std::vector<size_t>{1, 3});
    CHECK(res[5] == std::vector<size_t>{2, 3});
  }
}

TEST_CASE("calculate pauli strings") {
  {
    auto res = calcutate_pauli_strings(4, 0);
    CHECK(res.size() == 1);
    CHECK(res[0] == PauliString("IIII"));
  }

  {
    auto res = calcutate_pauli_strings(2, 1);
    CHECK(res.size() == 6);
    CHECK(res[0] == PauliString("XI"));
    CHECK(res[1] == PauliString("IX"));
    CHECK(res[2] == PauliString("YI"));
    CHECK(res[3] == PauliString("IY"));
    CHECK(res[4] == PauliString("ZI"));
    CHECK(res[5] == PauliString("IZ"));
  }

  {
    auto res = calcutate_pauli_strings(4, 2);
    CHECK(res.size() == 54);
    CHECK(res[0] == PauliString("XXII"));
    CHECK(res[1] == PauliString("XIXI"));
    CHECK(res[53] == PauliString("IIZZ"));
  }
}

TEST_CASE("calculate pauli string max weight") {
  {
    auto res = calculate_pauli_strings_max_weight(4, 0);
    CHECK(res.size() == 1);
    CHECK(res[0] == PauliString("IIII"));
  }

  {
    auto res = calculate_pauli_strings_max_weight(2, 1);
    CHECK(res.size() == 7);
    CHECK(res[0] == PauliString("II"));
    CHECK(res[1] == PauliString("XI"));
    CHECK(res[2] == PauliString("IX"));
    CHECK(res[3] == PauliString("YI"));
    CHECK(res[4] == PauliString("IY"));
    CHECK(res[5] == PauliString("ZI"));
    CHECK(res[6] == PauliString("IZ"));
  }

  {
    auto res = calculate_pauli_strings_max_weight(4, 2);
    CHECK(res.size() == 67);
  }

  {
    auto res = calculate_pauli_strings_max_weight(12, 2);
    CHECK(res.size() == 631);
  }
}