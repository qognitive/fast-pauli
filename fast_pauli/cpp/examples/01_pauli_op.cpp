#include <algorithm>
#include <random>

#include "fast_pauli.hpp"

using namespace fast_pauli;

int
main()
{
  std::vector<PauliString> pauli_strings(100000, "XYZXYZXYZXYZ");

  std::vector<std::complex<double>> coeffs(pauli_strings.size(), 1);
  PauliOp<double> pauli_op(coeffs, pauli_strings);

  size_t const dims = pauli_strings[0].dim();

  // Set up random state
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);
  std::vector<std::complex<double>> state(dims, 0);
  std::generate(state.begin(), state.end(), [&]() { return std::complex<double>(dis(gen), dis(gen)); });

  // Apply the PauliOp
  std::vector<std::complex<double>> res = pauli_op.apply(state);

  return 0;
}