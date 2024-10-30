#include <string>

#include "fast_pauli.hpp"

namespace fp = fast_pauli;

int main()
{
    size_t const n_qubits = 24;
    std::string pauli_string(n_qubits, 'X');
    fp::PauliString ps(pauli_string);
    std::vector<size_t> k;
    std::vector<std::complex<double>> m;
    std::tie(k, m) = get_sparse_repr<double>(ps.paulis);

    return 0;
}
