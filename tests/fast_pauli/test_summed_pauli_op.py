"""Test SummedPauliOp Python and C++ implementations."""

import numpy as np

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp


def test_expectation_values() -> None:
    """Test expectation value calculation."""
    n_states = 1000
    n_operators = 100
    n_qubits = 6
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    pauli_strings_str = [str(s) for s in pauli_strings]
    n_strings = len(pauli_strings)
    coeffs_2d = np.random.rand(n_strings, n_operators).astype(np.complex128)
    psi = np.random.rand(2**n_qubits, n_states).astype(np.complex128)

    op = fp.SummedPauliOp(pauli_strings, coeffs_2d)

    expectation_vals = op.expectation_value(psi)

    expectation_vals_naive = np.zeros((n_operators, n_states))

    for k in range(n_operators):
        A_k = pp.helpers.naive_pauli_operator(coeffs_2d[:, k], pauli_strings_str)
        expectation_vals_naive[k] = np.einsum("it,ij,jt->t", psi.conj(), A_k, psi)

    np.testing.assert_allclose(expectation_vals, expectation_vals_naive, atol=1e-13)
