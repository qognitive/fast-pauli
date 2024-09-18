"""Test SummedPauliOp Python and C++ implementations."""

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp
from tests.conftest import resolve_parameter_repr


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_states,n_operators,n_qubits",
    [(s, o, q) for s in [1, 10, 1000] for o in [1, 10, 100] for q in [1, 2, 6]],
)
def test_expectation_values(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_states: int,
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test expectation value calculation."""
    n_states = 1000
    n_operators = 100
    n_qubits = 6

    # TODO This will break on python version
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    pauli_strings_str = [str(s) for s in pauli_strings]
    n_strings = len(pauli_strings)

    coeffs_2d = np.random.rand(n_strings, n_operators).astype(np.complex128)
    psi = np.random.rand(2**n_qubits, n_states).astype(np.complex128)

    op = summed_pauli_op(pauli_strings, coeffs_2d)

    # The expectation values we want to test
    expectation_vals = op.expectation_value(psi)

    # The "trusted" expectation_values
    expectation_vals_naive = np.zeros((n_operators, n_states), dtype=np.complex128)

    # Calculate using brute force
    for k in range(n_operators):
        A_k = pp.helpers.naive_pauli_operator(coeffs_2d[:, k], pauli_strings_str)
        expectation_vals_naive[k] = np.einsum("it,ij,jt->t", psi.conj(), A_k, psi)

    # Check
    np.testing.assert_allclose(expectation_vals, expectation_vals_naive, atol=1e-13)
