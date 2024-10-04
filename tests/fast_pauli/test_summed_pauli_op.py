#############################################################################
# This code is part of Fast Pauli.
#
# (C) Copyright Qognitive Inc 2024.
#
# This code is licensed under the BSD 2-Clause License. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#############################################################################


"""Test SummedPauliOp Python and C++ implementations."""

import numpy as np
import pytest

import fast_pauli as fp
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
