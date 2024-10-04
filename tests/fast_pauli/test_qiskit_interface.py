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

"""Test Qiskit interface."""

from typing import Callable

import numpy as np
import pytest

import fast_pauli as fp


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_qiskit_interface(
    size: int, pauli_strings_with_size: Callable, generate_random_complex: Callable
) -> None:
    """Make sure we can go from fast_pauli to qiskit and back."""
    paulis = pauli_strings_with_size(size)
    op = fp.PauliOp(generate_random_complex(len(paulis)), paulis)
    q_op = fp.to_qiskit(op)
    fp_op = fp.from_qiskit(q_op)
    np.testing.assert_allclose(op.coeffs, fp_op.coeffs)
    np.testing.assert_array_equal(op.pauli_strings_as_str, fp_op.pauli_strings_as_str)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_qiskit_interface_pauli_string(
    n_qubits: int, pauli_strings_with_size: Callable
) -> None:
    """Make sure we can go from fast_pauli to qiskit and back."""
    for ps in pauli_strings_with_size(n_qubits):
        pauli = fp.PauliString(ps)
        q_pauli = fp.to_qiskit(pauli)

        # We don't currently save the phase for a PauliString in fast_pauli
        # so we need to make sure it's set to 1. Note that Qiskit keeps track
        # of the *exponent* of the phase, so it should be 0 here.
        assert q_pauli.phase == 0

        assert pauli.n_qubits == q_pauli.num_qubits

        # Check that the dense representation matches
        ps_matrix = pauli.to_tensor()
        q_matrix = q_pauli.to_matrix()
        np.testing.assert_allclose(ps_matrix, q_matrix)
