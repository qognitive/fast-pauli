"""Test Qiskit interface."""

from typing import Callable

import numpy as np
import pytest

import fast_pauli as fp


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_qiskit_interface(size: int, pauli_strings_with_size: Callable) -> None:
    """Make sure we can go from fast_pauli to qiskit and back."""
    paulis = pauli_strings_with_size(size)
    op = fp.PauliOp(np.random.rand(len(paulis)).astype(np.complex128), paulis)
    q_op = fp.to_qiskit(op)
    fp_op = fp.from_qiskit(q_op)
    # qiskit_strings = [s for s, _ in q_op.to_list(array=True)]
    np.testing.assert_allclose(op.coeffs, fp_op.coeffs)
    np.testing.assert_array_equal(op.pauli_strings_as_str, fp_op.pauli_strings_as_str)
