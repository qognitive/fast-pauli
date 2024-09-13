"""Compatibility layer for Qiskit."""

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

import fast_pauli._fast_pauli as fp


def from_qiskit(qiskit_type: Any) -> Any:
    if isinstance(qiskit_type, SparsePauliOp):
        return fp.PauliOp(qiskit_type.coeffs, qiskit_type.paulis)

    else:
        raise NotImplementedError(
            f"Conversion from {type(qiskit_type)} to "
            + "fast_pauli equivalent isn't supported."
        )


def to_qiskit(fast_pauli_type: Any) -> Any:
    if isinstance(fast_pauli_type, fp.PauliOp):
        return SparsePauliOp(
            fast_pauli_type.pauli_strings_as_str(),
            fast_pauli_type.coeffs,
        )

    else:
        raise NotImplementedError(
            f"Conversion from {type(fast_pauli_type)} to "
            + "Qiskit equivalent isn't supported."
        )


op = fp.PauliOp(np.random.rand(10).astype(np.complex128), ["IXYZ"] * 10)
q_op = to_qiskit(op)

fp_op = from_qiskit(q_op)
