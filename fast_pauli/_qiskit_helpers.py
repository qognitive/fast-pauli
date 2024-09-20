"""Compatibility layer for Qiskit."""

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from . import _fast_pauli as fp


def from_qiskit(qiskit_type: Any) -> Any:
    if isinstance(qiskit_type, SparsePauliOp):
        tuples = qiskit_type.to_list(array=True)
        coeffs = [c for _, c in tuples]
        strings = [s for s, _ in tuples]
        return fp.PauliOp(coeffs, strings)

    else:
        raise NotImplementedError(
            f"Conversion from {type(qiskit_type)} to "
            + "fast_pauli equivalent isn't supported."
        )


def to_qiskit(fast_pauli_type: Any) -> Any:
    if isinstance(fast_pauli_type, fp.PauliOp):
        return SparsePauliOp(
            fast_pauli_type.pauli_strings_as_str,
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
