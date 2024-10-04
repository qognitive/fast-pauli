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

"""Compatibility layer for Qiskit."""

from typing import Any

from qiskit.quantum_info import Pauli, SparsePauliOp

from . import _fast_pauli as fp


def from_qiskit(qiskit_type: Any) -> Any:
    """Convert Qiskit type to fast_pauli type.

    Parameters
    ----------
    qiskit_type : Any
        Qiskit type to convert to fast_pauli type.

    Returns
    -------
    Any
        fast_pauli type.

    Raises
    ------
    NotImplementedError
        Conversion from Qiskit type to fast_pauli type isn't supported.
    """
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
    """Convert fast_pauli type to Qiskit type.

    Parameters
    ----------
    fast_pauli_type : Any
        fast_pauli type to convert to Qiskit type.

    Returns
    -------
    Any
        Qiskit type.

    Raises
    ------
    NotImplementedError
        Conversion from fast_pauli type to Qiskit type isn't supported.
    """
    if isinstance(fast_pauli_type, fp.PauliOp):
        return SparsePauliOp(
            fast_pauli_type.pauli_strings_as_str,
            fast_pauli_type.coeffs,
        )

    elif isinstance(fast_pauli_type, fp.PauliString):
        return Pauli(str(fast_pauli_type))

    else:
        raise NotImplementedError(
            f"Conversion from {type(fast_pauli_type)} to "
            + "Qiskit equivalent isn't supported."
        )
