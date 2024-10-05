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


""":code:`fast_pauli` is a Python package with C++ backend, optimized for operations on Pauli Matrices and Pauli Strings."""  # noqa: E501

from ._fast_pauli import (  # noqa: F401
    Pauli,
    PauliOp,
    PauliString,
    SummedPauliOp,
    helpers,
)
from .qiskit_helpers import from_qiskit, to_qiskit  # noqa: F401
