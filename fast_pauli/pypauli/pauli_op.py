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

"""PauliOp implementation using numpy."""

from __future__ import annotations

import numpy as np

from fast_pauli.pypauli.pauli_string import PauliString, compose_sparse_pauli


class PauliOp:
    """Class representing a Pauli operator with efficient operations."""

    def __init__(
        self, coefficients: list | np.ndarray, strings: list[str] | list[PauliString]
    ) -> None:
        """Initialize the PauliOp object.

        Args:
        ----
            coefficients: The input coefficients for the Pauli Operator
            strings: Set of Pauli Strings to compose an operator.

        """
        if len(coefficients) != len(strings):
            raise ValueError("coefficients and strings must have the same size.")

        self.coeffs = np.array(coefficients, dtype=np.complex128, copy=True)
        self.pauli_strings = [
            PauliString(ps) if isinstance(ps, str) else ps for ps in strings
        ]

        if self.pauli_strings:
            self.n_qubits = self.pauli_strings[0].n_qubits
            self.dim = self.pauli_strings[0].dim
            for ps in self.pauli_strings[1:]:
                if ps.n_qubits != self.n_qubits:
                    raise ValueError("All Pauli strings must have the same length.")
        else:
            self.n_qubits = 0
            self.dim = 0

    @property
    def pauli_strings_as_str(self) -> list[str]:
        """Get the list of Pauli strings."""
        return [str(ps) for ps in self.pauli_strings]

    @property
    def n_pauli_strings(self) -> int:
        """Get the number of Pauli strings."""
        return len(self.pauli_strings)

    def copy(self) -> PauliOp:
        """Return a copy of the Pauli operator."""
        # TODO proper interface for deep&shallow copying of PauliOp and PauliString
        return PauliOp(self.coeffs, self.pauli_strings_as_str)

    def scale(self, factors: complex | np.ndarray) -> None:
        """Scale each individual term by a factor.

        Args:
        ----
            factors: The factors to scale Pauli operator's coefficients by.
        """
        if isinstance(factors, (complex, float, int)):
            self.coeffs *= factors
        else:
            if factors.shape != self.coeffs.shape:
                raise ValueError("Invalid shape for the scaling factors.")
            self.coeffs *= factors

    def to_tensor(self) -> np.ndarray:
        """Return the dense matrix representation of the Pauli operator."""
        matrix = np.zeros((self.dim, self.dim), dtype=np.complex128)

        for i, ps in enumerate(self.pauli_strings):
            columns, values = compose_sparse_pauli(ps.string)
            matrix[np.arange(self.dim), columns] += self.coeffs[i] * values

        return matrix

    def apply(self, states: np.ndarray) -> np.ndarray:
        """Efficient multiplication of Pauli operator with a given state.

        Args:
        ----
            state: The input state as a numpy array.

        Returns
        -------
            The result of multiplying the Pauli operator with the state.

        """
        if states.shape[0] != self.dim:
            raise ValueError(f"Provided state has inconsistent shape {states.shape}")

        result = np.zeros_like(states, dtype=np.complex128)
        for c, ps in zip(self.coeffs, self.pauli_strings):
            result += ps.apply(states, c)
        return result

    def expectation_value(self, states: np.ndarray) -> np.complex128 | np.ndarray:
        """Compute the expected value of the operator for the given state.

        Args:
        ----
            state: The input state as a numpy array.

        Returns
        -------
            The expected value of the operator with the given state.

        """
        result = np.zeros(
            1 if len(states.shape) == 1 else states.shape[-1], dtype=np.complex128
        )
        for c, ps in zip(self.coeffs, self.pauli_strings):
            result += c * ps.expectation_value(states)
        return result

    def __matmul__(self, rhs: PauliOp | PauliString) -> PauliOp:
        """Matrix multiplication with other PauliOp or PauliString object.

        Args:
        ----
            rhs: The Pauli operator or string

        Returns
        -------
            The result of multiplication
        """
        if self.dim != rhs.dim:
            raise ValueError("Pauli operators must have the same dimension.")

        if isinstance(rhs, PauliString):
            new_strings = []
            new_coeffs = self.coeffs.copy()

            for i, p_str in enumerate(self.pauli_strings):
                phase, new_p_str = p_str @ rhs
                new_strings.append(new_p_str)
                new_coeffs[i] *= phase

        elif isinstance(rhs, PauliOp):
            new_strings = []
            new_coeffs = []
            dedupe_strings: dict[PauliString, int] = {}

            for i, l_str in enumerate(self.pauli_strings):
                for j, r_str in enumerate(rhs.pauli_strings):
                    phase, new_p_str = l_str @ r_str
                    coeff_ij = phase * self.coeffs[i] * rhs.coeffs[j]

                    if new_p_str in dedupe_strings:
                        new_coeffs[dedupe_strings[new_p_str]] += coeff_ij
                    else:
                        dedupe_strings[new_p_str] = len(new_strings)
                        new_strings.append(new_p_str)
                        new_coeffs.append(coeff_ij)

            new_coeffs = np.array(new_coeffs, dtype=np.complex128)
        else:
            raise ValueError("Invalid type for the operand.")

        return PauliOp(new_coeffs, new_strings)

    def __rmatmul__(self, lhs: PauliString) -> PauliOp:
        """Matrix multiplication with PauliString on the left.

        Args:
        ----
            lhs: The Pauli string to multiply with.

        Returns
        -------
            The result of multiplication
        """
        if self.dim != lhs.dim:
            raise ValueError("Pauli operators must have the same dimension.")

        if isinstance(lhs, PauliString):
            new_strings = []
            new_coeffs = self.coeffs.copy()

            for i, p_str in enumerate(self.pauli_strings):
                phase, new_p_str = lhs @ p_str
                new_strings.append(new_p_str)
                new_coeffs[i] *= phase
        else:
            raise ValueError("Invalid type for the operand.")

        return PauliOp(new_coeffs, new_strings)

    def __mul__(self, rhs: complex) -> PauliOp:
        """Multiply the PauliOp with a scalar.

        Args:
        ----
            rhs: The scalar to multiply with.

        Returns
        -------
            PauliOp: Pauli Op holding the product of the PauliOp and the scalar.
        """
        return PauliOp(self.coeffs * rhs, self.pauli_strings_as_str)

    def __rmul__(self, lhs: complex) -> PauliOp:
        """Multiply the PauliOp with a scalar.

        Args:
        ----
            lhs: The scalar to multiply with.

        Returns
        -------
            PauliOp: Pauli Op holding the product of the PauliOp and the scalar.
        """
        return self * lhs

    def __imul__(self, other: complex) -> PauliOp:
        """Multiply the PauliOp with a scalar.

        Args:
        ----
            other: The scalar to multiply with.
        """
        self.scale(other)
        return self

    def __add__(self, rhs: PauliOp | PauliString) -> PauliOp:
        """Add two PauliOp objects together.

        Args:
        ----
            rhs: The other PauliOp object to add.

        Returns
        -------
            PauliOp: Pauli Op holding the sum of the two PauliOps.
        """
        res_op = self.copy()
        res_op.extend(rhs.copy())
        return res_op

    def __iadd__(self, rhs: PauliOp | PauliString) -> PauliOp:
        """Add another PauliOp to the current one.

        Args:
        ----
            rhs: The Pauli operator to be added.
        """
        self.extend(rhs.copy())
        return self

    def __sub__(self, rhs: PauliOp | PauliString) -> PauliOp:
        """Subtract another PauliOp from the current one.

        Args:
        ----
            rhs: The Pauli operator to be subtracted.

        Returns
        -------
            PauliOp: Pauli Op holding the difference of the two PauliOps.
        """
        res_op = self.copy()
        res_op.extend(rhs.copy(), -1.0)
        return res_op

    def __isub__(self, rhs: PauliOp | PauliString) -> PauliOp:
        """Subtract another PauliOp from the current one.

        Args:
        ----
            rhs: The Pauli operator to be subtracted.
        """
        self.extend(rhs.copy(), -1.0)
        return self

    def extend(
        self,
        other: PauliOp | PauliString,
        multiplier: complex | None = None,
        dedupe: bool = True,
    ) -> None:
        """Add another PauliOp to the current one
        by extending the internal summation with new terms.

        Note: this does a shallow copy of other's properties

        Args:
        ----
            other: The Pauli operator to be added.
            multiplier: The coefficient to be applied to the new terms.
            dedupe: Flag to remove duplicate terms.
        """  # noqa: D205, D210
        if self.dim != other.dim:
            raise ValueError("Pauli operators must have the same dimension.")

        if isinstance(other, PauliString):
            self.pauli_strings.append(other)
            self.coeffs = np.append(self.coeffs, multiplier or 1.0)
        elif isinstance(other, PauliOp):
            self.pauli_strings += other.pauli_strings
            self.coeffs = np.append(
                self.coeffs, other.coeffs * multiplier if multiplier else other.coeffs
            )
        else:
            raise ValueError("Invalid type for the operand.")

        if dedupe:
            dedupe_strings: dict[PauliString, complex] = {}
            for c, p_str in zip(self.coeffs, self.pauli_strings):
                dedupe_strings[p_str] = dedupe_strings.get(p_str, 0) + c
            self.pauli_strings = list(dedupe_strings.keys())
            self.coeffs = np.array(list(dedupe_strings.values()), dtype=np.complex128)
