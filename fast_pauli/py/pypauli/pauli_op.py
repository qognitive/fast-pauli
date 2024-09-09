"""PauliOp implementation using numpy."""

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

        self.coeffs = np.array(coefficients, dtype=np.complex128)
        self.pauli_strings = [PauliString(ps) for ps in strings if isinstance(ps, str)]
        self.n_strings = len(strings)

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
    def strings(self) -> list[str]:
        """Get the list of Pauli strings."""
        return [str(ps) for ps in self.pauli_strings]

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
