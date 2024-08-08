"""Efficient operations on Pauli string using numpy."""

import numpy as np


class PauliString:
    """Class representing a Pauli string with efficient operations."""

    def __init__(self, string: str) -> None:
        """Initialize the PauliString object.

        Args:
        ----
            string: The input string representing the Pauli string.
            Must contain only I, X, Y, Z characters.

        """
        if len(string) == 0 or not all([c in "IXYZ" for c in string]):
            raise ValueError(f"Invalid pauli string '{string}'")

        self.string = string
        self.dim = 1 << len(string)
        self.weight = len(string) - string.count("I")

    def __str__(self) -> str:
        """Convert pauli string object to regular string.

        Returns
        -------
            str: Plain pauli string.
        """
        return self.string

    def __repr__(self) -> str:
        """Get a string representation of the PauliString object.

        Returns
        -------
            str: A string representation of the PauliString object.
        """
        return f'PauliString("{self.string}")'

    def dense(self) -> np.ndarray:
        """Return the dense matrix representation of the Pauli string."""
        columns, values = compose_sparse_pauli(self.string)

        matrix = np.zeros((columns.size, values.size), dtype=np.complex128)
        matrix[np.arange(columns.size), columns] = values
        return matrix

    def multiply(self, states: np.ndarray, coeff: np.complex128 = 1.0) -> np.ndarray:
        """Efficient multiplication of Pauli string with a given state.

        Args:
        ----
            state: The input state as a numpy array.
            coeff: Multiplication factor to apply to the PauliString

        Returns
        -------
            The result of multiplying the Pauli string with the state.

        """
        if states.shape[0] != self.dim or states.ndim > 2:
            raise ValueError(f"Provided state has inconsistent shape {states.shape}")

        columns, values = compose_sparse_pauli(self.string)
        values *= coeff

        if states.ndim == 2:
            return values[:, np.newaxis] * states[columns]
        else:
            return values * states[columns]

    def expected_value(self, state: np.ndarray) -> np.complex128 | np.ndarray:
        """Compute the expected value of Pauli string for a given state.

        Args:
        ----
            state: The input state as a numpy array.
            coeff: Multiplication factor to apply to the PauliString

        Returns
        -------
            The expected value of the Pauli string with the state.

        """
        return np.multiply(state.conj(), self.multiply(state)).sum(axis=0)


def compose_sparse_pauli(string: str) -> tuple[np.ndarray, np.ndarray]:
    """Produce sparse representation of the pauli string.

    Args:
    ----
        string: The input string representing the Pauli string.
        Must contain only I, X, Y, Z characters.

    Returns
    -------
        A tuple containing the column numbers and values of the sparse matrix.

    """
    n_qubits = len(string)
    n_vals = 1 << n_qubits
    n_ys = string.count("Y")

    # initialize cols array with zeros as we need first element to be 0
    cols = np.zeros(n_vals, dtype=np.int32)
    vals = np.empty(n_vals, dtype=np.complex128)

    for p in string:
        cols[0] <<= 1
        if p == "X" or p == "Y":
            cols[0] += 1

    match n_ys % 4:
        case 0:
            vals[0] = 1.0
        case 1:
            vals[0] = -1.0j
        case 2:
            vals[0] = -1.0
        case 3:
            vals[0] = 1.0j

    for q in range(n_qubits):
        p = string[n_qubits - q - 1]
        pow_of_two = 1 << q

        new_slice = slice(pow_of_two, 2 * pow_of_two)
        old_slice = slice(0, pow_of_two)

        match p:
            case "I":
                cols[new_slice] = cols[old_slice] + pow_of_two
                vals[new_slice] = vals[old_slice]
            case "X":
                cols[new_slice] = cols[old_slice] - pow_of_two
                vals[new_slice] = vals[old_slice]
            case "Y":
                cols[new_slice] = cols[old_slice] - pow_of_two
                vals[new_slice] = -vals[old_slice]
            case "Z":
                cols[new_slice] = cols[old_slice] + pow_of_two
                vals[new_slice] = -vals[old_slice]

    return cols, vals


def compose_sparse_diag_pauli(string: str) -> np.ndarray:
    """Produce sparse representation of diagonal pauli string.

    Args:
    ----
        string: A Pauli string containing only 'I' and 'Z' characters.

    Returns
    -------
        np.ndarray: diagonal values from resulting sparse matrix.

    """
    if "X" in string or "Y" in string:
        raise ValueError("Pauli string must contain only I and Z characters")

    n_qubits = len(string)
    n_vals = 1 << n_qubits

    # initialize vals array with ones as we need first element to be 1
    vals = np.ones(n_vals, dtype=np.complex128)

    for q in range(n_qubits):
        p = string[n_qubits - q - 1]
        pow_of_two = 1 << q

        new_slice = slice(pow_of_two, 2 * pow_of_two)
        old_slice = slice(0, pow_of_two)

        match p:
            case "I":
                vals[new_slice] = vals[old_slice]
            case "Z":
                vals[new_slice] = -vals[old_slice]

    return vals
