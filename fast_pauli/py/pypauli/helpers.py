"""Separate file with helper functions for Pauli matrices."""

import numpy as np


def pauli_matrices() -> dict:
    """Provide a dictionary containing the Pauli matrices.

    The Pauli matrices are a set of 2x2 matrices commonly used in quantum mechanics.
    This function returns a dictionary with keys representing the matrix names
    ("I", "X", "Y", "Z", 0, 1, 2, 3) and values representing the corresponding matrix.

    Returns
    -------
        dict: A dictionary containing the Pauli matrices.

    """
    s0 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": s0, "X": s1, "Y": s2, "Z": s3, 0: s0, 1: s1, 2: s2, 3: s3}


def naive_pauli_converter(string: str) -> np.ndarray:
    """Convert Pauli string representation into corresponding dense matrix.

    Parameters
    ----------
    string : str
        The string representation of Pauli string.

    Returns
    -------
    np.ndarray
        The matrix representation of the Pauli string.

    """
    paulis = pauli_matrices()
    matrix = paulis[string[-1]]
    for p in reversed(string[:-1]):
        matrix = np.kron(paulis[p], matrix)
    return matrix
