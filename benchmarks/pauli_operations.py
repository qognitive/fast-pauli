import numpy as np


class PauliString:
    def __init__(self, string: str) -> None:
        if not all([c in "IXYZ" for c in string]):
            raise ValueError(f"Invalid pauli string {string}")

        self.string = string
        self.dim = 1 << len(string)
        self.weight = len(string) - string.count("I")

    def dense(self) -> np.ndarray:
        columns, values = compose_sparse_pauli(self.string)

        matrix = np.zeros((columns.size, values.size), dtype=np.complex128)
        matrix[np.arange(columns.size), columns] = values
        return matrix

    def multiply(self, state: np.ndarray) -> np.ndarray:
        if state.shape[0] != self.dim or state.ndim > 2:
            raise ValueError(f"Provided state has inconsistent shape {state.shape}")

        columns, values = compose_sparse_pauli(self.string)

        if state.ndim == 2:
            return values[:, np.newaxis] * state[columns]
        else:
            return values * state[columns]


def compose_sparse_pauli(string: str) -> tuple[np.ndarray, np.ndarray]:
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


def compose_sparse_diag_pauli(string) -> np.ndarray:
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
