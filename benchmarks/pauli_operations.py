import numpy as np
from dataclasses import dataclass


@dataclass
class PauliString:
    string: str
    weight: float = 1.0

@dataclass
class SparseMatrix:
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

@dataclass
class SparsePauliString:
    columns: np.ndarray
    values: np.ndarray
    weight: float = 1.0


class PauliComposer:
    def __init__(self, pauli: PauliString) -> None:
        self.pauli = pauli
        self.n_qubits = len(pauli.string)
        self.n_vals = 1 << self.n_qubits
        self.n_ys = pauli.string.count("Y")


    def __resolve_init_conditions(self) -> None:
        first_col = 0
        for p in self.pauli.string:
            first_col <<= 1
            if p == "X" or p == "Y":
                first_col += 1

        match self.n_ys % 4:
            case 0:
                first_val = 1.0
            case 1:
                first_val = -1.0j
            case 2:
                first_val = -1.0
            case 3:
                first_val = 1.0j
        
        return first_col, first_val


    def sparse_pauli(self) -> SparsePauliString:
        cols = np.empty(self.n_vals, dtype=np.int32)
        vals = np.empty(self.n_vals, dtype=np.complex128)
        cols[0], vals[0] = self.__resolve_init_conditions()

        for l in range(self.n_qubits):
            p = self.pauli.string[self.n_qubits - l - 1]
            pow_of_two = 1 << l
            
            new_slice = slice(pow_of_two, 2*pow_of_two)
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

        return SparsePauliString(weight=self.pauli.weight, columns=cols, values=vals)


    def sparse_diag_pauli(self) -> SparsePauliString:
        assert self.pauli.string.count("X") + self.pauli.string.count("Y") == 0

        cols = np.arange(self.n_vals, dtype=np.int32)
        vals = np.ones(self.n_vals, dtype=np.complex128)

        for l in range(self.n_qubits):
            p = self.pauli.string[self.n_qubits - l - 1]
            pow_of_two = 1 << l
            
            new_slice = slice(pow_of_two, 2*pow_of_two)
            old_slice = slice(0, pow_of_two)

            match p:
                case "I":
                    vals[new_slice] = vals[old_slice]
                case "Z":
                    vals[new_slice] = -vals[old_slice]

        return SparsePauliString(weight=self.pauli.weight, columns=cols, values=vals)


    def efficient_sparse_multiply(self, state: np.ndarray) -> np.ndarray:
        assert state.ndim == 2

        cols = np.empty(self.n_vals, dtype=np.int32)
        vals = np.empty(self.n_vals, dtype=np.complex128)
        cols[0], vals[0] = self.__resolve_init_conditions()

        product = np.empty((self.n_vals, state.shape[1]), dtype=np.complex128)
        product[0] = self.pauli.weight * vals[0] * state[cols[0]]

        for l in range(self.n_qubits):
            p = self.pauli.string[self.n_qubits - l - 1]
            pow_of_two = 1 << l
            
            new_slice = slice(pow_of_two, 2*pow_of_two)
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

            product[new_slice] = self.pauli.weight * vals[new_slice, np.newaxis] * state[cols[new_slice]]

        return product


def sparse_pauli_multiply(pauli_string: SparsePauliString, state: np.ndarray) -> np.ndarray:
    if state.ndim == 1:
        return pauli_string.weight * pauli_string.values * state[pauli_string.columns]
    elif state.ndim == 2:
        return pauli_string.weight * pauli_string.values[:, np.newaxis] * state[pauli_string.columns]
    else:
        raise ValueError("state must be a 1D or 2D array")


def dense_pauli_string():
    pass


def dense_pauli_multiply():
    pass
