"""Test the helpers module."""

import itertools

import pytest

import fast_pauli as fp


def calculate_trusted_pauli_strings(n_qubits: int, weight: int) -> list[str]:
    """Calculate all possible Pauli strings for a given number of qubits and weight.

    Parameters
    ----------
    n_qubits : int
        The number of qubits.
    weight : int
        The Pauli weight.

    Returns
    -------
    list[str]
        All possible Pauli strings for the given number of qubits and weight.
    """
    strings = []
    nontrivial_matrix_elements = list(itertools.product(["X", "Y", "Z"], repeat=weight))
    for indices in itertools.combinations(range(n_qubits), weight):  # n(n-1)/2 terms
        for elements in nontrivial_matrix_elements:
            pauli_string = []
            for qbit in range(n_qubits):
                for el_position, i in enumerate(indices):
                    if i == qbit:
                        pauli_string.append(elements[el_position])
                        break
                else:
                    pauli_string.append("I")
            strings.append("".join(pauli_string))
    return strings


@pytest.mark.parametrize("weight", [0, 1, 2, 3])
def test_get_nontrivial_paulis(weight: int) -> None:
    """Test the get_nontrivial_paulis function."""
    res = fp.helpers.get_nontrivial_paulis(weight)
    trusted = [
        "".join(x) for x in itertools.product("XYZ", repeat=weight) if len(x) > 0
    ]
    assert len(res) == len(trusted)
    assert set(res) == set(trusted)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4], ids=lambda x: f"nq={x}")
@pytest.mark.parametrize("weight", [1, 2, 3], ids=lambda x: f"w={x}")
def test_calculate_pauli_strings(n_qubits: int, weight: int) -> None:
    """Test the calculate_pauli_strings function."""
    if n_qubits < weight:
        pytest.skip("n_qubits must be greater than or equal to weight")

    res = [str(x) for x in fp.helpers.calculate_pauli_strings(n_qubits, weight)]
    trusted = calculate_trusted_pauli_strings(n_qubits, weight)

    assert len(res) == len(trusted)
    assert set(res) == set(trusted)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4], ids=lambda x: f"nq={x}")
@pytest.mark.parametrize("weight", [1, 2, 3], ids=lambda x: f"w={x}")
def test_calculate_pauli_strings_max_weight(n_qubits: int, weight: int) -> None:
    """Test the calculate_pauli_strings_max_weight function."""
    if n_qubits < weight:
        pytest.skip("n_qubits must be greater than or equal to weight")

    res = [
        str(x) for x in fp.helpers.calculate_pauli_strings_max_weight(n_qubits, weight)
    ]
    trusted = ["I" * n_qubits]
    for i in range(1, weight + 1):
        trusted.extend(calculate_trusted_pauli_strings(n_qubits, i))

    assert len(res) == len(trusted)
    assert set(res) == set(trusted)
