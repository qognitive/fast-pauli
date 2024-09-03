"""Temp tests for nanobind interface."""

import numpy as np

from fast_pauli import fppy  # type: ignore


def test_pauli_string_apply() -> None:
    """Test the PauliString.apply method."""
    ps_str = "I" * 10
    ps = fppy.PauliString(ps_str)
    states = np.random.rand(2 ** len(ps_str)).astype(np.complex128)  # .reshape(2**4, 1)

    # Test 1d
    new_states = ps.apply(states, np.complex128(1.0))
    np.testing.assert_allclose(states, new_states[:, 0])

    # Test 2d
    n_data = 100
    states = np.random.rand(2 ** len(ps_str), n_data)
    new_states = ps.apply(states)
    np.testing.assert_allclose(states, new_states)
    print("Done with tests")


def test_pauli_string_expectation_value() -> None:
    """Test the PauliString.expectation_value method."""
    ps = fppy.PauliString("I" * 10)
    n_qubits = ps.n_qubits
    dim = 2**n_qubits
    n_data = 10
    states = np.random.rand(dim, n_data).astype(np.complex128)
    states /= np.linalg.norm(states, axis=0)

    ev = ps.expectation_value(states)
    np.testing.assert_allclose(ev, 1.0)


def test_summed_pauli_op() -> None:
    """Very rough tests for SummedPauliOp."""
    # pauli_strings = ["III", "ZZZ"] * 100
    pauli_strings = ["II"] * 50
    n_qubits = len(pauli_strings[0])
    dim = 2**n_qubits
    n_pauli_strings = len(pauli_strings)
    n_operators = 100
    n_data = 100

    coeffs = np.random.rand(n_pauli_strings, n_operators).astype(np.complex128)
    op = fppy.SummedPauliOp(pauli_strings, coeffs)

    data = np.random.rand(n_operators, n_data)
    states = np.random.rand(dim, n_data).astype(np.complex128)
    new_states = op.apply(states, data)
    print(new_states)
