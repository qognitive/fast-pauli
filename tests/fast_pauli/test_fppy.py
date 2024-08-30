"""Temp tests for nanobind interface."""

import numpy as np

from fast_pauli import fppy  # type: ignore


def test_pauli_string_apply() -> None:
    """Test the PauliString.apply method."""
    ps = fppy.PauliString("IIII")
    states = np.random.rand(2**4).astype(np.complex128)  # .reshape(2**4, 1)

    # Test 1d
    new_states = ps.apply(states, np.complex128(1.0))
    np.testing.assert_allclose(states, new_states[:, 0])

    # Test 2d
    states = states.reshape(2**4, 1)
    new_states = ps.apply(states)
    np.testing.assert_allclose(states, new_states)


def test_pauli_string_expectation_value() -> None:
    """Test the PauliString.expectation_value method."""
    ps = fppy.PauliString("IIII")
    n_qubits = ps.n_qubits
    dim = 2**n_qubits
    n_data = 10
    states = np.random.rand(dim, n_data).astype(np.complex128)
    states /= np.linalg.norm(states, axis=0)

    ev = ps.expectation_value(states)
    np.testing.assert_allclose(ev, 1.0)
