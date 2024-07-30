"""Test pauli objects from c++ against python implementations."""

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp


def test_pauli_string_wrapper(paulis: dict) -> None:
    """Test pauli string wrapper in python land."""
    for empty_ps in [fp.PauliString(), fp.PauliString(""), fp.PauliString([])]:
        assert empty_ps.weight == 0
        assert empty_ps.dims == 0
        assert empty_ps.n_qubits == 0
        assert str(empty_ps) == ""
        assert len(empty_ps.to_tensor()) == 0

    # TODO


def test_pauli_string_exceptions() -> None:
    """Test that exceptions from c++ are raised and propagated correctly."""
    with np.testing.assert_raises(TypeError):
        fp.PauliString([1, 2])
    with np.testing.assert_raises(ValueError):
        fp.PauliString("ABC")
    with np.testing.assert_raises(ValueError):
        fp.PauliString("xyz")

    with np.testing.assert_raises(AttributeError):
        fp.PauliString("XYZ").dims = 99
    with np.testing.assert_raises(AttributeError):
        fp.PauliString("XYZ").weight = 99

    with np.testing.assert_raises(ValueError):
        fp.PauliString("II").apply([0.1, 0.2, 0.3])
    with np.testing.assert_raises(TypeError):
        fp.PauliString("II").apply_batch([0.1, 0.2, 0.3, 0.4])
    with np.testing.assert_raises(ValueError):
        fp.PauliString("XYZ").apply_batch(np.eye(4).tolist())


if __name__ == "__main__":
    pytest.main()
