"""Test pauli objects from c++ against python implementations."""

import itertools as it

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli.operations as pp


def test_pauli_wrapper(paulis: dict) -> None:
    """Test pauli wrapper in python land."""
    np.testing.assert_array_equal(
        np.array(fp.Pauli().to_tensor()),
        paulis["I"],
    )

    for i in [0, 1, 2, 3]:
        pcpp = fp.Pauli(code=i)
        np.testing.assert_array_equal(
            np.array(pcpp.to_tensor()),
            paulis[i],
        )

    for c in ["I", "X", "Y", "Z"]:
        pcpp = fp.Pauli(symbol=c)
        np.testing.assert_allclose(
            np.array(pcpp.to_tensor()),
            paulis[c],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            np.array(pcpp.to_tensor()),
            pp.PauliString(c).dense(),
            atol=1e-15,
        )
        np.testing.assert_string_equal(str(pcpp), c)
        np.testing.assert_string_equal(str(pcpp), str(pp.PauliString(c)))


def test_pauli_wrapper_multiply(paulis: dict) -> None:
    """Test custom __mul__ in c++ wrapper."""
    # TODO: counter-intuitive interface for * operator
    for p1, p2 in it.product("IXYZ", repeat=2):
        c, pcpp = fp.Pauli(p1) * fp.Pauli(p2)
        np.testing.assert_allclose(
            c * np.array(pcpp.to_tensor()),
            paulis[p1] @ paulis[p2],
            atol=1e-15,
        )


def test_pauli_wrapper_exceptions() -> None:
    """Test that exceptions from c++ are raised and propagated correctly."""
    with np.testing.assert_raises(ValueError):
        fp.Pauli("II")
    with np.testing.assert_raises(ValueError):
        fp.Pauli("A")
    with np.testing.assert_raises(ValueError):
        fp.Pauli(-1)
    with np.testing.assert_raises(ValueError):
        fp.Pauli(5)


if __name__ == "__main__":
    pytest.main([__file__])
