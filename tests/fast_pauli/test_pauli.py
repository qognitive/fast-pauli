"""Test pauli objects from c++ against python implementations."""

import itertools as it

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
from tests.conftest import resolve_parameter_repr


@pytest.mark.parametrize("pauli,", [(fp.Pauli)], ids=resolve_parameter_repr)
def test_basics(paulis: dict, pauli: type[fp.Pauli]) -> None:
    """Test pauli wrapper in python land."""
    np.testing.assert_array_equal(
        pauli().to_tensor(),
        paulis["I"],
    )

    for i in [0, 1, 2, 3]:
        pcpp = pauli(code=i)
        np.testing.assert_array_equal(
            pcpp.to_tensor(),
            paulis[i],
        )

    for c in ["I", "X", "Y", "Z"]:
        pcpp = pauli(symbol=c)
        np.testing.assert_allclose(
            pcpp.to_tensor(),
            paulis[c],
            atol=1e-15,
        )
        np.testing.assert_string_equal(str(pcpp), c)


@pytest.mark.parametrize("pauli,", [(fp.Pauli)], ids=resolve_parameter_repr)
def test_multiply(paulis: dict, pauli: type[fp.Pauli]) -> None:
    """Test custom __mul__ in c++ wrapper."""
    for p1, p2 in it.product("IXYZ", repeat=2):
        c, pcpp = pauli(p1) @ pauli(p2)
        np.testing.assert_allclose(
            c * np.array(pcpp.to_tensor()),
            paulis[p1] @ paulis[p2],
            atol=1e-15,
        )


@pytest.mark.parametrize("pauli,", [(fp.Pauli)], ids=resolve_parameter_repr)
def test_exceptions(pauli: type[fp.Pauli]) -> None:
    """Test that exceptions from c++ are raised and propagated correctly."""
    with np.testing.assert_raises(TypeError):
        pauli("II")
    with np.testing.assert_raises(ValueError):
        pauli("A")
    with np.testing.assert_raises(ValueError):
        pauli(-1)
    with np.testing.assert_raises(ValueError):
        pauli(5)


if __name__ == "__main__":
    pytest.main()
