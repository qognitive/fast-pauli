"""Test pauli objects from c++ against python implementations."""

import itertools as it

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
from tests.conftest import resolve_parameter_repr


@pytest.mark.parametrize("Pauli,", [(fp.Pauli)], ids=resolve_parameter_repr)
def test_pauli_wrapper(paulis: dict, Pauli: type) -> None:  # noqa: N803
    """Test pauli wrapper in python land."""
    np.testing.assert_array_equal(
        Pauli().to_tensor(),
        paulis["I"],
    )

    for i in [0, 1, 2, 3]:
        pcpp = Pauli(code=i)
        np.testing.assert_array_equal(
            pcpp.to_tensor(),
            paulis[i],
        )

    for c in ["I", "X", "Y", "Z"]:
        pcpp = Pauli(symbol=c)
        np.testing.assert_allclose(
            pcpp.to_tensor(),
            paulis[c],
            atol=1e-15,
        )
        np.testing.assert_string_equal(str(pcpp), c)


@pytest.mark.parametrize("Pauli,", [(fp.Pauli)], ids=resolve_parameter_repr)
def test_pauli_wrapper_multiply(paulis: dict, Pauli: type) -> None:  # noqa: N803
    """Test custom __mul__ in c++ wrapper."""
    for p1, p2 in it.product("IXYZ", repeat=2):
        c, pcpp = Pauli(p1).multiply(Pauli(p2))
        np.testing.assert_allclose(
            c * np.array(pcpp.to_tensor()),
            paulis[p1] @ paulis[p2],
            atol=1e-15,
        )


@pytest.mark.parametrize("Pauli,", [(fp.Pauli)], ids=resolve_parameter_repr)
def test_pauli_wrapper_exceptions(Pauli: type) -> None:  # noqa: N803
    """Test that exceptions from c++ are raised and propagated correctly."""
    with np.testing.assert_raises(ValueError):
        Pauli("II")
    with np.testing.assert_raises(ValueError):
        Pauli("A")
    with np.testing.assert_raises(ValueError):
        Pauli(-1)
    with np.testing.assert_raises(ValueError):
        Pauli(5)


if __name__ == "__main__":
    pytest.main()
