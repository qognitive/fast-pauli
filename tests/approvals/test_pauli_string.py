"""Test pauli c++ objects against python implementations."""

import itertools as it
from typing import Callable

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli.operations as pp

# TODO this is going to be blended into main tests/test_pauli_string.py


@pytest.fixture
def sample_strings() -> list[str]:
    """Provide sample strings for testing."""
    strings = it.chain(
        ["I", "X", "Y", "Z"],
        it.permutations("IXYZ", 2),
        it.permutations("IXYZ", 3),
        ["ZIZI", "YZYZ", "XYZXYZ", "ZZZIII", "XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"],
    )
    return list(map("".join, strings))


def test_pauli_object(paulis: dict) -> None:
    """Test that C++ Pauli object is numerically equivalent to Python Pauli object."""
    # ideally, here we want to test against corresponding Pauli struct
    # from python land, but currently we don't have one
    for c in ["I", "X", "Y", "Z"]:
        p = fp.Pauli(c)
        np.testing.assert_array_equal(
            np.array(p.to_tensor()),
            paulis[c],
        )
        np.testing.assert_string_equal(str(p), c)


def test_pauli_string_representations(sample_strings: list[str]) -> None:
    """Test that C++ PauliString is numerically equivalent to Python PauliString."""
    for s in sample_strings:
        pcpp = fp.PauliString(s)
        ppy = pp.PauliString(s)

        np.testing.assert_allclose(
            pcpp.to_tensor(),
            ppy.dense(),
            atol=1e-15,
        )
        np.testing.assert_string_equal(str(pcpp), str(ppy))
        assert pcpp.dims == ppy.dim
        assert pcpp.weight == ppy.weight


def test_pauli_string_apply_batch(
    sample_strings: list[str], generate_random_complex: Callable
) -> None:
    """Test that C++ PauliString is numerically equivalent to Python PauliString."""
    for s in sample_strings:
        n_dim = 2 ** len(s)
        n_states = 42
        psis = generate_random_complex(n_dim, n_states)

        np.testing.assert_allclose(
            np.array(fp.PauliString(s).apply_batch(psis.tolist())),
            pp.PauliString(s).multiply(psis),
            atol=1e-15,
        )


if __name__ == "__main__":
    pytest.main([__file__])
