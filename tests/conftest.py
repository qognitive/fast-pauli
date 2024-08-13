"""pytest configuration file for fast_pauli tests."""

import itertools as it
from typing import Any, Callable

import numpy as np
import pytest

from fast_pauli.pypauli.helpers import pauli_matrices


@pytest.fixture
def paulis() -> dict[str | int, np.ndarray]:
    """Fixture to provide dict with Pauli matrices."""
    return pauli_matrices()  # type: ignore


@pytest.fixture
def sample_pauli_strings() -> list[str]:
    """Fixture to provide sample Pauli strings for testing."""
    strings = it.chain(
        ["I", "X", "Y", "Z"],
        it.product("IXYZ", repeat=2),
        it.product("IXYZ", repeat=3),
        ["XYZXYZ", "ZZZIII", "XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"],
    )
    return list(map("".join, strings))


@pytest.fixture
def pauli_strings_with_size() -> Callable:
    """Fixture to provide Pauli strings of desired size for testing."""

    def generate_paulis(size: int, limit: int = 1_000) -> list[str]:
        strings: list[str] = []
        for s in it.product("IXYZ", repeat=size):
            if limit and len(strings) > limit:
                break
            strings.append("".join(s))
        return strings

    return generate_paulis


@pytest.fixture(scope="function")
def generate_random_complex(rng_seed: int = 321) -> np.ndarray:
    """Fixture to generate random complex numpy array with desired shape."""
    rng = np.random.default_rng(rng_seed)
    return lambda *shape: rng.random(shape) + 1j * rng.random(shape)


### TEST UTILITIES ###


def resolve_parameter_repr(val: Any) -> Any | str:
    """Regular function to resolve representation for pytest parametrization.

    Currently, the main purpose is to automatically distinct cpp and py implementations
    for test logging
    """
    module_name: str | None = getattr(val, "__module__", None)
    if module_name is None:
        return val
    elif "_fast_pauli" in module_name:
        return val.__qualname__ + "-cpp"
    elif "pypauli" in module_name:
        return val.__qualname__ + "-py"
    return val
