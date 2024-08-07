"""pytest configuration file for fast_pauli tests."""

import itertools as it

import numpy as np
import pytest

from fast_pauli.pypauli.helpers import pauli_matrices

# TODO: fixtures to wrap around numpy testing functions with default tolerances


@pytest.fixture
def paulis() -> dict[str | int, np.ndarray]:
    """Fixture to provide dict with Pauli matrices."""
    return pauli_matrices()  # type: ignore


@pytest.fixture
def sample_pauli_strings(limit_strings: int = 1_000) -> list[str]:
    """Fixture to provide sample Pauli strings for testing."""
    strings = it.chain(
        ["I", "X", "Y", "Z"],
        it.product("IXYZ", repeat=2),
        it.product("IXYZ", repeat=3),
        ["XYZXYZ", "ZZZIII", "XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"],
    )
    return list(map("".join, strings))[:limit_strings]


@pytest.fixture(scope="function")
def generate_random_complex(rng_seed: int = 321) -> np.ndarray:
    """Fixture to generate random complex numpy array with desired shape."""
    rng = np.random.default_rng(rng_seed)
    return lambda *shape: rng.random(shape) + 1j * rng.random(shape)


### TEST UTILITIES ###


def resolve_parameter_repr(val):  # type: ignore
    """Regular function to resolve representation for pytest parametrization."""
    module_name: str = getattr(val, "__module__", None)  # type: ignore
    if "_fast_pauli" in module_name:
        return val.__qualname__ + "-cpp"
    elif "pypauli" in module_name:
        return val.__qualname__ + "-py"
    return val
