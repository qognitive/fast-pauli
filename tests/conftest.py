"""pytest configuration file for fast_pauli tests."""

import numpy as np
import pytest

from fast_pauli.pypauli.helpers import pauli_matrices

# TODO: fixtures to wrap around numpy testing functions with default tolerances


# TODO: make this as global fixture shared with pypauli unit tests
@pytest.fixture
def paulis() -> dict[str | int, np.ndarray]:
    """Fixture to provide dict with Pauli matrices."""
    return pauli_matrices()  # type: ignore


@pytest.fixture(scope="function")
def generate_random_complex(rng_seed: int = 321) -> np.ndarray:
    """Generate random complex numpy array with desired shape."""
    rng = np.random.default_rng(rng_seed)
    return lambda *shape: rng.random(shape) + 1j * rng.random(shape)
