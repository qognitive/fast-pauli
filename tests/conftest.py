"""pytest configuration file for fast_pauli tests."""

import numpy as np
import pytest

from fast_pauli.pypauli.helpers import pauli_matrices


# TODO: make this as global fixture shared with pypauli unit tests
@pytest.fixture
def paulis() -> dict[str | int, np.ndarray]:
    """Fixture to provide dict with Pauli matrices."""
    return pauli_matrices()  # type: ignore
