import pytest
import numpy as np

from pauli_operations import *


def test_sparse_pauli_string():
    assert PauliComposer(PauliString("IZ")).sparse_pauli() == SparsePauliString(1.0, np.array([0,1]), np.array([1.0, 1.0])) 

# TODO test correspondance to strings made from dense mutliplications


if __name__ == "__main__":
    pytest.main([__file__])