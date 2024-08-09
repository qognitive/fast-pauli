import sys
import numpy as np
import pytest

sys.path.append("build")

import mdspan_wrapper


def test_scale_ndarray() -> None:
    size = 10
    scale = 1.34

    arr_np = np.random.rand(size, size, size)
    arr_np_scaled = arr_np * scale

    arr_mdspan = arr_np.copy()
    mdspan_wrapper.scale_ndarray(arr_mdspan, scale)

    np.testing.assert_allclose(arr_mdspan, arr_np_scaled)


def test_fake_pauli_op() -> None:
    pauli_strings = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    coeffs = np.array([1.0, 2.0, 3.0])
    op = mdspan_wrapper.PauliOp(pauli_strings, coeffs)

    print(coeffs)

    op.scale(2.0)
    print(coeffs)


@pytest.fixture
def my_op() -> mdspan_wrapper.PauliOp:
    pauli_strings = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    coeffs = np.array([1.0, 2.0, 3.0])
    return mdspan_wrapper.PauliOp(pauli_strings, coeffs)


def test_fake_pauli_op_scope(my_op) -> None:
    my_op.print()
    my_op.scale(2.0)
    my_op.print()
    my_op.print()
    my_op.print()

    pauli_strings = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    coeffs = np.array([1.0, 2.0, 3.0]) * 2
    op2 = mdspan_wrapper.PauliOp(pauli_strings, coeffs)
    op2.print()

    assert my_op == op2
