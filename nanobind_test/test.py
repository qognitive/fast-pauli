import sys
import numpy as np
import pytest

sys.path.append("build")

import mdspan_wrapper


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
    my_op.scale(2.0)
    op2 = mdspan_wrapper.PauliOp(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]) * 2,
    )

    # my_op.scale(2.0)
    # print("After scaling, after op2")
    # my_op.print()

    # op3 = mdspan_wrapper.PauliOp(
    #     np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32),
    #     np.array([1.0, 2.0, 3.0]) * 4,
    # )
    # print("op3")
    # op3.print()

    # print("my_op")
    # my_op.print()

    # print("op2")
    # op2.print()
    assert my_op == op2


def test_multiply_coeff(my_op) -> None:
    coeffs = np.array([1.0, 2.0, 3.0])
    my_op.multiply_coeff(coeffs)
    op2 = mdspan_wrapper.PauliOp(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32),
        np.array([1.0, 4.0, 9.0]),
    )
    assert my_op == op2


def test_multiply_coeff_diff_type(my_op) -> None:
    coeffs = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
    with pytest.raises(TypeError):
        my_op.multiply_coeff(coeffs)