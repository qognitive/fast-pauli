#############################################################################
# This code is part of Fast Pauli.
#
# (C) Copyright Qognitive Inc 2024.
#
# This code is licensed under the BSD 2-Clause License. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#############################################################################


"""Test SummedPauliOp Python and C++ implementations."""

import pickle

import numpy as np
import pytest

import fast_pauli as fp
import fast_pauli.pypauli as pp
from tests.conftest import resolve_parameter_repr


@pytest.mark.parametrize("n_qubits,n_operators", [(2, 2), (3, 2), (4, 3)])
def test_ctors(n_qubits: int, n_operators: int) -> None:
    """Test the constructor of SummedPauliOp."""
    # Test with PauliStrings
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    coeffs_2d = np.random.rand(len(pauli_strings), n_operators).astype(np.complex128)
    op = fp.SummedPauliOp(pauli_strings, coeffs_2d)
    assert op.dim == 2**n_qubits
    assert op.n_operators == n_operators
    assert op.n_pauli_strings == len(pauli_strings)

    # Test with list of strings
    pauli_strings_str = [str(s) for s in pauli_strings]
    op = fp.SummedPauliOp(pauli_strings_str, coeffs_2d)
    assert op.dim == 2**n_qubits
    assert op.n_operators == n_operators
    assert op.n_pauli_strings == len(pauli_strings)


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_states,n_operators,n_qubits",
    [(s, o, q) for s in [1, 10, 1000] for o in [1, 10, 100] for q in [1, 2, 6]],
)
def test_apply(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_states: int,
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test applying the summed pauli operator method."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    n_strings = len(pauli_strings)

    coeffs_2d = np.random.rand(n_strings, n_operators).astype(np.complex128)
    psi = np.random.rand(2**n_qubits, n_states).astype(np.complex128)

    # For a single state, test that a 1D array works
    if n_states == 1:
        psi = psi[:, 0]

    op = summed_pauli_op(pauli_strings, coeffs_2d)

    # The new_states we want to check
    new_states = op.apply(psi)

    # Trusted new_states
    new_states_naive = np.zeros((2**n_qubits, n_states), dtype=np.complex128)
    # For a single state, test that a 1D array works
    if n_states == 1:
        new_states_naive = new_states_naive[:, 0]

    for k in range(n_operators):
        A_k = fp.PauliOp(coeffs_2d[:, k].copy(), pauli_strings)
        new_states_naive += A_k.apply(psi)

    # Check
    np.testing.assert_allclose(new_states, new_states_naive, atol=1e-13)


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_states,n_operators,n_qubits",
    [(s, o, q) for s in [1, 10, 1000] for o in [1, 10, 100] for q in [1, 2, 6]],
)
def test_apply_weighted(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_states: int,
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test applying the summed pauli operator method."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    n_strings = len(pauli_strings)

    coeffs_2d = np.random.rand(n_strings, n_operators).astype(np.complex128)
    psi = np.random.rand(2**n_qubits, n_states).astype(np.complex128)
    data_weights = np.random.rand(n_operators, n_states).astype(np.float64)

    # For a single state, test that a 1D array works
    if n_states == 1:
        psi = psi[:, 0]
        data_weights = data_weights[:, 0]

    op = summed_pauli_op(pauli_strings, coeffs_2d)

    # The new_states we want to check
    new_states = op.apply_weighted(psi, data_weights)

    # Trusted new_states
    new_states_naive = np.zeros((2**n_qubits, n_states), dtype=np.complex128)
    # For a single state, test that a 1D array works
    if n_states == 1:
        new_states_naive = new_states_naive[:, 0]

    for k in range(n_operators):
        A_k = fp.PauliOp(coeffs_2d[:, k].copy(), pauli_strings)
        new_states_naive += A_k.apply(psi) * data_weights[k]

    # Check
    np.testing.assert_allclose(new_states, new_states_naive, atol=1e-13)


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_states,n_operators,n_qubits",
    [(s, o, q) for s in [1, 10, 1000] for o in [1, 10, 100] for q in [1, 2, 6]],
)
def test_expectation_values(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_states: int,
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test expectation value calculation."""
    # TODO This will break on python version
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    pauli_strings_str = [str(s) for s in pauli_strings]
    n_strings = len(pauli_strings)

    coeffs_2d = np.random.rand(n_strings, n_operators).astype(np.complex128)

    if n_states == 1:
        psi = np.random.rand(2**n_qubits).astype(np.complex128)
        op = summed_pauli_op(pauli_strings, coeffs_2d)
        # The expectation values we want to test
        expectation_vals = op.expectation_value(psi)
        # The "trusted" expectation_values
        expectation_vals_naive = np.zeros((n_operators), dtype=np.complex128)
        for k in range(n_operators):
            A_k = pp.helpers.naive_pauli_operator(coeffs_2d[:, k], pauli_strings_str)
            expectation_vals_naive[k] = np.einsum("i,ij,j->", psi.conj(), A_k, psi)

    else:
        psi = np.random.rand(2**n_qubits, n_states).astype(np.complex128)
        op = summed_pauli_op(pauli_strings, coeffs_2d)
        # The expectation values we want to test
        expectation_vals = op.expectation_value(psi)
        # The "trusted" expectation_values
        expectation_vals_naive = np.zeros((n_operators, n_states), dtype=np.complex128)

        for k in range(n_operators):
            A_k = pp.helpers.naive_pauli_operator(coeffs_2d[:, k], pauli_strings_str)
            expectation_vals_naive[k] = np.einsum("it,ij,jt->t", psi.conj(), A_k, psi)

    # Check
    np.testing.assert_allclose(expectation_vals, expectation_vals_naive, atol=1e-13)


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_operators,n_qubits",
    [(o, q) for o in [1, 10, 100] for q in [1, 2, 6]],
)
def test_to_tensor(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test the dense representation of the SummedPauliOp."""
    # initialize SummedPauliOp
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    n_strings = len(pauli_strings)
    coeffs_2d = np.random.rand(n_strings, n_operators).astype(np.complex128)
    op = summed_pauli_op(pauli_strings, coeffs_2d)

    # get dense representation
    dense_op = op.to_tensor()

    # Check against doing it manually
    # Get dense representation of each PauliString
    ps_dense = np.array([ps.to_tensor() for ps in pauli_strings])

    summed_pauli_op_check = np.zeros(
        (n_operators, 2**n_qubits, 2**n_qubits), dtype=np.complex128
    )

    for k in range(n_operators):
        for j in range(n_strings):
            summed_pauli_op_check[k] += coeffs_2d[j, k] * ps_dense[j]

    np.testing.assert_allclose(dense_op, summed_pauli_op_check)


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_operators,n_qubits",
    [(o, q) for o in [1, 10] for q in [1, 2, 4, 6]],
)
def test_square(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test squaring the SummedPauliOp."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    coeffs_2d = np.random.rand(len(pauli_strings), n_operators).astype(np.complex128)
    op = summed_pauli_op(pauli_strings, coeffs_2d)
    op2 = op.square()

    A_k = op.to_tensor()
    A_k2 = op2.to_tensor()

    np.testing.assert_allclose(A_k2, np.einsum("kab,kbc->kac", A_k, A_k))


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_operators,n_qubits",
    [(o, q) for o in [1, 10] for q in [1, 2, 4, 6]],
)
def test_clone(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test clone method."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    coeffs_2d = np.random.rand(len(pauli_strings), n_operators).astype(np.complex128)
    op1 = summed_pauli_op(pauli_strings, coeffs_2d)
    op2 = op1.clone()

    np.testing.assert_array_equal(
        op1.to_tensor(),
        op2.to_tensor(),
    )
    assert id(op1) != id(op2)


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_operators,n_qubits",
    [(o, q) for o in [1, 10] for q in [1, 2, 4, 6]],
)
def test_coeffs_prop(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test getter and setter for coeffs property of the SummedPauliOp."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    orig_coeffs = np.random.rand(len(pauli_strings), n_operators).astype(np.complex128)
    spo = summed_pauli_op(pauli_strings, orig_coeffs)

    np.testing.assert_allclose(spo.coeffs, orig_coeffs.T)
    new_coeffs = np.random.rand(n_operators, len(pauli_strings)).astype(np.complex128)
    spo.coeffs = new_coeffs
    np.testing.assert_allclose(spo.coeffs, new_coeffs)
    np.testing.assert_allclose(
        spo.to_tensor(), summed_pauli_op(pauli_strings, new_coeffs.T.copy()).to_tensor()
    )


@pytest.mark.parametrize(
    "summed_pauli_op", [fp.SummedPauliOp], ids=resolve_parameter_repr
)
@pytest.mark.parametrize(
    "n_operators,n_qubits",
    [(o, q) for o in [1, 10] for q in [1, 2, 4, 6]],
)
def test_split(
    summed_pauli_op: type[fp.SummedPauliOp],
    n_operators: int,
    n_qubits: int,
) -> None:
    """Test split method of the SummedPauliOp."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(n_qubits, 2)
    coeffs_2d = np.random.rand(len(pauli_strings), n_operators).astype(np.complex128)
    spo = summed_pauli_op(pauli_strings, coeffs_2d)

    components = spo.split()
    for k, (comp, op_dense) in enumerate(zip(components, spo.to_tensor())):
        np.testing.assert_allclose(comp.coeffs, spo.coeffs[k])
        np.testing.assert_allclose(comp.to_tensor(), op_dense)


def test_pickle() -> None:
    """Test that SummedPauliOp objects can be pickled and unpickled."""
    pauli_strings = fp.helpers.calculate_pauli_strings_max_weight(2, 2)
    coeffs_2d = np.random.rand(len(pauli_strings), 2).astype(np.complex128)
    op = fp.SummedPauliOp(pauli_strings, coeffs_2d)
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    np.testing.assert_allclose(op.to_tensor(), unpickled.to_tensor())
