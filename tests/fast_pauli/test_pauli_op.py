"""Test pauli operator objects from c++ and python."""

from typing import Callable

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp
from fast_pauli.pypauli.helpers import naive_pauli_converter, naive_pauli_operator
from tests.conftest import resolve_parameter_repr


def test_naive_pauli_operator(paulis: dict) -> None:
    """Test naive pauli operator helper for dense matrix representation."""
    for p in ["I", "X", "Y", "Z"]:
        np.testing.assert_equal(naive_pauli_operator([2], [p]), 2 * paulis[p])

    np.testing.assert_equal(naive_pauli_operator([1], ["III"]), np.eye(8))

    np.testing.assert_equal(
        naive_pauli_operator([2, 2, 2, 2], ["I", "X", "Y", "Z"]),
        2 * np.sum([paulis["I"], paulis["X"], paulis["Y"], paulis["Z"]], axis=0),
    )

    np.testing.assert_equal(
        naive_pauli_operator([3j, 3j, 3j], ["XY", "ZI", "YZ"]),
        3j
        * np.sum(
            [
                np.kron(paulis["X"], paulis["Y"]),
                np.kron(paulis["Z"], paulis["I"]),
                np.kron(paulis["Y"], paulis["Z"]),
            ],
            axis=0,
        ),
    )


@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_operator_trivial(
    paulis: dict,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test PauliOp class on trivial inputs."""
    empty = (
        [pauli_op(), pauli_op([]), pauli_op([], [])]
        if isinstance(pauli_op, fp.PauliOp)
        else [pauli_op([], [])]
    )
    for empty_op in empty:
        assert empty_op.dim == 0
        assert empty_op.n_qubits == 0
        assert empty_op.n_pauli_strings == 0
        assert len(empty_op.to_tensor()) == 0
        assert len(empty_op.coeffs) == 0
        assert len(empty_op.pauli_strings) == 0

    for p in ["I", "X", "Y", "Z"]:
        po = pauli_op([1.0], [p])
        assert po.dim == 2
        assert po.n_qubits == 1
        assert po.n_pauli_strings == 1
        # assert po.pauli_strings == [p]
        np.testing.assert_allclose(po.to_tensor(), paulis[p], atol=1e-15)


@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_operator_basics(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test PauliOp class in python land."""
    po = pauli_op([4j, 4j], ["III", "III"])
    assert po.dim == 8
    assert po.n_qubits == 3
    assert po.n_pauli_strings == 2
    assert po.pauli_strings_as_str == ["III", "III"]
    np.testing.assert_equal(po.coeffs, [4j, 4j])
    np.testing.assert_allclose(po.to_tensor(), 4j * 2 * np.eye(8), atol=1e-15)

    string_sets = [
        ["I", "X", "Y", "Z"],
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        ["XYZXYZXYZ", "ZZZIIIXXX"],
    ]
    if isinstance(po, fp.PauliOp):
        string_sets += [
            [fp.PauliString("XYXYX"), fp.PauliString("YXYXY"), fp.PauliString("ZXZXZ")],
        ]

    for strings in string_sets:
        coeffs = generate_random_complex(len(strings))
        po = pauli_op(coeffs, strings)

        n_qubits = (
            len(strings[0]) if isinstance(strings[0], str) else strings[0].n_qubits
        )
        assert po.dim == 2**n_qubits
        assert po.n_qubits == n_qubits
        assert po.n_pauli_strings == len(strings)

        assert set(po.pauli_strings_as_str) == set([str(ps) for ps in strings])
        np.testing.assert_allclose(
            po.coeffs,
            coeffs,
            atol=1e-15,
        )

        np.testing.assert_allclose(
            po.to_tensor(),
            naive_pauli_operator(coeffs, [str(ps) for ps in strings]),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            po.to_tensor(),
            pauli_op(list(reversed(coeffs)), list(reversed(strings))).to_tensor(),
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_dense_representation(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test dense representation of Pauli Operator against naive approach."""
    for strings in [
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        pauli_strings_with_size(4),
        pauli_strings_with_size(7, limit=128),
        pauli_strings_with_size(10, limit=64),
    ]:
        coeffs = generate_random_complex(len(strings))
        po = pauli_op(coeffs, strings)

        np.testing.assert_allclose(
            po.to_tensor(),
            naive_pauli_operator(coeffs, strings),
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_apply_1d(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test Pauli Operator multiplication with vector."""
    np.testing.assert_allclose(
        pauli_op([1, 2], ["IXYZ", "ZYXI"]).apply(np.zeros(16)),
        np.zeros(16),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pauli_op([0.5, 0.5], ["III", "III"]).apply(np.arange(8)),
        np.arange(8),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pauli_op([1j], ["XYZ"]).apply(np.ones(8)),
        naive_pauli_operator([1j], ["XYZ"]).sum(1),
        atol=1e-15,
    )

    for strings in [
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        pauli_strings_with_size(4),
        pauli_strings_with_size(4),
        pauli_strings_with_size(7, limit=128),
        pauli_strings_with_size(8, limit=200)[:100],
        pauli_strings_with_size(8, limit=200)[100:],
        pauli_strings_with_size(10, limit=64),
    ]:
        coeffs = generate_random_complex(len(strings))
        po = pauli_op(coeffs, strings)

        psi = generate_random_complex(po.dim)
        np.testing.assert_allclose(
            po.apply(psi),
            naive_pauli_operator(coeffs, strings).dot(psi),
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_apply_batch(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test Pauli Operator multiplication with a batch of vectors."""
    np.testing.assert_allclose(
        pauli_op([1, 2], ["IXYZ", "ZYXI"]).apply(np.zeros((16, 16))),
        np.zeros((16, 16)),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pauli_op([0.5, 0.5], ["III", "III"]).apply(np.arange(8 * 8).reshape(8, 8)),
        np.arange(8 * 8).reshape(8, 8),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pauli_op([1j], ["XYZ"]).apply(np.eye(8)),
        naive_pauli_operator([1j], ["XYZ"]),
        atol=1e-15,
    )

    for strings in [
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        pauli_strings_with_size(4),
        pauli_strings_with_size(7, limit=128),
        pauli_strings_with_size(8, limit=200)[:100],
        pauli_strings_with_size(8, limit=200)[100:],
        pauli_strings_with_size(10, limit=64),
    ]:
        n_states = int(100 * generate_random_complex(1)[0].real)
        coeffs = generate_random_complex(len(strings))
        po = pauli_op(coeffs, strings)

        psis = generate_random_complex(po.dim, n_states)
        np.testing.assert_allclose(
            po.apply(psis),
            naive_pauli_operator(coeffs, strings) @ psis,
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_expectation_value(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test Pauli Operator expected value."""
    np.testing.assert_allclose(
        pauli_op([1, 2], ["IXYZ", "ZYXI"]).expectation_value(np.zeros(16)),
        0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pauli_op([1, 2], ["IXYZ", "ZYXI"]).expectation_value(np.zeros((16, 16))),
        np.zeros(16),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        pauli_op([1, 1], ["III", "III"]).expectation_value(np.arange(8)),
        2 * np.square(np.arange(8)).sum(),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pauli_op([1, 1], ["III", "III"]).expectation_value(
            np.arange(8 * 8).reshape(8, 8)
        ),
        2 * np.square(np.arange(8 * 8)).reshape(8, 8).sum(0),
        atol=1e-15,
    )

    for strings in [
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        pauli_strings_with_size(4),
        pauli_strings_with_size(7, limit=128),
        pauli_strings_with_size(8, limit=200)[:100],
        pauli_strings_with_size(8, limit=200)[100:],
        pauli_strings_with_size(10, limit=64),
    ]:
        coeffs = generate_random_complex(len(strings))
        po = pauli_op(coeffs, strings)

        psi = generate_random_complex(po.dim)
        np.testing.assert_allclose(
            po.expectation_value(psi),
            naive_pauli_operator(coeffs, strings).dot(psi).dot(psi.conj()),
            atol=1e-15,
        )

        n_states = int(100 * generate_random_complex(1)[0].real)
        psis = generate_random_complex(n_states, po.dim)
        # compute <psi_t|A|psi_t>
        expected = np.einsum(
            "ti,ij,tj->t", psis.conj(), naive_pauli_operator(coeffs, strings), psis
        )
        np.testing.assert_allclose(
            po.expectation_value(psis.T.copy()),
            expected,
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(  # TODO pp.PauliOp
    "pauli_op,pauli_string,", [(fp.PauliOp, fp.PauliString)], ids=resolve_parameter_repr
)
def test_multiplication_with_string(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
    pauli_string: type[fp.PauliString] | type[pp.PauliString],
) -> None:
    """Test Pauli Operator multiplication with another Pauli Operator."""
    ixyz_op = pauli_op([1, 1, 1, 1], ["I", "X", "Y", "Z"])
    ixyz_expected = naive_pauli_operator([1, 1, 1, 1], ["I", "X", "Y", "Z"])

    np.testing.assert_allclose(
        (ixyz_op * pauli_string("I")).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (ixyz_op * pauli_string("I") * pauli_string("I")).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_string("I") * ixyz_op * pauli_string("I")).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )

    for strings in [
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        pauli_strings_with_size(4),
        pauli_strings_with_size(7, limit=128),
        pauli_strings_with_size(8, limit=200)[:100],
        pauli_strings_with_size(8, limit=200)[100:],
    ]:
        choose_string = int(generate_random_complex(1)[0].real * 100) % len(strings)
        coeffs = generate_random_complex(len(strings))
        p_str, coeffs = strings.pop(choose_string), np.delete(coeffs, choose_string)
        p_op = pauli_op(coeffs, strings)

        np.testing.assert_allclose(
            (p_op * pauli_string(p_str)).to_tensor(),
            naive_pauli_operator(coeffs, strings) @ naive_pauli_converter(p_str),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            (pauli_string(p_str) * p_op).to_tensor(),
            naive_pauli_converter(p_str) @ naive_pauli_operator(coeffs, strings),
            atol=1e-15,
        )

    for strings, n_mult in [
        (pauli_strings_with_size(3), 25),
        (pauli_strings_with_size(4), 20),
        (pauli_strings_with_size(7, limit=64), 10),
        (pauli_strings_with_size(10, limit=32), 5),
    ]:
        coeffs = generate_random_complex(len(strings))
        p_op = pauli_op(coeffs, strings)
        expected_op = naive_pauli_operator(coeffs, strings)

        for _ in range(n_mult):
            choose_string = int(generate_random_complex(1)[0].real * 100) % len(strings)
            p_str = strings[choose_string]
            p_op = p_op * pauli_string(p_str)
            expected_op @= naive_pauli_converter(p_str)

            np.testing.assert_allclose(
                p_op.to_tensor(),
                expected_op,
                atol=1e-15,
            )


# TODO test exceptions for PauliOp

if __name__ == "__main__":
    pytest.main()
