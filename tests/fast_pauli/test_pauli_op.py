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
@pytest.mark.parametrize(
    "pauli_op,pauli_string,",
    [(fp.PauliOp, fp.PauliString), (pp.PauliOp, pp.PauliString)],
    ids=resolve_parameter_repr,
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
        (ixyz_op @ pauli_string("I")).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (ixyz_op @ pauli_string("I") @ pauli_string("I")).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_string("I") @ ixyz_op @ pauli_string("I")).to_tensor(),
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
            (p_op @ pauli_string(p_str)).to_tensor(),
            naive_pauli_operator(coeffs, strings) @ naive_pauli_converter(p_str),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            (pauli_string(p_str) @ p_op).to_tensor(),
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
            p_op = p_op @ pauli_string(p_str)
            expected_op @= naive_pauli_converter(p_str)

            np.testing.assert_allclose(
                p_op.to_tensor(),
                expected_op,
                atol=1e-15,
            )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_multiplication_with_pauli_op(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test Pauli Operator multiplication with another Pauli Operator."""
    ixyz_op = pauli_op([1, 1, 1, 1], ["I", "X", "Y", "Z"])
    ixyz_expected = naive_pauli_operator([1, 1, 1, 1], ["I", "X", "Y", "Z"])

    np.testing.assert_allclose(
        (pauli_op([1], ["I"]) @ pauli_op([1], ["I"])).to_tensor(),
        np.eye(2),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (ixyz_op @ pauli_op([1], ["I"])).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_op([1], ["I"]) @ ixyz_op @ pauli_op([1], ["I"])).to_tensor(),
        ixyz_expected,
        atol=1e-15,
    )

    rng = np.random.RandomState(42)
    for strings, variations in [
        (pauli_strings_with_size(2), 15),
        (pauli_strings_with_size(3), 10),
        (pauli_strings_with_size(4), 7),
        (pauli_strings_with_size(7, limit=128), 5),
        (pauli_strings_with_size(8, limit=200)[100:], 3),
        (pauli_strings_with_size(10, limit=32), 2),
    ]:
        for _ in range(variations):
            l_size = rng.choice(len(strings)) or 1
            r_size = rng.choice(len(strings)) or 1
            l_strings = rng.choice(strings, l_size)
            r_strings = rng.choice(strings, r_size)
            l_coeffs = generate_random_complex(l_size)
            r_coeffs = generate_random_complex(r_size)
            l_op = pauli_op(l_coeffs, l_strings)
            r_op = pauli_op(r_coeffs, r_strings)

            np.testing.assert_allclose(
                (l_op @ r_op).to_tensor(),
                naive_pauli_operator(l_coeffs, l_strings)
                @ naive_pauli_operator(r_coeffs, r_strings),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (r_op @ r_op).to_tensor(),
                naive_pauli_operator(r_coeffs, r_strings)
                @ naive_pauli_operator(r_coeffs, r_strings),
                atol=1e-15,
            )

        np.testing.assert_allclose(
            (l_op @ r_op @ l_op @ r_op).to_tensor(),
            naive_pauli_operator(l_coeffs, l_strings)
            @ naive_pauli_operator(r_coeffs, r_strings)
            @ naive_pauli_operator(l_coeffs, l_strings)
            @ naive_pauli_operator(r_coeffs, r_strings),
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,pauli_string,",
    [(fp.PauliOp, fp.PauliString), (pp.PauliOp, pp.PauliString)],
    ids=resolve_parameter_repr,
)
def test_pauli_string_conversion(
    pauli_strings_with_size: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
    pauli_string: type[fp.PauliString] | type[pp.PauliString],
) -> None:
    """Test that addition of two pauli strings is resulting in expected pauli_op."""
    rng = np.random.RandomState(42)
    all_3qubit_strings = pauli_strings_with_size(3)
    all_4qubit_strings = pauli_strings_with_size(3)
    rng.shuffle(all_3qubit_strings)
    rng.shuffle(all_4qubit_strings)

    for string_set in [all_3qubit_strings, all_4qubit_strings]:
        while len(string_set) >= 2:
            l_str = string_set.pop()
            r_str = string_set.pop()

            np.testing.assert_allclose(
                (pauli_string(l_str) + pauli_string(r_str)).to_tensor(),
                naive_pauli_converter(l_str) + naive_pauli_converter(r_str),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (pauli_string(l_str) + pauli_string(r_str)).to_tensor(),
                pauli_op([1, 1], [l_str, r_str]).to_tensor(),
                atol=1e-15,
            )

            np.testing.assert_allclose(
                (pauli_string(l_str) - pauli_string(r_str)).to_tensor(),
                naive_pauli_converter(l_str) - naive_pauli_converter(r_str),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (pauli_string(l_str) - pauli_string(r_str)).to_tensor(),
                pauli_op([1, -1], [l_str, r_str]).to_tensor(),
                atol=1e-15,
            )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,pauli_string,",
    [(fp.PauliOp, fp.PauliString), (pp.PauliOp, pp.PauliString)],
    ids=resolve_parameter_repr,
)
def test_add_sub_with_string(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
    pauli_string: type[fp.PauliString] | type[pp.PauliString],
) -> None:
    """Test Pauli Operator addition with Pauli String."""
    ixyz_op = pauli_op([1, 1, 1, 1], ["I", "X", "Y", "Z"])

    np.testing.assert_allclose(
        (ixyz_op + pauli_string("X")).to_tensor(),
        naive_pauli_operator([1, 2, 1, 1], ["I", "X", "Y", "Z"]),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_string("I") + ixyz_op + pauli_string("I")).to_tensor(),
        naive_pauli_operator([3, 1, 1, 1], ["I", "X", "Y", "Z"]),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_string("Z") + ixyz_op - pauli_string("Z")).to_tensor(),
        naive_pauli_operator([1, 1, 1, 1], ["I", "X", "Y", "Z"]),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (
            pauli_op([1j, 1j, 1j], ["X", "Y", "Z"])
            + pauli_op([1j, 1j, 1j], ["Z", "Y", "X"])
        ).to_tensor(),
        naive_pauli_operator([2j, 2j, 2j], ["Y", "X", "Z"]),
        atol=1e-15,
    )

    ixyz_op += pauli_string("Z")
    np.testing.assert_allclose(
        (ixyz_op).to_tensor(),
        naive_pauli_operator([1, 1, 1, 2], ["I", "X", "Y", "Z"]),
        atol=1e-15,
    )
    ixyz_op += pauli_string("X")
    np.testing.assert_allclose(
        (ixyz_op).to_tensor(),
        naive_pauli_operator([1, 2, 1, 2], ["I", "X", "Y", "Z"]),
        atol=1e-15,
    )
    ixyz_op -= pauli_string("Z")
    ixyz_op -= pauli_string("X")
    np.testing.assert_allclose(
        (ixyz_op).to_tensor(),
        naive_pauli_operator([1, 1, 1, 1], ["I", "X", "Y", "Z"]),
        atol=1e-15,
    )

    n_passes = 3
    for strings in [
        pauli_strings_with_size(2),
        pauli_strings_with_size(3),
        pauli_strings_with_size(4),
        pauli_strings_with_size(7, limit=128),
        pauli_strings_with_size(8, limit=200)[100:],
        pauli_strings_with_size(10, limit=32),
    ]:
        for _ in range(n_passes):
            choose_string = int(generate_random_complex(1)[0].real * len(strings))
            coeffs = generate_random_complex(len(strings))
            p_str, coeffs = strings.pop(choose_string), np.delete(coeffs, choose_string)
            p_op = pauli_op(coeffs, strings)
            expected_dense_op = naive_pauli_operator(coeffs, strings)
            p_str_dense = naive_pauli_converter(p_str)

            np.testing.assert_allclose(
                (p_op + pauli_string(p_str)).to_tensor(),
                expected_dense_op + p_str_dense,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (pauli_string(p_str) + p_op).to_tensor(),
                p_str_dense + expected_dense_op,
                atol=1e-15,
            )

            np.testing.assert_allclose(
                (pauli_string(p_str) - p_op).to_tensor(),
                p_str_dense - expected_dense_op,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (pauli_string(p_str) + p_op - pauli_string(p_str)).to_tensor(),
                expected_dense_op,
                atol=1e-15,
            )

        choose_string = choose_string % len(strings)
        p_str, coeffs = strings.pop(choose_string), np.delete(coeffs, choose_string)
        p_op = pauli_op(coeffs, strings)
        expected = naive_pauli_operator(coeffs, strings)
        p_str_dense = naive_pauli_converter(p_str)

        p_op += pauli_string(p_str)
        expected += p_str_dense
        np.testing.assert_allclose(
            p_op.to_tensor(),
            expected,
            atol=1e-15,
        )
        p_op -= pauli_string(p_str)
        expected -= p_str_dense
        np.testing.assert_allclose(
            p_op.to_tensor(),
            expected,
            atol=1e-15,
        )

        p_op = pauli_op(coeffs, strings)
        c = generate_random_complex(1)[0]
        p_op.extend(pauli_string(p_str), c, dedupe=False)
        expected += c * p_str_dense
        np.testing.assert_allclose(
            p_op.to_tensor(),
            expected,
            atol=1e-15,
        )
        c = generate_random_complex(1)[0]
        p_op.extend(pauli_string(p_str), c, dedupe=True)
        expected += c * p_str_dense
        np.testing.assert_allclose(
            p_op.to_tensor(),
            expected,
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,", [fp.PauliOp, pp.PauliOp], ids=resolve_parameter_repr
)
def test_add_sub_with_pauli_op(
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
) -> None:
    """Test Pauli Operator addition with another Pauli Operator."""
    np.testing.assert_allclose(
        (pauli_op([1], ["II"]) + pauli_op([2], ["II"])).to_tensor(),
        3 * np.eye(4),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_op([1j], ["X"]) + pauli_op([1j], ["Y"])).to_tensor(),
        naive_pauli_operator([1j, 1j], ["X", "Y"]),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (pauli_op([1j], ["X"]) - pauli_op([1j], ["Y"])).to_tensor(),
        naive_pauli_operator([1j, -1j], ["X", "Y"]),
        atol=1e-15,
    )

    ixyz_op = pauli_op([1, 1, 1, 1], ["I", "X", "Y", "Z"])
    ixyz_expected = naive_pauli_operator([1, 1, 1, 1], ["I", "X", "Y", "Z"])

    np.testing.assert_allclose(
        (ixyz_op + ixyz_op - ixyz_op + ixyz_op).to_tensor(),
        2 * ixyz_expected,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (ixyz_op - ixyz_op).to_tensor(),
        naive_pauli_operator([0], ["X"]),
        atol=1e-15,
    )

    ixyz_op += pauli_op([1, 1, 1, 1], ["I", "X", "Y", "Z"])
    ixyz_op += ixyz_op
    np.testing.assert_allclose(
        ixyz_op.to_tensor(),
        4 * ixyz_expected,
        atol=1e-15,
    )

    ixyz_op -= pauli_op([4, 4], ["X", "Z"])
    np.testing.assert_allclose(
        ixyz_op.to_tensor(),
        naive_pauli_operator([4, 4], ["Y", "I"]),
        atol=1e-15,
    )
    ixyz_op -= pauli_op([3, 3], ["Y", "I"])
    np.testing.assert_allclose(
        ixyz_op.to_tensor(),
        naive_pauli_operator([1, 1], ["Y", "I"]),
        atol=1e-15,
    )

    rng = np.random.RandomState(42)
    for strings, variations in [
        (pauli_strings_with_size(2), 15),
        (pauli_strings_with_size(3), 10),
        (pauli_strings_with_size(4), 7),
        (pauli_strings_with_size(7, limit=128), 5),
        (pauli_strings_with_size(8, limit=200)[100:], 3),
        (pauli_strings_with_size(10, limit=32), 2),
    ]:
        for _ in range(variations):
            l_size = rng.choice(len(strings)) or 1
            r_size = rng.choice(len(strings)) or 1
            l_strings = rng.choice(strings, l_size)
            r_strings = rng.choice(strings, r_size)
            l_coeffs = generate_random_complex(l_size)
            r_coeffs = generate_random_complex(r_size)
            l_op = pauli_op(l_coeffs, l_strings)
            r_op = pauli_op(r_coeffs, r_strings)
            l_op_expected_dense = naive_pauli_operator(l_coeffs, l_strings)
            r_op_expected_dense = naive_pauli_operator(r_coeffs, r_strings)

            np.testing.assert_allclose(
                (l_op + r_op).to_tensor(),
                l_op_expected_dense + r_op_expected_dense,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (r_op + r_op).to_tensor(),
                2 * r_op_expected_dense,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                (l_op - r_op).to_tensor(),
                l_op_expected_dense - r_op_expected_dense,
                atol=1e-15,
            )

        np.testing.assert_allclose(
            ((l_op + r_op) + (l_op - r_op)).to_tensor(),
            l_op_expected_dense
            + r_op_expected_dense
            + l_op_expected_dense
            - r_op_expected_dense,
            atol=1e-13,
        )

        l_op += r_op
        np.testing.assert_allclose(
            (l_op).to_tensor(),
            l_op_expected_dense + r_op_expected_dense,
            atol=1e-13,
        )
        l_op -= r_op
        np.testing.assert_allclose(
            (l_op).to_tensor(),
            l_op_expected_dense,
            atol=1e-13,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "pauli_op,pauli_string,",
    [(fp.PauliOp, fp.PauliString), (pp.PauliOp, pp.PauliString)],
    ids=resolve_parameter_repr,
)
def test_exceptions(
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
    pauli_string: type[fp.PauliString] | type[pp.PauliString],
) -> None:
    """Test that exceptions are raised and propagated correctly."""
    if isinstance(pauli_op, fp.PauliOp):
        with np.testing.assert_raises(ValueError):
            pauli_op(["ABC"])
        with np.testing.assert_raises(ValueError):
            pauli_op(["XI", "XY", "XZ", "III"])
    else:
        with np.testing.assert_raises(ValueError):
            pauli_op([1], ["ABC"])
        with np.testing.assert_raises(ValueError):
            pauli_op([1, 1, 1, 1], ["XI", "XY", "XZ", "III"])
    with np.testing.assert_raises(ValueError):
        pauli_op([1, 1], ["X", "YZ"])
    with np.testing.assert_raises(ValueError):
        pauli_op([1, 2, 3], ["I", "X", "Y", "Z"])

    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]) @ pauli_op([1], ["III"])
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]) @ pauli_string("X")
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]) + pauli_op([1], ["X"])
    with np.testing.assert_raises(ValueError):
        p_op = pauli_op([-1, 1], ["IX", "YZ"])
        p_op += pauli_op([1], ["Z"])
    with np.testing.assert_raises(ValueError):
        p_op = pauli_op([-1, 1], ["IX", "YZ"])
        p_op -= pauli_op([1], ["ZZZ"])
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]) + pauli_string("XXXX")
    with np.testing.assert_raises(ValueError):
        p_op = pauli_op([-1, 1], ["IX", "YZ"])
        p_op += pauli_string("Z")
    with np.testing.assert_raises(ValueError):
        p_op = pauli_op([-1, 1], ["IX", "YZ"])
        p_op -= pauli_string("ZZZ")
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]) - pauli_op([1], ["X"])
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]) - pauli_string("XXXX")

    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]).extend(pauli_string("X"), 1j, dedupe=True)
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]).extend(pauli_string("XYZ"), 1j, dedupe=False)
    with np.testing.assert_raises(ValueError):
        pauli_op([-1, 1], ["IX", "YZ"]).extend(pauli_op([1j], ["XXX"]))

    with np.testing.assert_raises(ValueError):
        pauli_op([1, 1, 1], ["X", "Y", "Z"]).apply(np.ones(3))
    with np.testing.assert_raises(ValueError):
        pauli_op([1, 1, 1], ["X", "Y", "Z"]).apply(np.eye(4))

    with np.testing.assert_raises(ValueError):
        pauli_op([1, 1], ["XYZ", "ZYX"]).expectation_value(np.ones(32))
    with np.testing.assert_raises(ValueError):
        pauli_op([1, 1], ["XYZ", "ZYX"]).expectation_value(np.eye(16))


if __name__ == "__main__":
    pytest.main()
