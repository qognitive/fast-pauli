"""Test pauli objects from c++ against python implementations."""

import itertools as it
from typing import Callable

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
from fast_pauli.pypauli.helpers import naive_pauli_converter

# TODO: consider unifying unit tests (+ numerical validations) for equivalent structures
#       from pypauli and _fast_pauli submodules. Essentially, we would have unified
#       test cases parametrized by _fast_pauli.PauliString and pypauli.PauliString.
#       This would require identical interface for both things
#       (which is probably what are aiming for in a long term)


def test_pauli_string_wrapper(paulis: dict) -> None:
    """Test pauli string wrapper in python land."""
    for empty_ps in [fp.PauliString(), fp.PauliString(""), fp.PauliString([])]:
        assert empty_ps.weight == 0
        assert empty_ps.dims == 0
        assert empty_ps.n_qubits == 0
        assert str(empty_ps) == ""
        assert len(empty_ps.to_tensor()) == 0

    for s in it.chain(
        ["I", "X", "Y", "Z"],
        [[fp.Pauli("I")], [fp.Pauli("X")], [fp.Pauli("Y")], [fp.Pauli("Z")]],
    ):
        p = fp.PauliString(s)
        assert p.weight == 1 or str(p) == "I"
        assert p.dims == 2
        assert p.n_qubits == 1
        if isinstance(s, str):
            assert str(p) == s
            np.testing.assert_allclose(
                np.array(p.to_tensor()),
                paulis[s],
                atol=1e-15,
            )

    for s in map(lambda x: "".join(x), it.permutations("IXYZ", 3)):
        p = fp.PauliString(s)
        assert p.weight == len(s) - s.count("I")
        assert p.dims == 8
        assert p.n_qubits == 3
        assert str(p) == s
        np.testing.assert_allclose(
            np.array(p.to_tensor()),
            np.kron(paulis[s[0]], np.kron(paulis[s[1]], paulis[s[2]])),
            atol=1e-15,
        )


def test_sparse_pauli_composer(paulis: dict) -> None:
    """Test sparse pauli composer inside c++ implementation of to_tensor()."""
    ps = fp.PauliString("II")
    assert ps.weight == 0
    assert ps.dims == 4
    np.testing.assert_array_equal(np.array(ps.to_tensor()), np.eye(4))

    ps = fp.PauliString("IIII")
    assert ps.weight == 0
    assert ps.dims == 16
    np.testing.assert_array_equal(np.array(ps.to_tensor()), np.eye(16))

    ps = fp.PauliString("XXX")
    assert ps.weight == 3
    assert ps.dims == 8
    np.testing.assert_array_equal(np.array(ps.to_tensor()), np.fliplr(np.eye(8)))

    ps = fp.PauliString("IY")
    assert ps.weight == 1
    assert ps.dims == 4
    np.testing.assert_array_equal(
        np.array(ps.to_tensor()),
        np.block([[paulis["Y"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Y"]]]),
    )

    ps = fp.PauliString("IZ")
    assert ps.weight == 1
    assert ps.dims == 4
    np.testing.assert_array_equal(
        np.array(ps.to_tensor()),
        np.block([[paulis["Z"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Z"]]]),
    )

    for s in it.chain(
        map(lambda x: "".join(x), it.permutations("IXYZ", 2)),
        ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXXY"],
    ):
        ps = fp.PauliString(s)
        assert ps.weight == len(s) - s.count("I")
        assert ps.dims == 2 ** len(s)
        assert str(ps) == s
        np.testing.assert_allclose(
            np.array(ps.to_tensor()),
            naive_pauli_converter(s),
            atol=1e-15,
        )


def test_pauli_string_apply(generate_random_complex: Callable) -> None:
    """Test pauli string multiplication with 1d vector."""
    np.testing.assert_allclose(
        np.array(fp.PauliString("IXYZ").apply(np.zeros(16).tolist())),
        np.zeros(16),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        np.array(fp.PauliString("III").apply(np.arange(8).tolist())),
        np.arange(8),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        np.array(fp.PauliString("ZYX").apply(np.ones(8).tolist())),
        naive_pauli_converter("ZYX").sum(1),
        atol=1e-15,
    )

    for s in it.chain(
        list("IXYZ"),
        map(lambda x: "".join(x), it.permutations("IXYZ", 3)),
        ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"],
    ):
        n_dim = 2 ** len(s)
        psi = generate_random_complex(n_dim)
        np.testing.assert_allclose(
            np.array(fp.PauliString(s).apply(psi.tolist())),
            naive_pauli_converter(s).dot(psi),
            atol=1e-15,
        )


def test_pauli_string_apply_batch(generate_random_complex: Callable) -> None:
    """Test pauli string multiplication with 2d tensor."""
    np.testing.assert_allclose(
        np.array(fp.PauliString("IXYZ").apply_batch(np.zeros((16, 16)).tolist())),
        np.zeros((16, 16)),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        np.array(
            fp.PauliString("III").apply_batch(np.arange(8 * 8).reshape(8, 8).tolist())
        ),
        np.arange(8 * 8).reshape(8, 8),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        np.array(fp.PauliString("ZYX").apply_batch(np.eye(8).tolist())),
        naive_pauli_converter("ZYX"),
        atol=1e-15,
    )

    for s in it.chain(
        list("IXYZ"),
        map(lambda x: "".join(x), it.permutations("IXYZ", 3)),
        ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"],
    ):
        n_dim = 2 ** len(s)
        n_states = 42
        psis = generate_random_complex(n_dim, n_states)
        np.testing.assert_allclose(
            np.array(fp.PauliString(s).apply_batch(psis.tolist())),
            naive_pauli_converter(s) @ psis,
            atol=1e-15,
        )

    for s in map(lambda x: "".join(x), it.permutations("IXYZ", 2)):
        n_dim = 2 ** len(s)
        n_states = 7
        coeff = generate_random_complex(1)[0]
        psis = generate_random_complex(n_dim, n_states)

        np.testing.assert_allclose(
            np.array(fp.PauliString(s).apply_batch(psis.tolist(), coeff)),
            coeff * naive_pauli_converter(s) @ psis,
            atol=1e-15,
        )


def test_pauli_string_exceptions() -> None:
    """Test that exceptions from c++ are raised and propagated correctly."""
    with np.testing.assert_raises(TypeError):
        fp.PauliString([1, 2])
    with np.testing.assert_raises(ValueError):
        fp.PauliString("ABC")
    with np.testing.assert_raises(ValueError):
        fp.PauliString("xyz")

    with np.testing.assert_raises(AttributeError):
        fp.PauliString("XYZ").dims = 99
    with np.testing.assert_raises(AttributeError):
        fp.PauliString("XYZ").weight = 99

    with np.testing.assert_raises(ValueError):
        fp.PauliString("II").apply([0.1, 0.2, 0.3])
    with np.testing.assert_raises(TypeError):
        fp.PauliString("II").apply_batch([0.1, 0.2, 0.3, 0.4])
    with np.testing.assert_raises(ValueError):
        fp.PauliString("XYZ").apply_batch(np.eye(4).tolist())


if __name__ == "__main__":
    pytest.main([__file__])
