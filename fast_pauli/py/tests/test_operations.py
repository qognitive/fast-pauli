import pytest
import numpy as np
from itertools import permutations, chain

from pypauli.operations import (
    PauliString,
    compose_sparse_pauli,
    compose_sparse_diag_pauli,
)
from pypauli.helpers import pauli_matrices, naive_pauli_converter


@pytest.fixture
def paulis():
    return pauli_matrices()


def test_pauli_string(paulis):
    for p in ["I", "X", "Y", "Z"]:
        ps = PauliString(p)
        assert ps.dim == 2
        assert ps.weight == 1 or p == "I"
        np.testing.assert_array_equal(ps.dense(), paulis[p])
        np.testing.assert_array_equal(naive_pauli_converter(p), paulis[p])

    ps = PauliString("III")
    assert ps.dim == 8
    assert ps.weight == 0
    np.testing.assert_array_equal(ps.dense(), np.eye(8))
    np.testing.assert_array_equal(naive_pauli_converter(ps.string), np.eye(8))

    ps = PauliString(string="IZ")
    assert ps.dim == 4
    assert ps.weight == 1
    np.testing.assert_array_equal(ps.dense(), np.kron(paulis["I"], paulis["Z"]))
    np.testing.assert_array_equal(
        naive_pauli_converter(ps.string), np.kron(paulis["I"], paulis["Z"])
    )

    ps = PauliString(string="XYZ")
    assert ps.dim == 8
    assert ps.weight == 3
    np.testing.assert_array_equal(
        ps.dense(), np.kron(paulis["X"], np.kron(paulis["Y"], paulis["Z"]))
    )
    np.testing.assert_array_equal(
        naive_pauli_converter(ps.string),
        np.kron(paulis["X"], np.kron(paulis["Y"], paulis["Z"])),
    )

    assert PauliString("XYIZXYZ").dim == 2**7
    assert PauliString("XYIZXYZ").weight == 6
    assert PauliString("XXIYYIZZ").dim == 2**8
    assert PauliString("XXIYYIZZ").weight == 6
    assert PauliString("ZIXIZYXXY").dim == 2**9
    assert PauliString("ZIXIZYXXY").weight == 7


def test_sparse_pauli_composer(paulis):
    ps = PauliString("II")
    assert ps.dim == 4
    np.testing.assert_array_equal(ps.dense(), np.eye(4))
    np.testing.assert_array_equal(compose_sparse_diag_pauli(ps.string), np.ones(4))

    ps = PauliString("IIII")
    assert ps.dim == 16
    np.testing.assert_array_equal(ps.dense(), np.eye(16))
    np.testing.assert_array_equal(compose_sparse_diag_pauli(ps.string), np.ones(16))

    ps = PauliString("XXX")
    assert ps.dim == 8
    np.testing.assert_array_equal(ps.dense(), np.fliplr(np.eye(8)))

    ps = PauliString("IY")
    np.testing.assert_array_equal(
        ps.dense(),
        np.block([[paulis["Y"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Y"]]]),
    )

    ps = PauliString("IZ")
    np.testing.assert_array_equal(
        ps.dense(),
        np.block([[paulis["Z"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Z"]]]),
    )

    np.testing.assert_array_equal(
        compose_sparse_diag_pauli("IZI"),
        compose_sparse_pauli("IZI")[1],
    )

    np.testing.assert_array_equal(
        compose_sparse_diag_pauli("ZIZI"),
        compose_sparse_pauli("ZIZI")[1],
    )

    np.testing.assert_array_equal(
        compose_sparse_diag_pauli("ZZZIII"),
        compose_sparse_pauli("ZZZIII")[1],
    )


def test_sparse_pauli_composer_equivalence():
    for c in ["I", "X", "Y", "Z"]:
        np.testing.assert_array_equal(
            PauliString(c).dense(),
            naive_pauli_converter(c),
        )

    for s in permutations("XYZ", 2):
        s = "".join(s)
        np.testing.assert_array_equal(
            PauliString(s).dense(),
            naive_pauli_converter(s),
        )

    for s in permutations("IXYZ", 3):
        s = "".join(s)
        np.testing.assert_array_equal(
            PauliString(s).dense(),
            naive_pauli_converter(s),
        )

    ixyz = PauliString("IXYZ").dense()
    np.testing.assert_array_equal(ixyz, naive_pauli_converter("IXYZ"))

    zyxi = PauliString("ZYXI").dense()
    np.testing.assert_array_equal(zyxi, naive_pauli_converter("ZYXI"))

    assert np.abs(ixyz - zyxi).sum().sum() > 1e-10

    for s in ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"]:
        np.testing.assert_array_equal(PauliString(s).dense(), naive_pauli_converter(s))


def test_sparse_pauli_multiply():
    rng = np.random.default_rng(321)

    for s in chain(
        list("IXYZ"), list(permutations("IXYZ", 3)), ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"]
    ):
        s = "".join(s)
        n = 2 ** len(s)
        psi = rng.random(n)
        psi_batch = rng.random((n, 21))

        np.testing.assert_allclose(
            PauliString(s).multiply(psi),
            naive_pauli_converter(s).dot(psi),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            PauliString(s).multiply(psi_batch),
            naive_pauli_converter(s) @ psi_batch,
            atol=1e-15,
        )


if __name__ == "__main__":
    pytest.main([__file__])
