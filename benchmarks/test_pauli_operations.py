import pytest
import numpy as np
from itertools import permutations, chain

import pauli_operations
from pauli_operations import PauliString, SparsePauliString, PauliComposer


@pytest.fixture
def paulis():
    return pauli_operations.pauli_matrices()


def test_pauli_strings(paulis):
    for p in ["I", "X", "Y", "Z"]:
        assert PauliString(p).weight == 1 or p == "I"
        np.testing.assert_array_equal(PauliString(p).dense(), paulis[p])

    ps = PauliString("III")
    assert ps.weight == 0
    np.testing.assert_array_equal(ps.dense(), np.eye(8))

    ps = PauliString(string="IZ")
    assert ps.weight == 1
    np.testing.assert_array_equal(ps.dense(), np.kron(paulis["I"], paulis["Z"]))

    ps = PauliString(string="XYZ")
    assert ps.weight == 3
    np.testing.assert_array_equal(
        ps.dense(), np.kron(paulis["X"], np.kron(paulis["Y"], paulis["Z"]))
    )

    assert PauliString("XYIZXYZ").weight == 6
    assert PauliString("XXIYYIZZ").weight == 6
    assert PauliString("ZIXIZYXXY").weight == 7

    ps = SparsePauliString(np.arange(8), np.ones(8))
    np.testing.assert_array_equal(ps.dense(), np.eye(8))
    m = np.array([[0, 1, 0], [0, 0, 2], [3, 0, 0]])
    ps = SparsePauliString(columns=np.array([1, 2, 0]), values=np.array([1, 2, 3]))
    np.testing.assert_array_equal(ps.dense(), m)


def test_pauli_composer(paulis):
    for p in ["I", "X", "Y", "Z"]:
        pc = PauliComposer(PauliString(p))
        assert pc.n_qubits == 1
        assert pc.n_vals == 2
        assert pc.n_ys == 0 or p == "Y"
        assert pc.n_ys == 1 or p != "Y"
        np.testing.assert_array_equal(pc.sparse_pauli().dense(), paulis[p])

    pc = PauliComposer(PauliString("II"))
    assert pc.n_vals == 4
    np.testing.assert_array_equal(pc.sparse_pauli().dense(), np.eye(4))
    np.testing.assert_array_equal(pc.sparse_diag_pauli().dense(), np.eye(4))

    pc = PauliComposer(PauliString("IIII"))
    assert pc.n_vals == 16
    np.testing.assert_array_equal(pc.sparse_pauli().dense(), np.eye(16))
    np.testing.assert_array_equal(pc.sparse_diag_pauli().dense(), np.eye(16))

    pc = PauliComposer(PauliString("II"))
    assert pc.n_vals == 4
    np.testing.assert_array_equal(pc.sparse_pauli().dense(), np.eye(4))

    pc = PauliComposer(PauliString("XXX"))
    assert pc.n_vals == 8
    np.testing.assert_array_equal(pc.sparse_pauli().dense(), np.fliplr(np.eye(8)))

    pc = PauliComposer(PauliString("IY"))
    np.testing.assert_array_equal(
        pc.sparse_pauli().dense(),
        np.block([[paulis["Y"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Y"]]]),
    )

    pc = PauliComposer(PauliString("IZ"))
    np.testing.assert_array_equal(
        pc.sparse_pauli().dense(),
        np.block([[paulis["Z"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Z"]]]),
    )


def test_pauli_composer_equivalence():
    for c in ["I", "X", "Y", "Z"]:
        np.testing.assert_array_equal(
            PauliComposer(PauliString(c)).sparse_pauli().dense(),
            PauliString(c).dense(),
        )

    for s in permutations("XYZ", 2):
        s = "".join(s)
        np.testing.assert_array_equal(
            PauliComposer(PauliString(s)).sparse_pauli().dense(),
            PauliString(s).dense(),
        )

    for s in permutations("IXYZ", 3):
        s = "".join(s)
        np.testing.assert_array_equal(
            PauliComposer(PauliString(s)).sparse_pauli().dense(),
            PauliString(s).dense(),
        )

    ixyz = PauliComposer(PauliString("IXYZ")).sparse_pauli().dense()
    np.testing.assert_array_equal(ixyz, PauliString("IXYZ").dense())

    zyxi = PauliComposer(PauliString("ZYXI")).sparse_pauli().dense()
    np.testing.assert_array_equal(zyxi, PauliString("ZYXI").dense())

    assert np.abs(ixyz - zyxi).sum().sum() > 1e-10

    for s in ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"]:
        np.testing.assert_array_equal(
            PauliComposer(PauliString(s)).sparse_pauli().dense(), PauliString(s).dense()
        )


def test_sparse_pauli_multiply():
    rng = np.random.default_rng(321)

    for s in chain(
        list("IXYZ"), list(permutations("IXYZ", 3)), ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"]
    ):
        s = "".join(s)
        n = 2 ** len(s)
        psi = rng.random(n)
        psi_batch = rng.random((n, 25))

        np.testing.assert_allclose(
            PauliComposer(PauliString(s)).sparse_pauli().multiply(psi),
            PauliString(s).dense().dot(psi),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            PauliComposer(PauliString(s)).sparse_pauli().multiply(psi_batch),
            PauliString(s).dense() @ psi_batch,
            atol=1e-15,
        )


def test_pauli_composer_multiply():
    rng = np.random.default_rng(321)

    for s in chain(
        list("IXYZ"), list(permutations("IXYZ", 3)), ["XYIZXYZ", "XXIYYIZZ", "ZIXIZYXX"]
    ):
        s = "".join(s)
        n = 2 ** len(s)
        psi = rng.random(n)
        psi_batch = rng.random((n, 20))

        np.testing.assert_allclose(
            PauliComposer(PauliString(s))
            .efficient_sparse_multiply(psi.reshape(-1, 1))
            .ravel(),
            PauliString(s).dense().dot(psi),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            PauliComposer(PauliString(s)).efficient_sparse_multiply(psi_batch),
            PauliString(s).dense() @ psi_batch,
            atol=1e-15,
        )


if __name__ == "__main__":
    pytest.main([__file__])
