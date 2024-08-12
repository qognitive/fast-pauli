"""Test pauli objects from c++ against python implementations."""

import itertools as it
from typing import Callable

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp
from fast_pauli.pypauli.helpers import naive_pauli_converter
from tests.conftest import resolve_parameter_repr


def test_naive_pauli_converter(paulis: dict) -> None:
    """Test naive Pauli converter for dense matrix."""
    for p in ["I", "X", "Y", "Z"]:
        np.testing.assert_equal(naive_pauli_converter(p), paulis[p])

    np.testing.assert_equal(naive_pauli_converter("III"), np.eye(8))
    np.testing.assert_equal(naive_pauli_converter("XXX"), np.fliplr(np.eye(8)))

    np.testing.assert_equal(
        naive_pauli_converter("IZ"), np.kron(paulis["I"], paulis["Z"])
    )
    np.testing.assert_equal(
        naive_pauli_converter("IZ"),
        np.block([[paulis["Z"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Z"]]]),
    )
    np.testing.assert_equal(
        naive_pauli_converter("IY"), np.kron(paulis["I"], paulis["Y"])
    )
    np.testing.assert_equal(
        naive_pauli_converter("IY"),
        np.block([[paulis["Y"], np.zeros((2, 2))], [np.zeros((2, 2)), paulis["Y"]]]),
    )
    np.testing.assert_equal(
        naive_pauli_converter("YX"), np.kron(paulis["Y"], paulis["X"])
    )
    np.testing.assert_equal(
        naive_pauli_converter("XYZ"),
        np.kron(paulis["X"], np.kron(paulis["Y"], paulis["Z"])),
    )


@pytest.mark.parametrize(
    "PauliString,", [fp.PauliString, pp.PauliString], ids=resolve_parameter_repr
)
def test_string_trivial(PauliString: type) -> None:  # noqa: N803
    """Test pauli string class on trivial inputs."""
    empty_paulis = (
        [PauliString(), PauliString([]), PauliString("")]
        if isinstance(PauliString, fp.PauliString)
        else [PauliString("")]
    )
    for empty_ps in empty_paulis:
        assert empty_ps.weight == 0
        assert empty_ps.dim == 0
        assert empty_ps.n_qubits == 0
        assert str(empty_ps) == ""
        assert len(empty_ps.to_tensor()) == 0

    ps = PauliString("III")
    assert ps.dim == 8
    assert ps.weight == 0
    np.testing.assert_equal(ps.to_tensor(), np.eye(8))


@pytest.mark.parametrize(
    "PauliString,", [fp.PauliString, pp.PauliString], ids=resolve_parameter_repr
)
def test_string_basics(
    paulis: dict,
    pauli_strings_with_size: Callable,
    PauliString: type,  # noqa: N803
) -> None:
    """Test pauli string class in python land."""
    for s in it.chain(
        ["I", "X", "Y", "Z"],
        [[fp.Pauli("I")], [fp.Pauli("X")], [fp.Pauli("Y")], [fp.Pauli("Z")]]
        if isinstance(PauliString, fp.PauliString)
        else [],
    ):
        p = PauliString(s)
        assert p.weight == 1 or str(p) == "I"
        assert p.dim == 2
        assert p.n_qubits == 1
        if isinstance(s, str):
            assert str(p) == s
            np.testing.assert_allclose(
                p.to_tensor(),
                paulis[s],
                atol=1e-15,
            )

    for s in pauli_strings_with_size(3):
        p = PauliString(s)
        assert p.weight == len(s) - s.count("I")
        assert p.dim == 8
        assert p.n_qubits == 3
        assert str(p) == s
        np.testing.assert_allclose(
            p.to_tensor(),
            np.kron(paulis[s[0]], np.kron(paulis[s[1]], paulis[s[2]])),
            atol=1e-15,
        )

    assert PauliString("XYIZXYZ").dim == 2**7
    assert PauliString("XYIZXYZ").weight == 6
    assert PauliString("XXIYYIZZ").dim == 2**8
    assert PauliString("XXIYYIZZ").weight == 6
    assert PauliString("ZIXIZYXXY").dim == 2**9
    assert PauliString("ZIXIZYXXY").weight == 7


@pytest.mark.consistency
@pytest.mark.parametrize(
    "PauliString,", [fp.PauliString, pp.PauliString], ids=resolve_parameter_repr
)
def test_dense_representation(
    paulis: dict,
    sample_pauli_strings: list,
    PauliString: type,  # noqa: N803
) -> None:
    """Test dense representation of Pauli strings against naive approach."""
    ps = PauliString("II")
    assert ps.weight == 0
    assert ps.dim == 4
    np.testing.assert_equal(ps.to_tensor(), np.eye(4))

    ps = PauliString("IIII")
    assert ps.weight == 0
    assert ps.dim == 16
    np.testing.assert_equal(ps.to_tensor(), np.eye(16))

    ixyz = np.array(PauliString("IXYZ").to_tensor())
    np.testing.assert_allclose(ixyz, naive_pauli_converter("IXYZ"), atol=1e-15)

    zyxi = np.array(PauliString("ZYXI").to_tensor())
    np.testing.assert_allclose(zyxi, naive_pauli_converter("ZYXI"), atol=1e-15)

    assert np.abs(ixyz - zyxi).sum().sum() >= 1

    for s in sample_pauli_strings:
        ps = PauliString(s)
        assert ps.weight == len(s) - s.count("I")
        assert ps.dim == 2 ** len(s)
        assert str(ps) == s
        np.testing.assert_allclose(
            ps.to_tensor(),
            naive_pauli_converter(s),
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "PauliString,", [fp.PauliString, pp.PauliString], ids=resolve_parameter_repr
)
def test_apply_1d(
    sample_pauli_strings: list,
    generate_random_complex: Callable,
    PauliString: type,  # noqa: N803
) -> None:
    """Test pauli string multiplication with 1d vector."""
    np.testing.assert_allclose(
        PauliString("IXYZ").apply(np.zeros(16)),
        np.zeros(16),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        PauliString("III").apply(np.arange(8)),
        np.arange(8),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        PauliString("ZYX").apply(np.ones(8)),
        naive_pauli_converter("ZYX").sum(1),
        atol=1e-15,
    )

    for s in sample_pauli_strings:
        n_dim = 2 ** len(s)
        psi = generate_random_complex(n_dim)
        np.testing.assert_allclose(
            PauliString(s).apply(psi),
            naive_pauli_converter(s).dot(psi),
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "PauliString,", [fp.PauliString, pp.PauliString], ids=resolve_parameter_repr
)
def test_apply_batch(
    sample_pauli_strings: list,
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_string: type[fp.PauliString]|type[pp.PauliString],  
) -> None:
    """Test pauli string multiplication with 2d tensor."""
    np.testing.assert_allclose(
        PauliString("IXYZ").apply(np.zeros((16, 16))),
        np.zeros((16, 16)),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        PauliString("III").apply(np.arange(8 * 8).reshape(8, 8)),
        np.arange(8 * 8).reshape(8, 8),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        PauliString("ZYX").apply(np.eye(8)),
        naive_pauli_converter("ZYX"),
        atol=1e-15,
    )

    for s in sample_pauli_strings:
        n_dim = 2 ** len(s)
        n_states = 42
        psis = generate_random_complex(n_dim, n_states)
        np.testing.assert_allclose(
            PauliString(s).apply(psis),
            naive_pauli_converter(s) @ psis,
            atol=1e-15,
        )

    for s in pauli_strings_with_size(4):
        n_dim = 2 ** len(s)
        n_states = 7
        coeff = generate_random_complex(1)[0]
        psis = generate_random_complex(n_dim, n_states)

        np.testing.assert_allclose(
            PauliString(s).apply(psis, coeff),
            coeff * naive_pauli_converter(s) @ psis,
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "PauliString,", [(fp.PauliString), (pp.PauliString)], ids=resolve_parameter_repr
)
def test_expected_value(
    sample_pauli_strings: list,
    generate_random_complex: Callable,
    PauliString: type,  # noqa: N803
) -> None:
    """Test the expected value method."""
    np.testing.assert_allclose(
        PauliString("IXYZ").expected_value(np.zeros(16)),
        0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        PauliString("IXYZ").expected_value(np.zeros((16, 16))),
        np.zeros(16),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        PauliString("III").expected_value(np.arange(8)),
        np.square(np.arange(8)).sum(),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        PauliString("III").expected_value(np.arange(8 * 8).reshape(8, 8)),
        np.square(np.arange(8 * 8)).reshape(8, 8).sum(0),
        atol=1e-15,
    )

    for s in sample_pauli_strings:
        n_dim = 2 ** len(s)
        n_states = 21

        psi = generate_random_complex(n_dim)
        np.testing.assert_allclose(
            PauliString(s).expected_value(psi),
            naive_pauli_converter(s).dot(psi).dot(psi.conj()),
            atol=1e-15,
        )

        psis = generate_random_complex(n_states, n_dim)
        # compute <psi_t|P_i|psi_t>
        expected = np.einsum("ti,ij,tj->t", psis.conj(), naive_pauli_converter(s), psis)
        np.testing.assert_allclose(
            PauliString(s).expected_value(psis.T),
            expected,
            atol=1e-15,
        )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "PauliString,", [(fp.PauliString), (pp.PauliString)], ids=resolve_parameter_repr
)
def test_exceptions(PauliString: type) -> None:  # noqa: N803
    """Test that exceptions are raised and propagated correctly."""
    with np.testing.assert_raises(ValueError):
        PauliString("ABC")
    with np.testing.assert_raises(ValueError):
        PauliString("xyz")

    if isinstance(PauliString, fp.PauliString):
        with np.testing.assert_raises(TypeError):
            PauliString([1, 2])
        with np.testing.assert_raises(AttributeError):
            PauliString("XYZ").dim = 99
        with np.testing.assert_raises(AttributeError):
            PauliString("XYZ").weight = 99

    with np.testing.assert_raises(ValueError):
        PauliString("II").apply(np.array([0.1, 0.2, 0.3]))
    with np.testing.assert_raises(ValueError):
        PauliString("XYZ").apply(np.eye(4))


@pytest.mark.consistency
def test_sparse_composers(paulis: dict, pauli_strings_with_size: Callable) -> None:
    """Test consistency for sparse pauli composers."""
    np.testing.assert_equal(
        pp.pauli_string.compose_sparse_diag_pauli("IZI"),
        pp.pauli_string.compose_sparse_pauli("IZI")[1],
    )

    np.testing.assert_equal(
        pp.pauli_string.compose_sparse_diag_pauli("ZIZI"),
        pp.pauli_string.compose_sparse_pauli("ZIZI")[1],
    )

    np.testing.assert_equal(
        pp.pauli_string.compose_sparse_diag_pauli("ZZZIII"),
        pp.pauli_string.compose_sparse_pauli("ZZZIII")[1],
    )

    for s in ["I", "X", "Y", "Z"] + pauli_strings_with_size(3):
        py_cols, py_vals = pp.pauli_string.compose_sparse_pauli(s)
        cpp_cols, cpp_vals = fp.helpers.pauli_string_sparse_repr(
            [fp.Pauli(c) for c in s]
        )
        np.testing.assert_allclose(py_cols, cpp_cols, atol=1e-15)
        np.testing.assert_allclose(py_vals, cpp_vals, atol=1e-15)


if __name__ == "__main__":
    pytest.main()
