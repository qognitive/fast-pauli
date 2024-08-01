"""Test pauli c++ objects against python implementations."""

import itertools as it

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli.operations as pp

# TODO add a separate benchmark for get_sparse_repr vs compose_sparse_pauli


@pytest.fixture
def all_strings_for_qubits() -> list[str]:
    """Provide sample strings for testing."""
    return lambda q: list(map("".join, it.product("IXYZ", repeat=q)))  # type: ignore


# following two helper functions are going to be removed once we align interfaces:
def benchmark_dense_conversion_cpp(paulis: list) -> None:
    """Benchmark dense conversion."""
    for p in paulis:
        dense_repr = p.to_tensor()  # noqa: F841


def benchmark_dense_conversion_py(paulis: list) -> None:
    """Benchmark dense conversion."""
    for p in paulis:
        dense_repr = p.dense()  # noqa: F841


@pytest.mark.parametrize(
    "lang,qubits,pauli_class,bench_func",
    it.chain(
        [("cpp", q, fp.PauliString, benchmark_dense_conversion_cpp) for q in [1, 2, 4]],
        [("py", q, pp.PauliString, benchmark_dense_conversion_py) for q in [1, 2, 4]],
    ),
)
def test_dense_conversion_n_qubits(  # type: ignore[no-untyped-def]
    benchmark, all_strings_for_qubits, lang, qubits, pauli_class, bench_func
) -> None:
    """Benchmark dense conversion.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    prepared_paulis = list(
        map(lambda s: pauli_class(s), all_strings_for_qubits(qubits))
    )
    benchmark(bench_func, paulis=prepared_paulis)


def benchmark_apply_cpp(paulis: list, states: list) -> None:
    """Benchmark apply method."""
    for p, psi in zip(paulis, states):
        result = p.apply(psi)  # noqa: F841


def benchmark_apply_py(paulis: list, states: list) -> None:
    """Benchmark apply method."""
    for p, psi in zip(paulis, states):
        result = p.multiply(psi)  # noqa: F841


@pytest.mark.parametrize(
    "lang,qubits,pauli_class,bench_func",
    it.chain(
        [("cpp", q, fp.PauliString, benchmark_apply_cpp) for q in [1, 2, 4]],
        [("py", q, pp.PauliString, benchmark_apply_py) for q in [1, 2, 4]],
    ),
)
def test_apply_n_qubits(  # type: ignore[no-untyped-def]
    benchmark, all_strings_for_qubits, lang, qubits, pauli_class, bench_func
) -> None:
    """Benchmark PauliString multiplication with provided state vector.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    rng = np.random.default_rng(321)
    dim = 1 << qubits

    prepared_paulis = list(
        map(lambda s: pauli_class(s), all_strings_for_qubits(qubits))
    )
    prepared_states = [rng.random(dim) for _ in range(len(prepared_paulis))]

    benchmark(bench_func, paulis=prepared_paulis, states=prepared_states)


def benchmark_apply_batch_cpp(paulis: list, states: list) -> None:
    """Benchmark apply_batch method."""
    for p, psi in zip(paulis, states):
        result = p.apply_batch(psi)  # noqa: F841


def benchmark_apply_batch_py(paulis: list, states: list) -> None:
    """Benchmark apply_batch method."""
    for p, psi in zip(paulis, states):
        result = p.multiply(psi)  # noqa: F841


@pytest.mark.parametrize(
    "lang,qubits,states,pauli_class,bench_func",
    it.chain(
        [
            ("cpp", q, n, fp.PauliString, benchmark_apply_batch_cpp)
            for q in [1, 2, 4]
            for n in [16, 128]
        ],
        [
            ("py", q, n, pp.PauliString, benchmark_apply_batch_py)
            for q in [1, 2, 4]
            for n in [16, 128]
        ],
    ),
)
def test_apply_batch_n_qubits(  # type: ignore[no-untyped-def]
    benchmark, all_strings_for_qubits, lang, qubits, states, pauli_class, bench_func
) -> None:
    """Benchmark PauliString multiplication with provided set of state vectors.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    rng = np.random.default_rng(321)
    dim = 1 << qubits

    prepared_paulis = list(
        map(lambda s: pauli_class(s), all_strings_for_qubits(qubits))
    )
    prepared_states = [rng.random((dim, states)) for _ in range(len(prepared_paulis))]

    benchmark(bench_func, paulis=prepared_paulis, states=prepared_states)


if __name__ == "__main__":
    pytest.main()
