"""Test pauli c++ objects against python implementations."""

import itertools as it

import numpy as np
import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli.operations as pp

# TODO add a separate benchmark for get_sparse_repr vs compose_sparse_pauli
# TODO control numpy threading in a fixture for fair comparison


@pytest.fixture
def all_strings_for_qubits() -> list[str]:
    """Provide sample strings for testing."""

    def generate_paulis(qubits: int, limit: int = 1_000) -> list[str]:
        strings: list[str] = []
        for s in it.product("IXYZ", repeat=qubits):
            if limit and len(strings) > limit:
                break
            strings.append("".join(s))
        return strings

    return generate_paulis  # type: ignore


@pytest.fixture(scope="function")
def generate_random_complex(rng_seed: int = 321) -> np.ndarray:
    """Generate random complex numpy array with desired shape."""
    rng = np.random.default_rng(rng_seed)
    return lambda *shape: rng.random(shape) + 1j * rng.random(shape)


QUBITS_TO_BENCHMARK = [1, 2, 4, 10]


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
    "pauli_class,qubits,lang,bench_func",
    it.chain(
        [
            (fp.PauliString, q, "cpp", benchmark_dense_conversion_cpp)
            for q in QUBITS_TO_BENCHMARK
        ],
        [
            (pp.PauliString, q, "py", benchmark_dense_conversion_py)
            for q in QUBITS_TO_BENCHMARK
        ],
    ),
)
def test_dense_conversion_n_qubits(  # type: ignore[no-untyped-def]
    benchmark, all_strings_for_qubits, pauli_class, qubits, lang, bench_func
) -> None:
    """Benchmark dense conversion.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_strings_limit = 10 if qubits > 4 else None
    prepared_paulis = list(
        map(lambda s: pauli_class(s), all_strings_for_qubits(qubits, n_strings_limit))
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
    "pauli_class,qubits,lang,bench_func",
    it.chain(
        [(fp.PauliString, q, "cpp", benchmark_apply_cpp) for q in QUBITS_TO_BENCHMARK],
        [(pp.PauliString, q, "py", benchmark_apply_py) for q in QUBITS_TO_BENCHMARK],
    ),
)
def test_apply_n_qubits(  # type: ignore[no-untyped-def]
    benchmark,
    all_strings_for_qubits,
    generate_random_complex,
    pauli_class,
    qubits,
    lang,
    bench_func,
) -> None:
    """Benchmark PauliString multiplication with provided state vector.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_dims = 1 << qubits
    n_strings_limit = 10 if qubits > 4 else None

    prepared_paulis = list(
        map(lambda s: pauli_class(s), all_strings_for_qubits(qubits, n_strings_limit))
    )
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_paulis))
    ]

    benchmark(bench_func, paulis=prepared_paulis, states=prepared_states)


def benchmark_apply_batch_cpp(paulis: list, states: list) -> None:
    """Benchmark apply_batch method."""
    for p, psi in zip(paulis, states):
        result = p.apply_batch(psi.tolist())  # noqa: F841


def benchmark_apply_batch_py(paulis: list, states: list) -> None:
    """Benchmark apply_batch method."""
    for p, psi in zip(paulis, states):
        result = p.multiply(psi)  # noqa: F841


@pytest.mark.parametrize(
    "pauli_class,qubits,states,lang,bench_func",
    it.chain(
        [
            (fp.PauliString, q, n, "cpp", benchmark_apply_batch_cpp)
            for q in QUBITS_TO_BENCHMARK
            for n in [16, 128]
        ],
        [
            (pp.PauliString, q, n, "py", benchmark_apply_batch_py)
            for q in QUBITS_TO_BENCHMARK
            for n in [16, 128]
        ],
    ),
)
def test_apply_batch_n_qubits_n_states(  # type: ignore[no-untyped-def]
    benchmark,
    all_strings_for_qubits,
    generate_random_complex,
    pauli_class,
    qubits,
    states,
    lang,
    bench_func,
) -> None:
    """Benchmark PauliString multiplication with provided set of state vectors.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_dims = 1 << qubits
    n_strings_limit = 10 if qubits > 4 else None

    prepared_paulis = list(
        map(lambda s: pauli_class(s), all_strings_for_qubits(qubits, n_strings_limit))
    )
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_paulis))
    ]

    benchmark(bench_func, paulis=prepared_paulis, states=prepared_states)


if __name__ == "__main__":
    pytest.main([__file__])
