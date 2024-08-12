"""Test pauli c++ objects against python implementations."""

import itertools as it
from typing import Callable

import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp
from tests.conftest import resolve_parameter_repr

QUBITS_TO_BENCHMARK = [1, 2, 4, 10]


def benchmark_sparse_composer(paulis: list, composer: Callable) -> None:
    """Benchmark algorithm for sparse representation of pauli string."""
    for ps in paulis:
        cols, vals = composer(ps)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,composer_func,",
    it.chain(
        [(q, fp.helpers.pauli_string_sparse_repr) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.pauli_string.compose_sparse_pauli) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_string_sparse_composer_n_qubits(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    composer_func: Callable,
    qubits: int,
) -> None:
    """Benchmark algorithm for sparse representation of pauli string.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_strings_limit = 10 if qubits > 4 else None
    prepared_paulis = pauli_strings_with_size(qubits, n_strings_limit)

    if "pypauli" not in composer_func.__module__:  # check if it's c++ wrapper
        prepared_paulis = list(
            map(lambda pstr: [fp.Pauli(c) for c in pstr], prepared_paulis)
        )

    benchmark(benchmark_sparse_composer, paulis=prepared_paulis, composer=composer_func)


def benchmark_dense_conversion(paulis: list) -> None:
    """Benchmark dense conversion."""
    for p in paulis:
        dense_repr = p.to_tensor()  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,",
    it.chain(
        [(q, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_dense_conversion_n_qubits(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark dense conversion.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_strings_limit = 10 if qubits > 4 else None
    prepared_paulis = list(
        map(lambda s: pauli_class(s), pauli_strings_with_size(qubits, n_strings_limit))
    )
    benchmark(benchmark_dense_conversion, paulis=prepared_paulis)


def benchmark_apply(paulis: list, states: list) -> None:
    """Benchmark apply method."""
    for p, psi in zip(paulis, states):
        result = p.apply(psi)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,",
    it.chain(
        [(q, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_apply_n_qubits(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark PauliString multiplication with provided state vector.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_dims = 1 << qubits
    n_strings_limit = 10 if qubits > 4 else None

    prepared_paulis = list(
        map(lambda s: pauli_class(s), pauli_strings_with_size(qubits, n_strings_limit))
    )
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_paulis))
    ]

    benchmark(benchmark_apply, paulis=prepared_paulis, states=prepared_states)


@pytest.mark.parametrize(
    "qubits,states,pauli_class,",
    it.chain(
        [(q, n, fp.PauliString) for q in QUBITS_TO_BENCHMARK for n in [16, 128]],
        [(q, n, pp.PauliString) for q in QUBITS_TO_BENCHMARK for n in [16, 128]],
    ),
    ids=resolve_parameter_repr,
)
def test_apply_batch_n_qubits_n_states(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
    states: int,
) -> None:
    """Benchmark PauliString multiplication with provided set of state vectors.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_dims = 1 << qubits
    n_strings_limit = 10 if qubits > 4 else None

    prepared_paulis = list(
        map(lambda s: pauli_class(s), pauli_strings_with_size(qubits, n_strings_limit))
    )
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_paulis))
    ]

    benchmark(benchmark_apply, paulis=prepared_paulis, states=prepared_states)


def benchmark_expected_value(paulis: list, states: list) -> None:
    """Benchmark expected_value method."""
    for p, psi in zip(paulis, states):
        result = p.expected_value(psi)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,",
    it.chain(
        [(q, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_expected_value_n_qubits(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark PauliString expected_value with provided state vector."""
    n_dims = 1 << qubits
    n_strings_limit = 10 if qubits > 4 else None

    prepared_paulis = list(
        map(lambda s: pauli_class(s), pauli_strings_with_size(qubits, n_strings_limit))
    )
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_paulis))
    ]

    benchmark(benchmark_expected_value, paulis=prepared_paulis, states=prepared_states)


@pytest.mark.parametrize(
    "qubits,states,pauli_class,",
    it.chain(
        [(q, n, fp.PauliString) for q in QUBITS_TO_BENCHMARK for n in [16, 128]],
        [(q, n, pp.PauliString) for q in QUBITS_TO_BENCHMARK for n in [16, 128]],
    ),
    ids=resolve_parameter_repr,
)
def test_expected_value_batch_n_qubits_n_states(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
    states: int,
) -> None:
    """Benchmark PauliString expected_value with provided set of state vectors."""
    n_dims = 1 << qubits
    n_strings_limit = 10 if qubits > 4 else None

    prepared_paulis = list(
        map(lambda s: pauli_class(s), pauli_strings_with_size(qubits, n_strings_limit))
    )
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_paulis))
    ]

    benchmark(benchmark_expected_value, paulis=prepared_paulis, states=prepared_states)


if __name__ == "__main__":
    pytest.main()
