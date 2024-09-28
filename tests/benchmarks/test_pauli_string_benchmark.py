"""Test pauli c++ objects against python implementations."""

import itertools as it
from typing import Callable

import numpy as np
import pytest

import fast_pauli as fp
import fast_pauli.pypauli as pp
from tests.conftest import (
    QUBITS_TO_BENCHMARK,
    SAMPLE_STRINGS_LIMIT,
    resolve_parameter_repr,
)

N_STATES_TO_BENCHMARK = [16, 128, 1024]


@pytest.fixture
def prepared_paulis(
    pauli_strings_with_size: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> list:
    """Fixture to provide initialized Pauli strings for testing."""
    return list(
        map(
            lambda s: pauli_class(s),
            pauli_strings_with_size(qubits, limit=SAMPLE_STRINGS_LIMIT),
        )
    )


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
    prepared_paulis: list,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark dense conversion.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
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
    prepared_paulis: list,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark PauliString multiplication with provided state vector.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_paulis))
    ]

    benchmark(benchmark_apply, paulis=prepared_paulis, states=prepared_states)


@pytest.mark.parametrize(
    "qubits,states,pauli_class,",
    it.chain(
        [
            (q, n, fp.PauliString)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
        [
            (q, n, pp.PauliString)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
    ),
    ids=resolve_parameter_repr,
)
def test_apply_batch_n_qubits_n_states(
    benchmark: Callable,
    prepared_paulis: list,
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
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_paulis))
    ]

    benchmark(benchmark_apply, paulis=prepared_paulis, states=prepared_states)


def benchmark_expectation_value(paulis: list, states: list) -> None:
    """Benchmark expectation_value method."""
    for p, psi in zip(paulis, states):
        result = p.expectation_value(psi)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,",
    it.chain(
        [(q, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_expectation_value_n_qubits(
    benchmark: Callable,
    prepared_paulis: list,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark PauliString expectation_value with provided state vector."""
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_paulis))
    ]

    benchmark(
        benchmark_expectation_value, paulis=prepared_paulis, states=prepared_states
    )


@pytest.mark.parametrize(
    "qubits,states,pauli_class,",
    it.chain(
        [
            (q, n, fp.PauliString)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
        [
            (q, n, pp.PauliString)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
    ),
    ids=resolve_parameter_repr,
)
def test_expectation_value_batch_n_qubits_n_states(
    benchmark: Callable,
    prepared_paulis: list,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
    states: int,
) -> None:
    """Benchmark PauliString expectation_value with provided set of state vectors."""
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_paulis))
    ]

    benchmark(
        benchmark_expectation_value, paulis=prepared_paulis, states=prepared_states
    )


def benchmark_matmul(left_paulis: list, right_paulis: list) -> None:
    """Benchmark PauliString multiplication."""
    for lp, rp in zip(left_paulis, right_paulis):
        phase, p_str = lp @ rp  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,",
    it.chain(
        [(q, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_multiplication_n_qubits(
    benchmark: Callable,
    prepared_paulis: list,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark matrix multiplication of PauliStrings.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    left_paulis, right_paulis = np.array_split(prepared_paulis, 2)

    benchmark(benchmark_matmul, left_paulis=left_paulis, right_paulis=right_paulis)


def benchmark_arithmetic(left_paulis: list, right_paulis: list) -> None:
    """Benchmark PauliString arithmetic."""
    for lp, rp in zip(left_paulis, right_paulis):
        p_op1 = lp + rp  # noqa: F841
        p_op2 = rp - lp  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,",
    it.chain(
        [(q, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_arithmetic_n_qubits(
    benchmark: Callable,
    prepared_paulis: list,
    pauli_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark addition and subtraction of PauliStrings.

    Parametrized test case to run the benchmark across
    all Pauli strings of given length for given PauliString class.
    """
    left_paulis, right_paulis = np.array_split(prepared_paulis, 2)

    benchmark(benchmark_arithmetic, left_paulis=left_paulis, right_paulis=right_paulis)


def benchmark_sparse_composer(paulis: list, composer: Callable) -> None:
    """Benchmark algorithm for sparse representation of pauli string."""
    for ps in paulis:
        cols, vals = composer(ps)  # noqa: F841


@pytest.mark.skip(
    reason="currently std::vector conversion on c++ side "
    "takes majority of time -> results are not representative"
)
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
    prepared_paulis: list = pauli_strings_with_size(qubits, limit=SAMPLE_STRINGS_LIMIT)
    if "pypauli" not in composer_func.__module__:  # check if it's c++ wrapper
        prepared_paulis = [[fp.Pauli(c) for c in s] for s in prepared_paulis]

    benchmark(benchmark_sparse_composer, paulis=prepared_paulis, composer=composer_func)


if __name__ == "__main__":
    pytest.main()
