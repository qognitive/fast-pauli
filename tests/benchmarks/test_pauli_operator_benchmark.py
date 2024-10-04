#############################################################################
# This code is part of Fast Pauli.
#
# (C) Copyright Qognitive Inc 2024.
#
# This code is licensed under the BSD 2-Clause License. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


#############################################################################

"""Test pauli c++ objects against python implementations."""

import itertools as it
from typing import Callable

import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp
from tests.conftest import (
    QUBITS_TO_BENCHMARK,
    SAMPLE_STRINGS_LIMIT,
    resolve_parameter_repr,
)

N_OPERATORS_TO_BENCHMARK = 64
N_STATES_TO_BENCHMARK = [16, 64, 256]


@pytest.fixture
def prepared_operators(
    pauli_strings_with_size: Callable,
    pauli_strings_shuffled: Callable,
    generate_random_complex: Callable,
    pauli_op: type[fp.PauliOp] | type[pp.PauliOp],
    qubits: int,
) -> list:
    """Fixture to provide random initialized Pauli operators for testing."""
    operators = []
    for _ in range(N_OPERATORS_TO_BENCHMARK // 2):
        paulis = pauli_strings_shuffled(qubits, limit=SAMPLE_STRINGS_LIMIT)
        coeffs = generate_random_complex(len(paulis))
        operators.append(pauli_op(coeffs, paulis))
    for _ in range(N_OPERATORS_TO_BENCHMARK // 2):
        paulis = pauli_strings_with_size(qubits, limit=SAMPLE_STRINGS_LIMIT)
        coeffs = generate_random_complex(len(paulis))
        operators.append(pauli_op(coeffs, paulis))
    return operators


def benchmark_dense_conversion(operators: list) -> None:
    """Benchmark dense conversion."""
    for op in operators:
        dense_repr = op.to_tensor()  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_op,",
    it.chain(
        [(q, fp.PauliOp) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_dense_conversion_n_qubits(
    benchmark: Callable,
    prepared_operators: list,
) -> None:
    """Benchmark dense conversion."""
    benchmark(benchmark_dense_conversion, operators=prepared_operators)


def benchmark_apply(operators: list, states: list) -> None:
    """Benchmark apply method."""
    for op, psi in zip(operators, states):
        result = op.apply(psi)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_op,",
    it.chain(
        [(q, fp.PauliOp) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_apply_n_qubits(
    benchmark: Callable,
    generate_random_complex: Callable,
    prepared_operators: list,
    qubits: int,
) -> None:
    """Benchmark PauliOp multiplication with provided state vector."""
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_operators))
    ]

    benchmark(benchmark_apply, operators=prepared_operators, states=prepared_states)


@pytest.mark.parametrize(
    "qubits,states,pauli_op,",
    it.chain(
        [
            (q, n, fp.PauliOp)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
        [
            (q, n, pp.PauliOp)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
    ),
    ids=resolve_parameter_repr,
)
def test_apply_batch_n_qubits_n_states(
    benchmark: Callable,
    prepared_operators: list,
    generate_random_complex: Callable,
    qubits: int,
    states: int,
) -> None:
    """Benchmark PauliOp multiplication with provided set of state vectors."""
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_operators))
    ]

    benchmark(benchmark_apply, operators=prepared_operators, states=prepared_states)


def benchmark_expectation_value(operators: list, states: list) -> None:
    """Benchmark expectation_value method."""
    for op, psi in zip(operators, states):
        result = op.expectation_value(psi)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_op,",
    it.chain(
        [(q, fp.PauliOp) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_expectation_value_n_qubits(
    benchmark: Callable,
    prepared_operators: list,
    generate_random_complex: Callable,
    qubits: int,
) -> None:
    """Benchmark PauliOp expectation_value with provided state vector."""
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims) for _ in range(len(prepared_operators))
    ]

    benchmark(
        benchmark_expectation_value,
        operators=prepared_operators,
        states=prepared_states,
    )


@pytest.mark.parametrize(
    "qubits,states,pauli_op,",
    it.chain(
        [
            (q, n, fp.PauliOp)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
        [
            (q, n, pp.PauliOp)
            for q in QUBITS_TO_BENCHMARK
            for n in N_STATES_TO_BENCHMARK
        ],
    ),
    ids=resolve_parameter_repr,
)
def test_expectation_value_batch_n_qubits_n_states(
    benchmark: Callable,
    prepared_operators: list,
    generate_random_complex: Callable,
    qubits: int,
    states: int,
) -> None:
    """Benchmark PauliOp expectation_value with provided set of state vectors."""
    n_dims = 1 << qubits
    prepared_states = [
        generate_random_complex(n_dims, states) for _ in range(len(prepared_operators))
    ]

    benchmark(
        benchmark_expectation_value,
        operators=prepared_operators,
        states=prepared_states,
    )


def benchmark_arithmetic(left_ops: list, right_ops: list) -> None:
    """Benchmark PauliOp arithmetic."""
    for l_op, r_op in zip(left_ops, right_ops):
        p_op1 = l_op + r_op  # noqa: F841
        p_op2 = r_op - l_op  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_op,",
    it.chain(
        [(q, fp.PauliOp) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_arithmetic_n_qubits(
    benchmark: Callable,
    prepared_operators: list,
    qubits: int,
) -> None:
    """Benchmark addition and subtraction of PauliOps."""
    benchmark(
        benchmark_arithmetic,
        left_ops=prepared_operators,
        right_ops=reversed(prepared_operators),
    )


def benchmark_inplace_arithmetic(left_ops: list, right_ops: list) -> None:
    """Benchmark PauliOp inplace arithmetic."""
    for l_op, r_op in zip(left_ops, right_ops):
        l_op -= r_op
        r_op += l_op


@pytest.mark.parametrize(
    "qubits,pauli_op,",
    it.chain(
        [(q, fp.PauliOp) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_inplace_arithmetic_n_qubits(
    benchmark: Callable,
    prepared_operators: list,
    qubits: int,
) -> None:
    """Benchmark inplace addition and subtraction of PauliOps."""
    benchmark(
        benchmark_inplace_arithmetic,
        left_ops=prepared_operators,
        right_ops=reversed(prepared_operators),
    )


def benchmark_matmul(left_ops: list, right_ops: list) -> None:
    """Benchmark PauliOp/PauliString matrix multiplication."""
    for left, right in zip(left_ops, right_ops):
        op = left @ right  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_op,",
    it.chain(
        [(q, fp.PauliOp) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_multiplication_n_qubits(
    benchmark: Callable,
    prepared_operators: list,
    qubits: int,
) -> None:
    """Benchmark matrix multiplication of PauliOps."""
    benchmark(
        benchmark_matmul,
        left_ops=prepared_operators,
        right_ops=reversed(prepared_operators),
    )


@pytest.mark.parametrize(
    "qubits,pauli_op,pauli_string,",
    it.chain(
        [(q, fp.PauliOp, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, pp.PauliOp, pp.PauliString) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_multiplication_with_string_n_qubits(
    benchmark: Callable,
    pauli_strings_shuffled: Callable,
    prepared_operators: list,
    pauli_string: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
) -> None:
    """Benchmark matrix multiplication of PauliOp with PauliString."""
    prepared_strings = list(
        map(
            lambda s: pauli_string(s),
            pauli_strings_shuffled(qubits, limit=len(prepared_operators)),
        )
    )

    benchmark(benchmark_matmul, left_ops=prepared_operators, right_ops=prepared_strings)


if __name__ == "__main__":
    pytest.main()
