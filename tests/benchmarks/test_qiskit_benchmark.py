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


"""Benchmark fast_pauli operations with qiskit operations."""

import itertools as it
from typing import Callable, List

import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp

import fast_pauli as fp
from tests.conftest import (
    QUBITS_TO_BENCHMARK,
    SAMPLE_STRINGS_LIMIT,
    resolve_parameter_repr,
)

N_OPERATORS_TO_BENCHMARK = 64


@pytest.fixture
def prepared_paulis(
    pauli_strings_with_size: Callable,
    pauli_class: type[fp.PauliString] | type[Pauli],
    qubits: int,
) -> list:
    """Fixture to provide initialized Pauli strings for testing."""
    return list(
        map(
            lambda s: pauli_class(s),
            pauli_strings_with_size(qubits, limit=SAMPLE_STRINGS_LIMIT),
        )
    )


@pytest.fixture
def prepared_pauliops(
    pauli_strings_with_size: Callable,
    pauli_strings_shuffled: Callable,
    generate_random_complex: Callable,
    pauli_class: type[fp.PauliOp] | type[SparsePauliOp],
    qubits: int,
) -> list:
    """Fixture to provide random initialized Pauli operators for testing."""
    operators = []
    for _ in range(N_OPERATORS_TO_BENCHMARK // 2):
        paulis = pauli_strings_shuffled(qubits, limit=SAMPLE_STRINGS_LIMIT)
        coeffs = generate_random_complex(len(paulis))
        if pauli_class == fp.PauliOp:
            operators.append(fp.PauliOp(coeffs, paulis))
        elif pauli_class == SparsePauliOp:
            operators.append(SparsePauliOp(paulis, coeffs=coeffs))
        else:
            raise ValueError(f"Unknown Pauli class: {pauli_class}")
    for _ in range(N_OPERATORS_TO_BENCHMARK // 2):
        paulis = pauli_strings_with_size(qubits, limit=SAMPLE_STRINGS_LIMIT)
        coeffs = generate_random_complex(len(paulis))
        if pauli_class == fp.PauliOp:
            operators.append(fp.PauliOp(coeffs, paulis))
        else:
            operators.append(SparsePauliOp(paulis, coeffs=coeffs))
    return operators


def benchmark_sum(paulis: List) -> None:
    """Benchmark addition."""
    for p0, p1 in zip(paulis, reversed(paulis)):
        _ = p0 + p1  # noqa: F841


def benchmark_dot(paulis: List) -> None:
    """Benchmark multiplication."""
    for p0, p1 in zip(paulis, reversed(paulis)):
        _ = p0 @ p1  # noqa: F841


def benchmark_qiskit_to_dense(paulis: List) -> None:
    """Benchmark Qiskit SparsePauliOp.to_matrix()."""
    for p in paulis:
        _ = p.to_matrix()  # noqa: F841


def benchmark_fp_to_dense(paulis: List) -> None:
    """Benchmark FastPauli PauliOp.to_tensor()."""
    for p in paulis:
        _ = p.to_tensor()  # noqa: F841


def benchmark_fp_squared(paulis: List) -> None:
    """Benchmark squaring for Fast_Pauli."""
    for p in paulis:
        _ = p @ p  # noqa: F841


def benchmark_qiskit_squared(paulis: List) -> None:
    """Benchmark squaring for Qiskit."""
    for p in paulis:
        _ = p.power(2)


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliOp, SparsePauliOp],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_sum_pauliops(
    benchmark: Callable,
    prepared_pauliops: List,
) -> None:
    """Benchmark addition for Pauli operators."""
    benchmark(benchmark_sum, paulis=prepared_pauliops)


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliString, Pauli],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_dot_paulis(
    benchmark: Callable,
    prepared_paulis: List,
) -> None:
    """Benchmark dot product for fp.PauliString and Qiskit Pauli."""
    benchmark(benchmark_dot, paulis=prepared_paulis)


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliOp, SparsePauliOp],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_dot_pauliops(
    benchmark: Callable,
    prepared_pauliops: List,
) -> None:
    """Benchmark dot product for fp.PauliOp and Qiskit SparsePauliOp."""
    benchmark(benchmark_dot, paulis=prepared_pauliops)


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliString, Pauli],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_to_dense_paulis(
    benchmark: Callable,
    prepared_paulis: List,
) -> None:
    """Benchmark conversion to dense matrix for fp.PauliString and Qiskit Pauli."""
    # Qiskit and Fast_Pauli use differently named methods to convert to dense matrix
    if type(prepared_paulis[0]) is fp.PauliString:
        benchmark(benchmark_fp_to_dense, paulis=prepared_paulis)
    elif type(prepared_paulis[0]) is Pauli:
        benchmark(benchmark_qiskit_to_dense, paulis=prepared_paulis)
    else:
        raise ValueError(f"Unknown Pauli class: {type(prepared_paulis[0])}")


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliOp, SparsePauliOp],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_to_dense_pauliops(
    benchmark: Callable,
    prepared_pauliops: List,
) -> None:
    """Benchmark conversion to dense matrix for fp.PauliOp and Qiskit SparsePauliOp."""
    # Qiskit and Fast_Pauli use differently named methods to convert to dense matrix
    if type(prepared_pauliops[0]) is fp.PauliOp:
        benchmark(benchmark_fp_to_dense, paulis=prepared_pauliops)
    elif type(prepared_pauliops[0]) is SparsePauliOp:
        benchmark(benchmark_qiskit_to_dense, paulis=prepared_pauliops)
    else:
        raise ValueError(f"Unknown Pauli class: {type(prepared_pauliops[0])}")


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliString, Pauli],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_square_paulis(
    benchmark: Callable,
    prepared_paulis: List,
) -> None:
    """Benchmark squaring for fp.PauliString and Qiskit Pauli."""
    if type(prepared_paulis[0]) is fp.PauliString:
        benchmark(benchmark_fp_squared, paulis=prepared_paulis)
    elif type(prepared_paulis[0]) is Pauli:
        benchmark(benchmark_qiskit_squared, paulis=prepared_paulis)
    else:
        raise ValueError(f"Unknown Pauli class: {type(prepared_paulis[0])}")


@pytest.mark.parametrize(
    "pauli_class, qubits",
    it.product(
        [fp.PauliOp, SparsePauliOp],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr,
)
def test_square_pauliops(
    benchmark: Callable,
    prepared_pauliops: List,
) -> None:
    """Benchmark squaring for fp.PauliOp and Qiskit SparsePauliOp."""
    if type(prepared_pauliops[0]) is fp.PauliOp:
        benchmark(benchmark_fp_squared, paulis=prepared_pauliops)
    elif type(prepared_pauliops[0]) is SparsePauliOp:
        benchmark(benchmark_qiskit_squared, paulis=prepared_pauliops)
    else:
        raise ValueError(f"Unknown Pauli class: {type(prepared_pauliops[0])}")


def benchmark_pauliop_pauli_mult(pauli_ops: List, pauli_strings: List) -> None:
    """Benchmark fast_pauli.PauliOp @ fast_pauli.PauliString."""
    for pOp, ps in zip(pauli_ops, pauli_strings):
        _ = pOp @ ps  # noqa: F841


def benchmark_qiskit_pauliop_pauli_mult(pauli_ops: List, pauli_strings: List) -> None:
    """Benchmark Qiskit SparsePauliOp @ Qiskit Pauli."""
    for pOp, ps in zip(pauli_ops, pauli_strings):
        _ = pOp.dot(ps)  # noqa: F841


@pytest.mark.parametrize(
    "qubits,pauli_class,pauli_string_class,",
    it.chain(
        [(q, fp.PauliOp, fp.PauliString) for q in QUBITS_TO_BENCHMARK],
        [(q, SparsePauliOp, Pauli) for q in QUBITS_TO_BENCHMARK],
    ),
    ids=resolve_parameter_repr,
)
def test_pauliop_pauli_mult(
    benchmark: Callable,
    pauli_class: type[fp.PauliOp] | type[SparsePauliOp],
    pauli_string_class: type[fp.PauliString] | type[Pauli],
    prepared_pauliops: List,
    qubits: int,
    pauli_strings_with_size: Callable,
) -> None:
    """Benchmark PauliOp @ PauliString vs Qiskit SparsePauliOp @ Pauli."""
    prepared_strings = [
        pauli_string_class(s)
        for s in pauli_strings_with_size(qubits, limit=SAMPLE_STRINGS_LIMIT)
    ]
    if pauli_class == fp.PauliOp:
        benchmark(
            benchmark_pauliop_pauli_mult,
            pauli_ops=prepared_pauliops,
            pauli_strings=prepared_strings,
        )
    else:
        benchmark(
            benchmark_qiskit_pauliop_pauli_mult,
            pauli_ops=prepared_pauliops,
            pauli_strings=prepared_strings,
        )
