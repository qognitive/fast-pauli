"""Benchmark fast_pauli operations with qiskit operations."""

import itertools as it
from typing import List, Callable

import pytest

import fast_pauli._fast_pauli as fp
import fast_pauli.pypauli as pp
from tests.conftest import resolve_parameter_repr

from qiskit.quantum_info import Pauli
from qiskit.quantum_info import SparsePauliOp

QUBITS_TO_BENCHMARK = [1, 2, 4, 10]


def benchmark_sum(paulis: List) -> None:
    for p0, p1 in zip(paulis, reversed(paulis)):
        _ = p0 + p1  # noqa: F841


def benchmark_dot(paulis: List) -> None:
    for p0, p1 in zip(paulis, reversed(paulis)):
        _ = p0 @ p1  # noqa: F841


def benchmark_to_dense(paulis: List) -> None:
    for p in paulis:
        if isinstance(p, (Pauli, SparsePauliOp)):
            _ = p.to_matrix()  # noqa: F841
        elif isinstance(p, (fp.PauliString, pp.PauliString,
                            fp.PauliOp, pp.PauliOp)):
            _ = p.to_tensor()  # noqa: F841
        else:
            raise ValueError(f"Unsupported Pauli type: {type(p)}")


def benchmark_squared(paulis: List) -> None:
    for p in paulis:
        _ = p @ p  # noqa: F841


operation_funcs = {
    "sum": benchmark_sum,
    "dot": benchmark_dot,
    "to_dense": benchmark_to_dense,
    "square": benchmark_squared,
}


@pytest.mark.parametrize(
    "pauli_class, operation_name, qubits",
    it.product(
        [fp.PauliOp, pp.PauliOp, SparsePauliOp],
        ["sum", "dot", "to_dense", "square"],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr
)
def test_PauliOp_SparsePauliOp(
    benchmark: Callable,
    pauli_class:
        type[fp.PauliOp] | type[pp.PauliOp] |
        type[SparsePauliOp],
    operation_name: str,
    qubits: int,
    pauli_strings_with_size: Callable,
) -> None:
    n_strings_limit = 128 if qubits > 4 else None
    if pauli_class in [fp.PauliOp, pp.PauliOp]:
        strings = pauli_strings_with_size(qubits, n_strings_limit)
        coeffs = [1.0] * len(strings)
        paulis = [pauli_class(coeffs, strings)]
    else:
        paulis = list(
            map(lambda s: pauli_class(s),
                pauli_strings_with_size(qubits, n_strings_limit))
        )
    benchmark_func = operation_funcs[operation_name]
    if operation_name == 'power':
        benchmark(benchmark_func, paulis=paulis, exponent=2)
    else:
        benchmark(benchmark_func, paulis=paulis)


@pytest.mark.parametrize(
    "pauli_class, operation_name, qubits",
    it.product(
        [fp.PauliString, pp.PauliString, Pauli],
        ["sum", "dot", "to_dense", "square"],
        QUBITS_TO_BENCHMARK,
    ),
    ids=resolve_parameter_repr
)
def test_PauliString_Pauli(
    benchmark: Callable,
    pauli_class:
        type[fp.PauliString] | type[pp.PauliString] |
        type[Pauli],
    operation_name: str,
    qubits: int,
    pauli_strings_with_size: Callable,
) -> None:
    n_strings_limit = 128 if qubits > 4 else None
    paulis = list(
        map(lambda s: pauli_class(s),
            pauli_strings_with_size(qubits, n_strings_limit))
    )
    benchmark_func = operation_funcs[operation_name]
    if operation_name == 'power':
        benchmark(benchmark_func, paulis=paulis, exponent=2)
    else:
        if pauli_class == Pauli and operation_name == "sum":
            return
        benchmark(benchmark_func, paulis=paulis)


def benchmark_pauliop_pauli_mult(
    pauli_ops: List,
    pauli_strings: List
) -> None:
    """Benchmark fast_pauli.PauliOp @ fast_pauli.PauliString."""
    for pOp, ps in zip(pauli_ops, pauli_strings):
        _ = pOp @ ps  # noqa: F841


def benchmark_qiskit_pauliop_pauli_mult(
    pauli_ops: List,
    pauli_strings: List
) -> None:
    """Benchmark Qiskit SparsePauliOp @ Qiskit Pauli."""
    for pOp, ps in zip(pauli_ops, pauli_strings):
        _ = pOp.dot(ps)  # noqa: F841


@pytest.mark.parametrize(
    "benchmark_func, pauli_op_class, pauli_string_class, qubits",
    [
        (benchmark_func, pauli_op_class, pauli_string_class, qubits)
        for (benchmark_func, pauli_op_class, pauli_string_class), qubits
        in it.product(
            [
                (benchmark_pauliop_pauli_mult, fp.PauliOp, fp.PauliString),
                (benchmark_qiskit_pauliop_pauli_mult, SparsePauliOp, Pauli),
            ],
            QUBITS_TO_BENCHMARK,
        )
    ],
    ids=resolve_parameter_repr,
)
def test_pauliop_pauli_mult(
    benchmark: Callable,
    benchmark_func: Callable,
    pauli_op_class: type[fp.PauliOp] | type[pp.PauliOp],
    pauli_string_class: type[fp.PauliString] | type[pp.PauliString],
    qubits: int,
    pauli_strings_with_size: Callable,
) -> None:
    n_strings_limit = 128 if qubits > 4 else None
    strings = pauli_strings_with_size(qubits, n_strings_limit)

    pauli_strings = [pauli_string_class(s) for s in strings]

    if pauli_op_class == fp.PauliOp:
        pauli_ops = [fp.PauliOp([1.0], [s]) for s in strings]
    elif pauli_op_class == SparsePauliOp:
        pauli_ops = [SparsePauliOp(p) for p in pauli_strings]
    else:
        raise NotImplementedError(f"Unknown PauliOp class: {pauli_op_class}")

    benchmark(benchmark_func, pauli_ops=pauli_ops, pauli_strings=pauli_strings)
