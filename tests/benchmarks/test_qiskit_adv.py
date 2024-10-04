"""Advanced Qiskit benchmarks."""

import itertools as it
import os
from typing import Callable

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

import fast_pauli as fp

extra_benchmark_mark = pytest.mark.skipif(
    os.getenv("EXTRA_BENCHMARKS", "false").lower() != "true",
    reason="ENV variable EXTRA_BENCHMARKS is not set to true",
)


################################################################################
# Pauli String Apply
################################################################################
qiskit_params_sm = [pytest.param(nq, m) for nq, m in it.product([2, 8], ["qiskit"])]
qiskit_params_big = [
    pytest.param(nq, m, marks=extra_benchmark_mark)
    for nq, m in it.product([12, 14], ["qiskit"])
]

fast_pauli_params_sm = [
    pytest.param(nq, m) for nq, m in it.product([2, 8], ["fast_pauli"])
]

fast_pauli_params_big = [
    pytest.param(nq, m, marks=extra_benchmark_mark)
    for nq, m in it.product([12, 14, 16, 18, 20, 22, 24], ["fast_pauli"])
]


@pytest.mark.parametrize(
    "n_qubits, method",
    qiskit_params_sm + qiskit_params_big + fast_pauli_params_sm + fast_pauli_params_big,
)
def test_pauli_string_apply(benchmark: Callable, n_qubits: int, method: str) -> None:
    """Benchmark applying a Pauli String to a state."""
    np.random.seed(18)

    all_pauli_strings = [
        "".join(s) for s in it.combinations_with_replacement("IXYZ", n_qubits)
    ]
    pauli_string = np.random.choice(all_pauli_strings, size=1)[0]
    print(f"n_qubits: {n_qubits}, method: {method}, pauli_string: {pauli_string}")

    if method == "qiskit":
        v = Statevector(np.random.random(2**n_qubits).astype(np.complex128))
        ps = Pauli(pauli_string)

        def f() -> None:
            v.evolve(ps)

    else:
        v = np.random.random(2**n_qubits).astype(np.complex128)
        ps = fp.PauliString(pauli_string)

        def f() -> None:
            ps.apply(v)

    benchmark(f)


################################################################################
# Pauli String Expectation Value
################################################################################
# qiskit_params_sm = [
#     pytest.param(nq, m)
#     for nq, m in it.product([2, 8, 10, 12, 14, 16, 18, 20, 22, 24], ["qiskit"])
# ]
# fast_pauli_params_sm = [
#     pytest.param(nq, m)
#     for nq, m in it.product([2, 8, 10, 12, 14, 16, 18, 20, 22, 24], ["fast_pauli"])
# ]


# # NOTE: uses same settings as above
# @pytest.mark.parametrize(
#     "n_qubits, method",
#     qiskit_params_sm + qiskit_params_big + fast_pauli_params_sm +
#     fast_pauli_params_big,
# )
# def test_pauli_string_expectation_value(
#     benchmark: Callable, n_qubits: int, method: str
# ) -> None:
#     """Benchmark applying a Pauli String to a state."""
#     np.random.seed(18)
#     all_pauli_strings = [
#         "".join(s) for s in it.combinations_with_replacement("IXYZ", n_qubits)
#     ]
#     pauli_string = np.random.choice(all_pauli_strings, size=1)[0]
#     print(f"n_qubits: {n_qubits}, method: {method}, pauli_string: {pauli_string}")

#     v_np = np.random.random(2**n_qubits).astype(np.complex128)

#     if method == "qiskit":
#         v = Statevector(v_np)
#         ps = Pauli(pauli_string)

#         def f() -> None:
#             v.expectation_value(ps)

#     else:
#         ps = fp.PauliString(pauli_string)

#         def f() -> None:
#             ps.expectation_value(v_np)

#     benchmark(f)


################################################################################
# Pauli Op to Dense
################################################################################
# qiskit_params_sm = [
#     pytest.param(nq, ns, m)
#     for nq, ns, m in it.product([2, 8, 10, 12, 14], [10, 100, 1000], ["qiskit"])
# ]
# qiskit_params_big = [
#     pytest.param(nq, ns, m, marks=extra_benchmark_mark)
#     for nq, ns, m in it.product([12, 13, 14, 15], [10, 100, 1000], ["qiskit"])
# ]

# fast_pauli_params_sm = [
#     pytest.param(nq, ns, m)
#     for nq, ns, m in it.product([2, 8, 10, 12, 14], [10, 100, 1000], ["fast_pauli"])
# ]

# fast_pauli_params_big = [
#     pytest.param(nq, ns, m, marks=extra_benchmark_mark)
#     for nq, ns, m in it.product(
#         [12, 13, 14, 15, 16, 18], [10, 100, 1000], ["fast_pauli"]
#     )
# ]


# @pytest.mark.parametrize(
#     "n_qubits, n_pauli_strings, method",
#     qiskit_params_sm + qiskit_params_big + fast_pauli_params_sm +
#     fast_pauli_params_big,
# )
# def test_pauli_op_to_dense(
#     benchmark: Callable, n_qubits: int, n_pauli_strings: int, method: str
# ) -> None:
#     """Benchmark converting a Pauli Op to a dense matrix."""
#     np.random.seed(18)
#     all_paulis = [
#         "".join(s) for s in it.combinations_with_replacement("IXYZ", n_qubits)
#     ]
#     paulis = np.random.choice(all_paulis, size=n_pauli_strings, replace=True)

#     print(f"n_qubits: {n_qubits}, method: {method}, len(paulis): {len(paulis)}")

#     if method == "qiskit":
#         op = SparsePauliOp(paulis, coeffs=np.ones(len(paulis)))

#         def f() -> None:
#             op.to_matrix()  # noqa: F841
#     else:
#         op = fp.PauliOp(np.ones(len(paulis)), paulis)

#         def f() -> None:
#             op.to_tensor()  # noqa: F841

#     benchmark(f)


################################################################################
# Pauli Op Apply
################################################################################
qiskit_params_sm = [
    pytest.param(nq, ns, m)
    for nq, ns, m in it.product([2, 8, 10], [10, 100, 1000], ["qiskit"])
]
qiskit_params_big = [
    pytest.param(nq, ns, m, marks=extra_benchmark_mark)
    for nq, ns, m in it.product([12, 13, 14, 15], [10, 100, 1000], ["qiskit"])
]

fast_pauli_params_sm = [
    pytest.param(nq, ns, m)
    for nq, ns, m in it.product([2, 8, 10], [10, 100, 1000], ["fast_pauli"])
]

fast_pauli_params_big = [
    pytest.param(nq, ns, m, marks=extra_benchmark_mark)
    for nq, ns, m in it.product(
        [12, 13, 14, 15, 16, 18], [10, 100, 1000], ["fast_pauli"]
    )
]


@pytest.mark.parametrize(
    "n_qubits, n_pauli_strings, method",
    qiskit_params_sm + qiskit_params_big + fast_pauli_params_sm + fast_pauli_params_big,
)
def test_pauli_op_apply(
    benchmark: Callable,
    pauli_strings_with_size: Callable,
    n_qubits: int,
    n_pauli_strings: int,
    method: str,
) -> None:
    """Benchmark applying a Pauli Operator (in the general sense) to a state."""
    paulis = pauli_strings_with_size(n_qubits, limit=n_pauli_strings)
    print(
        f"n_qubits: {n_qubits}, n_pauli_strings:"
        + f" {n_pauli_strings} ?= {len(paulis)}, method: {method}"
    )

    if method == "qiskit":
        v = Statevector(np.random.random(2**n_qubits).astype(np.complex128))
        op = SparsePauliOp(paulis, coeffs=np.ones(len(paulis)))

        def f() -> None:
            v.evolve(op)
    else:
        op = fp.PauliOp(np.ones(len(paulis)), paulis)
        v = np.random.random(2**n_qubits).astype(np.complex128)

        def f() -> None:
            op.apply(v)

    benchmark(f)


################################################################################
# Pauli Op Expectation Value
################################################################################
# @pytest.mark.parametrize(
#     "n_qubits, n_pauli_strings, method",
#     qiskit_params_sm + qiskit_params_big + fast_pauli_params_sm +
#     fast_pauli_params_big,
# )
# def test_pauli_op_expectation_value(
#     benchmark: Callable, n_qubits: int, n_pauli_strings: int, method: str
# ) -> None:
#     """Benchmark applying a Pauli String to a state."""
#     np.random.seed(18)
#     all_pauli_strings = [
#         "".join(s) for s in it.combinations_with_replacement("IXYZ", n_qubits)
#     ]

#     pauli_strings = np.random.choice(
#         all_pauli_strings,
#         size=n_pauli_strings,
#         replace=True,
#     )
#     print(
#         f"n_qubits: {n_qubits}, n_pauli_strings: "
#         + f"{n_pauli_strings}, method: {method}, "
#         + f"len(pauli_strings): {len(pauli_strings)}"
#     )

#     v_np = np.random.random(2**n_qubits).astype(np.complex128)

#     if method == "qiskit":
#         v = Statevector(v_np)
#         op = SparsePauliOp(pauli_strings, coeffs=np.ones(len(pauli_strings)))

#         def f() -> None:
#             v.expectation_value(op)

#     else:
#         op = fp.PauliOp(np.ones(len(pauli_strings)), pauli_strings)

#         def f() -> None:
#             op.expectation_value(v_np)

#     benchmark(f)


################################################################################
# Pauli Op Expectation Value Batch
################################################################################
qiskit_params_sm = [
    pytest.param(nq, npauli, nstate, m)
    for nq, npauli, nstate, m in it.product([2, 8, 10, 12], [1024], [1, 10], ["qiskit"])
]
qiskit_params_big = [
    pytest.param(nq, npauli, nstate, m, marks=extra_benchmark_mark)
    for nq, npauli, nstate, m in it.product(
        [2, 8, 10, 12, 14, 16], [1024], [1, 10, 100], ["qiskit"]
    )
]

fast_pauli_params_sm = [
    pytest.param(nq, npauli, nstate, m)
    for nq, npauli, nstate, m in it.product(
        [2, 8, 10, 12], [1024], [1, 10, 100, 1000], ["fast_pauli"]
    )
]

fast_pauli_params_big = [
    pytest.param(nq, npauli, nstate, m, marks=extra_benchmark_mark)
    for nq, npauli, nstate, m in it.product(
        [14, 15, 16, 18], [1024], [1, 10, 100, 1000], ["fast_pauli"]
    )
]


@pytest.mark.parametrize(
    "n_qubits, n_pauli_strings, n_states, method",
    qiskit_params_sm + qiskit_params_big + fast_pauli_params_sm + fast_pauli_params_big,
)
def test_pauli_op_expectation_value_batch(
    benchmark: Callable, n_qubits: int, n_pauli_strings: int, n_states: int, method: str
) -> None:
    """Benchmark the expectation value of a Pauli Operator to a batch of states."""
    np.random.seed(18)
    all_pauli_strings = [
        "".join(s) for s in it.combinations_with_replacement("IXYZ", n_qubits)
    ]
    pauli_strings = np.random.choice(
        all_pauli_strings, size=n_pauli_strings, replace=True
    )
    print(
        f"n_qubits: {n_qubits}, n_pauli_strings:"
        + f" {n_pauli_strings}, n_states: {n_states}, "
        + f"method: {method}, "
        + f"len(pauli_strings): {len(pauli_strings)}"
    )

    v_np = np.random.rand(2**n_qubits, n_states).astype(np.complex128)

    if method == "qiskit":
        op = SparsePauliOp(pauli_strings, coeffs=np.ones(len(pauli_strings)))
        v_np = v_np.T.copy()
        state_batch = [Statevector(v) for v in v_np]

        def f() -> None:
            results = [s.expectation_value(op) for s in state_batch]  # noqa: F841

    else:
        op = fp.PauliOp(np.ones(len(pauli_strings)), pauli_strings)

        def f() -> None:
            op.expectation_value(v_np)

    benchmark(f)
