from typing import Any
import numpy as np
import time

from qcog.src.classical.pauli_symbolic import ClassicalPauliSymbolic as PauliHSM
from qcog.src.classical.common import OptimizationMethod, StateMethod
from qcog.src.compiled.pauli.pauli_utils import get_state_iteration_unequal_weights

config: dict[str, Any] = {
    "hsm_type": "pauli",
    "data_params": {"train_size": 10000, "test_size": 1000, "dataset": "mnist"},
    "random_seed": 42,
    "initialization_params": {
        "qbits": 14,
        "pauli_weight": 2,
    },
    "train_params": {
        "weight_optimization_kwargs": {
            "optimization_method": OptimizationMethod.ANALYTIC.value,
        },
        "batch_size": 10000,
        "num_passes": 4,
        "state_kwargs": {
            "state_method": StateMethod.LOBPCG_FAST.value,
            "iterations": 10,
            "tolerance": 0.1,
        },
    },
    "forecast_params": {
        "state_method": StateMethod.LOBPCG_FAST.value,
        "iterations": 10,
    },
}


n_operators = 10000
n_data = 1000
operators = [str(i) for i in range(n_operators)]
data = np.zeros((n_data, n_operators))
states = np.zeros(
    (n_data, 2 ** config["initialization_params"]["qbits"]), dtype=np.complex128
)


hsm = PauliHSM(operators=operators, **config["initialization_params"])

ham_terms = hsm._generate_ham_components(data, operators)
A2, xA_weights, data_term = ham_terms

t0 = time.perf_counter()
new_states = get_state_iteration_unequal_weights(
    states, hsm._pauli_str_array, xA_weights
)

print(f"Time for x_kt * A_k |psi_t>: {time.perf_counter() - t0:.2f} s")
