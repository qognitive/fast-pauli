"""Process benchmark data."""

import os

import pandas as pd
import plotly.express as px

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(FILE_DIR, "../../docs/benchmark_results/figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("qiskit_jets.csv")
df.rename(
    columns={
        "param:n_qubits": "N<sub>qubits</sub>",
        "param:n_strings": "N<sub>pauli strings</sub>",
        "param:method": "Library",
    },
    inplace=True,
)
sorted_unique_n_pauli_strings = df["N<sub>pauli strings</sub>"].unique().tolist()
sorted_unique_n_pauli_strings.sort()

#
# Pauli string apply
#
df_ps = df[df["name"].str.contains("test_pauli_string_apply")]

fig = px.scatter(
    df_ps,
    x="N<sub>qubits</sub>",
    y="mean",
    error_y="stddev",
    color="Library",
    # facet_col="N<sub>pauli strings</sub>",
    log_y=True,
)

fig.update_layout(
    yaxis_title="Time (s)",
    template="plotly_white",
    font=dict(size=18),
)
fig.update_traces(marker=dict(size=12))
fig.show()

fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_string_apply.html",
    full_html=False,
    include_plotlyjs="cdn",
)


#
# Pauli String Expectation Value
#
df_ps = df[df["name"].str.contains("test_pauli_string_expectation_value")]

fig = px.scatter(
    df_ps,
    x="N<sub>qubits</sub>",
    y="mean",
    error_y="stddev",
    color="Library",
    log_y=True,
)

fig.update_layout(
    yaxis_title="Time (s)",
    template="plotly_white",
    font=dict(size=18),
)
fig.update_traces(marker=dict(size=12))
fig.show()

fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_string_expectation_value.html",
    full_html=False,
    include_plotlyjs="cdn",
)

#
# Pauli Op applied to a statevector
#
df_app = df[df["name"].str.contains("test_sparse_pauli_op_apply")]

fig = px.scatter(
    df_app,
    x="N<sub>qubits</sub>",
    y="mean",
    error_y="stddev",
    color="Library",
    facet_col="N<sub>pauli strings</sub>",
    log_y=True,
    category_orders={
        "N<sub>pauli strings</sub>": sorted_unique_n_pauli_strings,
    },
)


fig.update_layout(
    # title=f"Benchmark: {f} for class {c}",
    # xaxis_title="Number of Qubits",
    yaxis_title="Time (s)",
    template="plotly_white",
    font=dict(size=18),
)
fig.update_traces(marker=dict(size=12))
fig.show()

fig.write_html(
    f"{OUTPUT_DIR}/qiskit_sparse_pauli_op_apply.html",
    full_html=False,
    include_plotlyjs="cdn",
)
