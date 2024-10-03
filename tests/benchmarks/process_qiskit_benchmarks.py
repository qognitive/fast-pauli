"""Process benchmark data."""

import os
import sys
from typing import Any

import pandas as pd
import plotly.express as px

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(FILE_DIR, "../../docs/benchmark_results/figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


#
# Helper functions
#
def format_fix(fig: Any) -> None:
    """Add consistent formatting to a plotly figure."""
    fig.update_layout(
        yaxis_title="Time (s)",
        template="seaborn",
        font=dict(size=18),
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_yaxes(exponentformat="power")
    fig.show()


COLOR_MAP = {
    "qiskit": "#0530AD",  # IBM blue
    "fast_pauli": "#f64135",  # red
}

#
# Load data
#

# WARNING DANGEROUS CLI (just don't wanna deal with argparse or click)
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
print(df)
df.rename(
    columns={
        "param:n_qubits": "N<sub>qubits</sub>",
        "param:n_pauli_strings": "N<sub>pauli strings</sub>",
        "param:method": "Library",
        "param:n_states": "N<sub>states</sub>",
    },
    inplace=True,
)
sorted_unique_n_pauli_strings = (
    df["N<sub>pauli strings</sub>"].dropna().unique().tolist()
)
sorted_unique_n_pauli_strings.sort()

if "N<sub>states</sub>" in df.columns:
    sorted_unique_n_states = df["N<sub>states</sub>"].dropna().unique().tolist()
    sorted_unique_n_states.sort()

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
    color_discrete_map=COLOR_MAP,
    # facet_col="N<sub>pauli strings</sub>",
    log_y=True,
    title="Pauli String Apply",
)
format_fix(fig)

fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_string_apply.html",
    full_html=False,
    include_plotlyjs="cdn",
)


#
# Pauli String Expectation Value
#
# df_ps = df[df["name"].str.contains("test_pauli_string_expectation_value")]

# fig = px.scatter(
#     df_ps,
#     x="N<sub>qubits</sub>",
#     y="mean",
#     error_y="stddev",
#     color="Library",
#     color_discrete_map=COLOR_MAP,
#     log_y=True,
# )
# format_fix(fig)
# fig.write_html(
#     f"{OUTPUT_DIR}/qiskit_pauli_string_expectation_value.html",
#     full_html=False,
#     include_plotlyjs="cdn",
# )


#
# Pauli Op to dense
#
df_op_dense = df[df["name"].str.contains("test_pauli_op_to_dense")]

fig = px.scatter(
    df_op_dense,
    x="N<sub>qubits</sub>",
    y="mean",
    error_y="stddev",
    color="Library",
    color_discrete_map=COLOR_MAP,
    facet_col="N<sub>pauli strings</sub>",
    log_y=True,
    category_orders={
        "N<sub>pauli strings</sub>": sorted_unique_n_pauli_strings,
    },
    title="Pauli Op to Dense",
)

format_fix(fig)
fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_op_to_dense.html",
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
    color_discrete_map=COLOR_MAP,
    facet_col="N<sub>pauli strings</sub>",
    log_y=True,
    category_orders={
        "N<sub>pauli strings</sub>": sorted_unique_n_pauli_strings,
    },
)

format_fix(fig)
fig.write_html(
    f"{OUTPUT_DIR}/qiskit_sparse_pauli_op_apply.html",
    full_html=False,
    include_plotlyjs="cdn",
)

# #
# # Pauli Op Expectation Value
# #
# df_op_exp = df[df["name"].str.contains("test_pauli_op_expectation_value")]

# fig = px.scatter(
#     df_op_exp,
#     x="N<sub>qubits</sub>",
#     y="mean",
#     error_y="stddev",
#     color="Library",
#     color_discrete_map=COLOR_MAP,
#     facet_col="N<sub>pauli strings</sub>",
#     log_y=True,
#     category_orders={
#         "N<sub>pauli strings</sub>": sorted_unique_n_pauli_strings,
#     },
# )
# format_fix(fig)
# fig.write_html(
#     f"{OUTPUT_DIR}/qiskit_pauli_op_expectation_value.html",
#     full_html=False,
#     include_plotlyjs="cdn",
# )


#
# Pauli Op Expectation Value Batch
#
df_op_exp_batch = df[df["name"].str.contains("test_pauli_op_expectation_value_batch")]

fig = px.scatter(
    df_op_exp_batch,
    x="N<sub>qubits</sub>",
    y="mean",
    error_y="stddev",
    color="Library",
    color_discrete_map=COLOR_MAP,
    log_y=True,
    facet_col="N<sub>states</sub>",
    category_orders={
        "N<sub>states</sub>": sorted_unique_n_states,
    },
)
format_fix(fig)
fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_op_expectation_value_batch.html",
    full_html=False,
    include_plotlyjs="cdn",
)
