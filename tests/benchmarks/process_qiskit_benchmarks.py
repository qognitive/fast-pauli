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

"""Process benchmark data from test_qiskit_adv.py."""

import os
import sys
from typing import Any

import pandas as pd
import plotly.express as px

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(FILE_DIR, "../../docs/benchmark_results/figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


################################################################################
# Helper functions
################################################################################
def format_fix(fig: Any) -> None:
    """Add consistent formatting to a plotly figure."""
    fig.update_layout(
        yaxis_title="Time (s)",
        template="seaborn",
        font=dict(size=18),
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_yaxes(exponentformat="power")


COLOR_MAP = {
    "qiskit": "#0530AD",  # IBM blue
    "fast_pauli": "#fb8501",  # orange
}


################################################################################
# Load data
################################################################################

# WARNING DANGEROUS CLI (just don't wanna deal with argparse or click)
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

# Rename columns to be more readable in final HTML plots
df.rename(
    columns={
        "param:n_qubits": "N<sub>qubits</sub>",
        "param:n_pauli_strings": "N<sub>pauli strings</sub>",
        "param:method": "Library",
        "param:n_states": "N<sub>states</sub>",
    },
    inplace=True,
)


def get_sorted_unique_n_pauli_strings(df: pd.DataFrame) -> list[int]:
    """Get the sorted unique number of pauli strings from a dataframe.

    Used to make sure that the subplots are ordering consistently.
    """
    if "N<sub>pauli strings</sub>" in df.columns:
        sorted_unique_n_pauli_strings: list[int] = (
            df["N<sub>pauli strings</sub>"].dropna().unique().tolist()
        )
        sorted_unique_n_pauli_strings.sort()
        return sorted_unique_n_pauli_strings
    else:
        rv: list[int] = []
        return rv


def get_sorted_unique_n_states(df: pd.DataFrame) -> list[int]:
    """Get the sorted unique number of states from a dataframe.

    Used to make sure that the subplots are ordering consistently.
    """
    if "N<sub>states</sub>" in df.columns:
        sorted_unique_n_states: list[int] = (
            df["N<sub>states</sub>"].dropna().unique().tolist()
        )
        sorted_unique_n_states.sort()
        return sorted_unique_n_states
    else:
        rv: list[int] = []
        return rv


################################################################################
# Pauli string apply
################################################################################
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
    # title="Pauli String Applied to a State Vector",
)
format_fix(fig)

fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_string_apply.html",
    full_html=False,
    include_plotlyjs="cdn",
)

################################################################################
# Pauli Op applied to a statevector
################################################################################
df_op_app = df[df["name"].str.contains("test_pauli_op_apply")]

fig = px.scatter(
    df_op_app,
    x="N<sub>qubits</sub>",
    y="mean",
    error_y="stddev",
    color="Library",
    color_discrete_map=COLOR_MAP,
    facet_col="N<sub>pauli strings</sub>",
    log_y=True,
    # Make sure the subplots are ordered correctly
    category_orders={
        "N<sub>pauli strings</sub>": get_sorted_unique_n_pauli_strings(df_op_app),
    },
    # title="Pauli Operator Applied to a State Vector",
)

format_fix(fig)
fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_op_apply.html",
    full_html=False,
    include_plotlyjs="cdn",
)

################################################################################
# Pauli Op Expectation Value Batch
################################################################################
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
        "N<sub>states</sub>": get_sorted_unique_n_states(df_op_exp_batch),
    },
    # title="Expectation Value of Pauli Operator for Batch of States",
)
format_fix(fig)
fig.write_html(
    f"{OUTPUT_DIR}/qiskit_pauli_op_expectation_value_batch.html",
    full_html=False,
    include_plotlyjs="cdn",
)
