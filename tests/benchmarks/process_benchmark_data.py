"""Process benchmark data."""

import itertools
import os

import pandas as pd
import plotly.express as px

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(FILE_DIR, "../../docs/benchmark_results/figs")

df = pd.read_csv("benchmark_results.csv")

functions_to_save = [
    # "test_dense_conversion_n_qubits",
    "test_apply_n_qubits",
    "test_apply_batch_n_qubits_n_states",
    "test_expectation_value_n_qubits",
    "test_expectation_value_batch_n_qubits_n_states",
    "test_multiplication_n_qubits",
    "test_arithmetic_n_qubits",
    # "test_string_sparse_composer_n_qubits",
]

# Extract what Class is being tested
# WARNING: This is brittle and assumes the class name is either
# "PauliString" or "PauliOp"
df.loc[:, "class"] = [
    "PauliString" if "PauliString" in name else "PauliOp" for name in df["name"]
]


for f, c in itertools.product(functions_to_save, ["PauliString", "PauliOp"]):
    # General processing

    # Simplify the name
    df_f = df[df["name"].str.contains(f) & (df["class"] == c)]
    if df_f.empty:
        continue

    df_f.loc[:, "name"] = df_f["name"].str.split("::").str[-1]

    # Extract C++/Python impl details
    df_f.loc[:, "impl"] = ["cpp" if "cpp" in name else "py" for name in df_f["name"]]
    print(df_f)

    # Use n_states as facet_col if it exists
    if df_f["param:states"].isna().any():
        fig = px.scatter(
            df_f,
            x="param:qubits",
            y="mean",
            error_y="stddev",
            color="impl",
            log_y=True,
        )

    else:
        fig = px.scatter(
            df_f,
            x="param:qubits",
            y="mean",
            error_y="stddev",
            color="impl",
            facet_col="param:states",
            log_y=True,
        )

    fig.update_layout(
        # title=f"Benchmark: {f} for class {c}",
        xaxis_title="Number of Qubits",
        yaxis_title="Time (s)",
        template="plotly_white",
        font=dict(size=18),
    )
    fig.update_traces(marker=dict(size=12))
    # fig.show()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.write_html(
        f"{OUTPUT_DIR}/{f}_{c}.html",
        full_html=False,
        include_plotlyjs="cdn",
    )
