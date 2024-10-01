"""Process benchmark data."""

import pandas as pd
import plotly.express as px

df = pd.read_csv("benchmark_20241001_003605.csv")
# print(df)

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


for f in functions_to_save:
    # General processing

    # Simplify the name
    df_f = df[df["name"].str.contains(f)]
    df_f.loc[:, "name"] = df_f["name"].str.split("::").str[-1]

    # Extract C++/Python impl details
    df_f.loc[:, "impl"] = ["cpp" if "cpp" in name else "py" for name in df_f["name"]]

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
        title=f"Benchmark: {f}",
        xaxis_title="Number of Qubits",
        yaxis_title="Time (s)",
        template="plotly_white",
        font=dict(size=18),
    )
    fig.update_traces(marker=dict(size=12))
    fig.show()
