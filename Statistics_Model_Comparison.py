"""
Experimental statistics and comparison of experimental, linearized,
and nonlinear normal-mode frequencies.

Required files:
    experimental_frequency_data.csv
    linearized_frequencies.csv
    numerical_frequencies.csv

Expected experimental columns:
    d_m,Mode,Trial_1,Trial_2,Trial_3,Trial_4,Trial_5

Expected numerical columns:
    d_m,Mode,Numerical_frequency_Hz
"""

import numpy as np
import pandas as pd

EXPERIMENTAL_FILE = "experimental_frequency_data.csv"
LINEAR_FILE = "linearized_frequencies.csv"
NUMERICAL_FILE = "numerical_frequencies.csv"

TRIAL_COLUMNS = [
    "Trial_1",
    "Trial_2",
    "Trial_3",
    "Trial_4",
    "Trial_5",
]


def rmse(group, prediction_column):
    return np.sqrt(
        np.mean(
            (
                group[prediction_column]
                - group["Experimental_mean_Hz"]
            ) ** 2
        )
    )


def main():
    experimental = pd.read_csv(EXPERIMENTAL_FILE)

    experimental["Experimental_mean_Hz"] = (
        experimental[TRIAL_COLUMNS].mean(axis=1)
    )

    experimental["Sample_SD_Hz"] = (
        experimental[TRIAL_COLUMNS].std(axis=1, ddof=1)
    )

    experimental["SEM_Hz"] = (
        experimental["Sample_SD_Hz"] / np.sqrt(len(TRIAL_COLUMNS))
    )

    linear = pd.read_csv(LINEAR_FILE)
    numerical = pd.read_csv(NUMERICAL_FILE)

    data = experimental.merge(
        linear[["d_m", "Mode", "Linear_frequency_Hz"]],
        on=["d_m", "Mode"],
        how="left",
    )

    data = data.merge(
        numerical[["d_m", "Mode", "Numerical_frequency_Hz"]],
        on=["d_m", "Mode"],
        how="left",
    )

    # This matches the current residual definition used in the EE:
    # Z = (f_lin - f_exp) / s_f
    data["Z"] = (
        data["Linear_frequency_Hz"]
        - data["Experimental_mean_Hz"]
    ) / data["Sample_SD_Hz"]

    data["Signed_percentage_error"] = (
        (
            data["Linear_frequency_Hz"]
            - data["Experimental_mean_Hz"]
        )
        / data["Experimental_mean_Hz"]
        * 100
    )

    data["Absolute_discrepancy_Hz"] = np.abs(
        data["Linear_frequency_Hz"]
        - data["Experimental_mean_Hz"]
    )

    rows = []

    for d, group in data.groupby("d_m"):
        linear_rmse = rmse(group, "Linear_frequency_Hz")
        numerical_rmse = rmse(group, "Numerical_frequency_Hz")

        improvement = (
            1.0 - numerical_rmse / linear_rmse
        ) * 100 if linear_rmse != 0 else np.nan

        rows.append(
            {
                "d_m": d,
                "Linear_RMSE_Hz": linear_rmse,
                "Numerical_RMSE_Hz": numerical_rmse,
                "Numerical_improvement_percent": improvement,
            }
        )

    comparison = pd.DataFrame(rows)

    data.to_csv("processed_model_comparison.csv", index=False)
    comparison.to_csv("rmse_model_comparison.csv", index=False)

    print(data.to_string(index=False))
    print("\nRMSE comparison")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
