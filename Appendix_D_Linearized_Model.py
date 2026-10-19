"""
Linearized analytical model for the five-pendulum magnetic oscillator.

The model solves the generalized eigenvalue problem

    K(d) phi_m = omega_m^2 M phi_m

with magnetic coupling stiffness

    K_mag = 6 mu0 mu^2 l^2 / (pi d^5).

The physical parameters are those used in the EE parameter table.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh

MU0 = 4 * np.pi * 1e-7
G = 9.81

M_MAGNET = 0.0796       # kg
M_ROD = 0.0124          # kg, one rod
L_ROD = 0.160           # m
L = 0.170               # m, pivot axis to magnet centre
R = 0.0150              # m
H = 0.0150              # m
MU_DIPOLE = 0.80        # A m^2
N = 5


def moment_of_inertia():
    return (
        (2.0 / 3.0) * M_ROD * L_ROD**2
        + M_MAGNET * L**2
        + (1.0 / 12.0) * M_MAGNET * (3 * R**2 + H**2)
    )


def coupling_laplacian(n):
    matrix = np.zeros((n, n))

    for i in range(n - 1):
        matrix[i, i] += 1
        matrix[i + 1, i + 1] += 1
        matrix[i, i + 1] -= 1
        matrix[i + 1, i] -= 1

    return matrix


def normal_modes(d):
    inertia = moment_of_inertia()

    k_gravity = G * (M_MAGNET * L + M_ROD * L_ROD)
    k_magnetic = (
        6 * MU0 * MU_DIPOLE**2 * L**2
        / (np.pi * d**5)
    )

    mass_matrix = inertia * np.eye(N)
    stiffness_matrix = (
        k_gravity * np.eye(N)
        + k_magnetic * coupling_laplacian(N)
    )

    eigenvalues, eigenvectors = eigh(stiffness_matrix, mass_matrix)
    eigenvalues = np.clip(eigenvalues, 0, None)

    omega = np.sqrt(eigenvalues)
    frequency = omega / (2 * np.pi)

    return frequency, eigenvectors, k_magnetic


def main():
    print(f"I = {moment_of_inertia():.6e} kg m^2")

    rows = []
    vector_rows = []

    for d in np.arange(0.04, 0.121, 0.01):
        frequency, eigenvectors, k_magnetic = normal_modes(d)

        for mode in range(N):
            rows.append(
                {
                    "d_m": round(float(d), 2),
                    "Mode": mode + 1,
                    "Linear_frequency_Hz": float(frequency[mode]),
                    "Kmag_Nm_per_rad": float(k_magnetic),
                }
            )

            vector_rows.append(
                {
                    "d_m": round(float(d), 2),
                    "Mode": mode + 1,
                    **{
                        f"phi_{j + 1}": float(eigenvectors[j, mode])
                        for j in range(N)
                    },
                }
            )

    pd.DataFrame(rows).to_csv("linearized_frequencies.csv", index=False)
    pd.DataFrame(vector_rows).to_csv("linearized_eigenvectors.csv", index=False)

    print("Saved linearized_frequencies.csv")
    print("Saved linearized_eigenvectors.csv")


if __name__ == "__main__":
    main()
