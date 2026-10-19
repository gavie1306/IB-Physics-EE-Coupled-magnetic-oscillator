"""
Nonlinear numerical model for the five-pendulum magnetic oscillator.

The equations of motion are

I theta_i'' =
    tau_mag,i
    - g(m_M l + m_r L_r) sin(theta_i)
    - gamma_i theta_i'
    - alpha_i sgn(theta_i').

The magnetic model uses an equivalent bound surface-current
representation for uniformly magnetized finite cylindrical magnets and
includes every magnet pair.

Required auxiliary file:
    damping_parameters.csv

Expected damping_parameters.csv columns:
    Magnet,gamma_Nms,alpha_Nm

The surface discretization settings can be changed below to test
numerical convergence.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

MU0 = 4 * np.pi * 1e-7
G = 9.81
N = 5

M_MAGNET = 0.0796
M_ROD = 0.0124
L_ROD = 0.160
L = 0.170
R = 0.0150
H = 0.0150
MAGNETIZATION = 7.6e4

N_PHI = 40
N_Z = 16

INERTIA = (
    (2.0 / 3.0) * M_ROD * L_ROD**2
    + M_MAGNET * L**2
    + (1.0 / 12.0) * M_MAGNET * (3 * R**2 + H**2)
)


def load_damping(path="damping_parameters.csv"):
    table = pd.read_csv(path).sort_values("Magnet")
    gamma = table["gamma_Nms"].to_numpy(dtype=float)
    alpha = table["alpha_Nm"].to_numpy(dtype=float)

    if len(gamma) != N or len(alpha) != N:
        raise ValueError("Damping file must contain one row for each of five magnets.")

    return gamma, alpha


def pivot_positions(d):
    return np.column_stack(
        (
            np.arange(N, dtype=float) * d,
            np.zeros(N, dtype=float),
            np.zeros(N, dtype=float),
        )
    )


def magnet_centres(theta, d):
    pivots = pivot_positions(d)

    centres = pivots.copy()
    centres[:, 0] += L * np.sin(theta)
    centres[:, 1] -= L * np.cos(theta)

    return centres


def local_cylinder_surface():
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    z = np.linspace(-H / 2, H / 2, N_Z)

    dphi = 2 * np.pi / N_PHI
    dz = H / (N_Z - 1)

    points = []
    normals = []
    areas = []

    for zz in z:
        for pp in phi:
            points.append(
                [R * np.cos(pp), R * np.sin(pp), zz]
            )
            normals.append(
                [np.cos(pp), np.sin(pp), 0.0]
            )
            areas.append(R * dphi * dz)

    return (
        np.asarray(points, dtype=float),
        np.asarray(normals, dtype=float),
        np.asarray(areas, dtype=float),
    )


SURFACE_POINTS, SURFACE_NORMALS, SURFACE_AREAS = local_cylinder_surface()


def bound_surface_current(normal):
    magnetization_vector = np.array([0.0, 0.0, MAGNETIZATION])
    return np.cross(magnetization_vector, normal)


def magnetic_field(field_point, source_centre):
    total = np.zeros(3)

    for local_point, normal, dS in zip(
        SURFACE_POINTS,
        SURFACE_NORMALS,
        SURFACE_AREAS,
    ):
        source_point = source_centre + local_point
        separation = field_point - source_point
        distance = np.linalg.norm(separation)

        if distance < 1e-10:
            continue

        kb = bound_surface_current(normal)

        total += (
            MU0
            / (4 * np.pi)
            * np.cross(kb, separation)
            / distance**3
            * dS
        )

    return total


def magnetic_force_pair(target_centre, source_centre):
    total = np.zeros(3)

    for local_point, normal, dS in zip(
        SURFACE_POINTS,
        SURFACE_NORMALS,
        SURFACE_AREAS,
    ):
        field_point = target_centre + local_point
        field = magnetic_field(field_point, source_centre)
        kb = bound_surface_current(normal)

        total += np.cross(kb, field) * dS

    return total


def magnetic_torques(theta, d):
    centres = magnet_centres(theta, d)
    pivots = pivot_positions(d)
    torque = np.zeros(N)

    for i in range(N):
        force = np.zeros(3)

        for j in range(N):
            if i == j:
                continue

            force += magnetic_force_pair(centres[i], centres[j])

        r_i = centres[i] - pivots[i]
        torque[i] = np.cross(r_i, force)[2]

    return torque


def equations_of_motion(t, state, d, gamma, alpha):
    theta = state[:N]
    omega = state[N:]

    tau_mag = magnetic_torques(theta, d)
    tau_g = -G * (M_MAGNET * L + M_ROD * L_ROD) * np.sin(theta)
    tau_viscous = -gamma * omega

    # Smooth sign approximation prevents numerical chatter near omega = 0.
    velocity_scale = 1e-4
    tau_coulomb = -alpha * np.tanh(omega / velocity_scale)

    acceleration = (
        tau_mag + tau_g + tau_viscous + tau_coulomb
    ) / INERTIA

    return np.concatenate((omega, acceleration))


def simulate(
    d,
    release_angle,
    duration=100.0,
    fs=120.0,
    damping_file="damping_parameters.csv",
):
    gamma, alpha = load_damping(damping_file)

    theta0 = np.zeros(N)
    theta0[0] = release_angle

    omega0 = np.zeros(N)
    initial_state = np.concatenate((theta0, omega0))

    time = np.arange(0, duration, 1 / fs)

    solution = solve_ivp(
        lambda t, y: equations_of_motion(t, y, d, gamma, alpha),
        (0, duration),
        initial_state,
        t_eval=time,
        method="DOP853",
        rtol=1e-7,
        atol=1e-9,
    )

    return solution.t, solution.y[:N].T


def dominant_frequencies(time, theta, f_min=0.1, f_max=4.0):
    dt = float(np.mean(np.diff(time)))
    spectra = []

    for i in range(N):
        signal = theta[:, i] - np.mean(theta[:, i])
        transform = np.abs(np.fft.rfft(signal))
        frequency = np.fft.rfftfreq(len(signal), d=dt)

        band = (frequency >= f_min) & (frequency <= f_max)
        peaks, _ = find_peaks(transform[band])

        spectra.append(
            (
                frequency[band][peaks],
                transform[band][peaks],
            )
        )

    return spectra


def main():
    # Set these to the experimentally verified release angle and separation.
    d = 0.04
    release_angle = 0.18

    time, theta = simulate(d, release_angle)

    output = pd.DataFrame({"time_s": time})
    for i in range(N):
        output[f"theta{i + 1}_rad"] = theta[:, i]

    output.to_csv("numerical_trajectory.csv", index=False)
    print("Saved numerical_trajectory.csv")

    peaks = dominant_frequencies(time, theta)
    for i, (frequency, amplitude) in enumerate(peaks, start=1):
        order = np.argsort(amplitude)[::-1]
        print(f"Magnet {i} strongest candidate frequencies:")
        print(np.round(frequency[order[:8]], 5))


if __name__ == "__main__":
    main()
