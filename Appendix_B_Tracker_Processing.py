"""
Tracker coordinate processing for the five-pendulum magnetic oscillator.

Expected trial CSV columns:
time,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5

Expected equilibrium CSV columns:
x1,y1,x2,y2,x3,y3,x4,y4,x5,y5

Pivot coordinates are entered in PIVOTS below using the same coordinate
system and units as the Tracker export.
"""

import numpy as np
import pandas as pd

TRACKER_FILE = "tracker_trial.csv"
EQUILIBRIUM_FILE = "equilibrium_positions.csv"
OUTPUT_FILE = "angular_displacement.csv"

PIVOTS = {
    1: (0.0000, 0.0000),
    2: (0.0400, 0.0000),
    3: (0.0800, 0.0000),
    4: (0.1200, 0.0000),
    5: (0.1600, 0.0000),
}


def angular_position(x, y, xp, yp):
    """Return theta_i(t) = atan2(x_i-x_p,i, y_p,i-y_i), in radians."""
    return np.arctan2(x - xp, yp - y)


def main():
    trial = pd.read_csv(TRACKER_FILE)
    equilibrium = pd.read_csv(EQUILIBRIUM_FILE)

    processed = pd.DataFrame({"time": trial["time"]})

    for i in range(1, 6):
        xp, yp = PIVOTS[i]

        theta = angular_position(
            trial[f"x{i}"].to_numpy(),
            trial[f"y{i}"].to_numpy(),
            xp,
            yp,
        )

        theta_eq = angular_position(
            equilibrium[f"x{i}"].to_numpy(),
            equilibrium[f"y{i}"].to_numpy(),
            xp,
            yp,
        )

        theta0 = float(np.mean(theta_eq))
        q = theta - theta0

        length = np.sqrt(
            (trial[f"x{i}"].to_numpy() - xp) ** 2
            + (trial[f"y{i}"].to_numpy() - yp) ** 2
        )

        processed[f"theta{i}_rad"] = theta
        processed[f"theta0_{i}_rad"] = theta0
        processed[f"q{i}_rad"] = q
        processed[f"length{i}_m"] = length

        print(
            f"Magnet {i}: theta0={theta0:.6f} rad, "
            f"mean length={length.mean():.6f} m, "
            f"length SD={length.std(ddof=1):.6e} m"
        )

    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
