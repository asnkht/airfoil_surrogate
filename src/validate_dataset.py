
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airfoil_dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

expected_angles = np.arange(-2, 7, 1)

print("--- MISSING ANGLES ---")

for airfoil, group in df.groupby("airfoil"):
    present = set(group["alpha_deg"].astype(float))
    missing = sorted(set(expected_angles) - present)

    print(
        f"{airfoil}: "
        f"{'none' if not missing else missing}"
    )

print("\n--- DRAG COEFFICIENT RANGE BY AIRFOIL ---")
print(
    df.groupby("airfoil")["cd"]
    .agg(["min", "max", "mean", "std"])
    .round(6)
)

plt.figure(figsize=(10, 6))

for airfoil, group in df.groupby("airfoil"):
    group = group.sort_values("alpha_deg")

    plt.plot(
        group["alpha_deg"],
        group["cd"],
        marker="o",
        label=airfoil,
    )

plt.xlabel("Angle of attack (degrees)")
plt.ylabel("Drag coefficient ($C_D$)")
plt.title("XFOIL Drag Polar: Initial Dataset")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
