
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airfoil_dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:8], "...")

print("\n--- ROWS PER AIRFOIL ---")
print(df.groupby("airfoil").size())

print("\n--- ANGLE COVERAGE ---")
print(
    df.groupby("airfoil")["alpha_deg"]
    .agg(["min", "max", "count", "nunique"])
)

print("\n--- TARGET: DRAG COEFFICIENT ---")
print(df["cd"].describe())

print("\n--- DUPLICATE OPERATING POINTS ---")
print(
    df.duplicated(
        subset=["airfoil", "alpha_deg", "reynolds_number"]
    ).sum()
)

print("\n--- MISSING VALUES ---")
print(df.isna().sum().sum())

print("\n--- FIRST FIVE ROWS ---")
print(df[["airfoil", "alpha_deg", "reynolds_number", "cd"]].head())
