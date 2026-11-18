
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

files = {
    "Training": "airfoil_train.csv",
    "Validation": "airfoil_validation.csv",
    "Testing": "airfoil_test.csv",
}

datasets = {
    name: pd.read_csv(DATA / filename)
    for name, filename in files.items()
}

# Check identical columns
reference_columns = list(datasets["Training"].columns)

for name, df in datasets.items():
    assert list(df.columns) == reference_columns, (
        f"{name} columns do not match!"
    )

# Check no overlapping airfoil geometries
names = {
    name: set(df["airfoil"].astype(str))
    for name, df in datasets.items()
}

assert not (names["Training"] & names["Validation"])
assert not (names["Training"] & names["Testing"])
assert not (names["Validation"] & names["Testing"])

# Check missing and infinite numerical values
for name, df in datasets.items():
    numeric = df.select_dtypes(include=np.number)

    assert not numeric.isna().any().any(), (
        f"{name} contains missing numerical values!"
    )
    assert np.isfinite(numeric.to_numpy()).all(), (
        f"{name} contains infinite values!"
    )

print("ALL DATASET CHECKS PASSED")

for name, df in datasets.items():
    print(f"{name}: {df.shape}")

print("No airfoil overlap.")
print("No missing numerical values.")
print("No infinite numerical values.")
