
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "airfoil_dataset_500.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

df = pd.read_csv(INPUT_CSV)

# Split by unique airfoil, never by individual rows.
airfoils = df["airfoil"].unique()

rng = np.random.default_rng(42)
rng.shuffle(airfoils)

train_names = set(airfoils[:343])
validation_names = set(airfoils[343:417])
test_names = set(airfoils[417:])

train_df = df[df["airfoil"].isin(train_names)].copy()
validation_df = df[df["airfoil"].isin(validation_names)].copy()
test_df = df[df["airfoil"].isin(test_names)].copy()

train_df.to_csv(OUTPUT_DIR / "airfoil_train.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "airfoil_validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "airfoil_test.csv", index=False)

print("Dataset splitting complete.")
for name, subset in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df),
]:
    print(
        f"{name}: {subset['airfoil'].nunique()} airfoils, "
        f"{len(subset)} rows"
    )

print("Total unique airfoils:", df["airfoil"].nunique())
print("Total rows:", len(df))
