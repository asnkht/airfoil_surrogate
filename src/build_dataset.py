from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from geometry_features import (
    load_airfoil_coordinates,
    resample_airfoil,
    validate_single_element_airfoil,
)


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AIRFOIL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "airfoils"
    / "coord_seligFmt"
)

XFOIL_EXECUTABLE = (
    Path.home() / "Downloads" / "XFOIL6.99" / "xfoil.exe"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Keep the original 30-airfoil dataset and failure log unchanged.
OUTPUT_CSV = OUTPUT_DIR / "airfoil_dataset_300.csv"
FAILURE_LOG = OUTPUT_DIR / "dataset_failures_300.csv"

XFOIL_DIR = XFOIL_EXECUTABLE.parent
LOCAL_AIRFOIL = XFOIL_DIR / "test_airfoil.dat"
LOCAL_POLAR = XFOIL_DIR / "dataset_polar.txt"

REYNOLDS_NUMBER = 1_000_000
ANGLES_OF_ATTACK = np.arange(-2, 7, 1)
MAX_AIRFOILS = 300

EXCLUDED_FILES = {
    "30p-30n.dat",
    "30p-30n-main.dat",
    "30p-30n-flap.dat",
    "30p-30n-slat.dat",
}


# --------------------------------------------------
# 2. Parse polar data
# --------------------------------------------------

def parse_polar_file(polar_file):
    rows = []

    if not polar_file.exists():
        return rows

    for line in polar_file.read_text(
        errors="replace"
    ).splitlines():
        values = line.split()

        if len(values) < 7:
            continue

        try:
            numbers = [float(value) for value in values[:7]]
        except ValueError:
            continue

        if not np.isfinite(numbers).all():
            continue

        alpha, cl, cd, cdp, cm, top_xtr, bot_xtr = numbers

        if cd <= 0 or cdp < 0:
            continue

        rows.append({
            "alpha_deg": alpha,
            "cl": cl,
            "cd": cd,
            "cdp": cdp,
            "cm": cm,
            "top_xtr": top_xtr,
            "bot_xtr": bot_xtr,
        })

    return rows


# --------------------------------------------------
# 3. Run XFOIL at one angle
# --------------------------------------------------

def run_xfoil_at_angle(airfoil_file, alpha):
    LOCAL_AIRFOIL.write_text(
        airfoil_file.read_text(
            encoding="utf-8",
            errors="replace",
        ),
        encoding="ascii",
        errors="replace",
    )

    if LOCAL_POLAR.exists():
        LOCAL_POLAR.unlink()

    commands = f"""LOAD test_airfoil.dat
PANE
OPER
VISC {REYNOLDS_NUMBER}
ITER 500
PACC
dataset_polar.txt

ALFA {alpha}


QUIT
"""

    result = subprocess.run(
        [str(XFOIL_EXECUTABLE)],
        input=commands,
        text=True,
        capture_output=True,
        timeout=60,
        cwd=str(XFOIL_DIR),
    )

    output = result.stdout + "\n" + result.stderr

    # Require XFOIL to explicitly add a final polar point.
    if "Point added to stored polar" not in output:
        raise RuntimeError(
            "XFOIL did not confirm that the point was added. "
            f"Console tail:\n{output[-1200:]}"
        )

    rows = parse_polar_file(LOCAL_POLAR)

    # A one-angle run should yield exactly one matching numeric row.
    matching_rows = [
        row for row in rows
        if np.isclose(row["alpha_deg"], alpha, atol=0.01)
    ]

    if len(matching_rows) != 1:
        raise RuntimeError(
            f"Expected one polar row at alpha={alpha}; "
            f"found {len(matching_rows)}."
        )

    return matching_rows[0]


# --------------------------------------------------
# 4. Select candidate airfoils
# --------------------------------------------------

def find_valid_airfoils():
    candidates = sorted(AIRFOIL_DIR.glob("*.dat"))
    selected = []
    seen_geometries = set()

    excluded = {name.lower() for name in EXCLUDED_FILES}

    for airfoil_file in candidates:
        if airfoil_file.name.lower() in excluded:
            continue

        try:
            coordinates = load_airfoil_coordinates(airfoil_file)

            is_valid, reason = validate_single_element_airfoil(
                coordinates
            )

            if not is_valid:
                print(
                    f"Geometry skipped: {airfoil_file.name}: {reason}"
                )
                continue

            features = resample_airfoil(coordinates)

            if features.shape != (200,):
                print(
                    f"Geometry skipped: {airfoil_file.name}: "
                    "unexpected feature shape"
                )
                continue

            # Treat geometries with identical rounded features as duplicates.
            geometry_key = np.round(
                features, decimals=6
            ).tobytes()

            if geometry_key in seen_geometries:
                print(
                    f"Duplicate geometry skipped: "
                    f"{airfoil_file.name}"
                )
                continue

            seen_geometries.add(geometry_key)

            selected.append((airfoil_file, features))
            print(f"Geometry accepted: {airfoil_file.name}")

            if len(selected) >= MAX_AIRFOILS:
                break

        except Exception as error:
            print(
                f"Geometry skipped: {airfoil_file.name}: {error}"
            )

    print(f"\nTotal unique valid airfoils selected: {len(selected)}")
    return selected


# --------------------------------------------------
# 5. Build dataset
# --------------------------------------------------

def main():
    if not XFOIL_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"XFOIL not found: {XFOIL_EXECUTABLE}"
        )

    if not AIRFOIL_DIR.exists():
        raise FileNotFoundError(
            f"Airfoil directory not found: {AIRFOIL_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_airfoils = find_valid_airfoils()

    if not selected_airfoils:
        raise RuntimeError("No valid airfoils found.")

    dataset_rows = []
    failures = []

    for airfoil_file, geometry_features in selected_airfoils:
        print(f"\nProcessing {airfoil_file.name}")

        successful_points = 0

        for alpha in ANGLES_OF_ATTACK:
            try:
                polar = run_xfoil_at_angle(
                    airfoil_file, float(alpha)
                )

                row = {
                    "airfoil": airfoil_file.stem,
                    "alpha_deg": polar["alpha_deg"],
                    "reynolds_number": REYNOLDS_NUMBER,
                    "cd": polar["cd"],
                }

                for index, value in enumerate(geometry_features):
                    row[f"geometry_{index:03d}"] = value

                dataset_rows.append(row)
                successful_points += 1

                print(
                    f"  alpha={alpha:>3}: "
                    f"CD={polar['cd']:.6f} [accepted]"
                )

            except Exception as error:
                print(
                    f"  alpha={alpha:>3}: FAILED — {error}"
                )

                failures.append({
                    "airfoil": airfoil_file.name,
                    "alpha_deg": alpha,
                    "reason": str(error),
                })

        print(
            f"  Successful points: "
            f"{successful_points}/{len(ANGLES_OF_ATTACK)}"
        )

    if dataset_rows:
        dataset = pd.DataFrame(dataset_rows)

        dataset = dataset.drop_duplicates(
            subset=["airfoil", "alpha_deg", "reynolds_number"]
        )

        dataset.to_csv(OUTPUT_CSV, index=False)

        print("\nDataset generation complete.")
        print(f"Saved to: {OUTPUT_CSV}")
        print(f"Shape: {dataset.shape}")
        print(f"Airfoils represented: {dataset['airfoil'].nunique()}")
        print("Rows per airfoil:")
        print(dataset.groupby("airfoil").size())
        print(f"Missing values: {dataset.isna().sum().sum()}")

    else:
        print("\nNo dataset rows were generated.")

    pd.DataFrame(
        failures,
        columns=["airfoil", "alpha_deg", "reason"],
    ).to_csv(FAILURE_LOG, index=False)

    print(f"Failure log saved to: {FAILURE_LOG}")


if __name__ == "__main__":
    main()
