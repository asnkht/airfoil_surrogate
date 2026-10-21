
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AIRFOIL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "airfoils"
    / "coord_seligFmt"
)

coordinate_files = sorted(
    path
    for path in AIRFOIL_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() == ".dat"
)

print(f"Directory: {AIRFOIL_DIR}")
print(f"Directory exists: {AIRFOIL_DIR.exists()}")
print(f"Airfoil coordinate files found: {len(coordinate_files)}")

print("\nFirst 20 airfoils: ")
for path in coordinate_files[:20]:
    print(f"  {path.name}")

if coordinate_files:
    first_file = coordinate_files[0]

    print(f"\nFirst 12 lines of {first_file.name}:")
    with first_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if line_number > 12:
                break
            print(line.rstrip())
