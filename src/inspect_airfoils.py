
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIRFOIL_DIR = PROJECT_ROOT / "data" / "raw" / "airfoils"

print("Checking directory:", AIRFOIL_DIR)
print("Directory exists:", AIRFOIL_DIR.exists())

if AIRFOIL_DIR.exists():
    print("\nTop-level contents:")
    for item in AIRFOIL_DIR.iterdir():
        print(f"  {'[DIR] ' if item.is_dir() else '[FILE] '}{item.name}")

    print("\nFirst 30 files of any type:")
    all_files = [p for p in AIRFOIL_DIR.rglob("*") if p.is_file()]
    print("Total files:", len(all_files))

    for path in all_files[:30]:
        print(f"  {path.relative_to(AIRFOIL_DIR)}")
