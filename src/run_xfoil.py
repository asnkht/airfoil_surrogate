from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XFOIL_EXECUTABLE = Path.home()/"Downloads"/"XFOIL6.99"/"xfoil.exe"

AIRFOIL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "airfoils"
    / "coord_seligFmt"
    / "2032c.dat"
)
xfoil_dir = XFOIL_EXECUTABLE.parent
local_airfoil = xfoil_dir / "test_airfoil.dat"

local_airfoil.write_text(
    AIRFOIL_FILE.read_text(encoding="utf-8", errors="replace"),
    encoding="ascii",
    errors="replace",
)

OUTPUT_FILE= PROJECT_ROOT/"data"/"raw"/"test_polar.txt"
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

if not XFOIL_EXECUTABLE.exists():
    raise FileNotFoundError(f"XFOIL executable not found at {XFOIL_EXECUTABLE}")

if not AIRFOIL_FILE.exists():
    raise FileNotFoundError(f"Airfoil coordinate file not found at {AIRFOIL_FILE}")


commands = f"""LOAD test_airfoil.dat
PANE
OPER
VISC 1000000
ITER 500
PACC
{OUTPUT_FILE}

ASEQ -2 6 1


QUIT
"""


print(f"XFOIL executable: {XFOIL_EXECUTABLE}")
print(f"Airfoil: {AIRFOIL_FILE.name}")
print("Running XFOIL...")

result = subprocess.run(
    [str(XFOIL_EXECUTABLE)],
    input=commands,
    text=True,
    capture_output=True,
    timeout=60,
    cwd =str(xfoil_dir),
)
print(result.stdout)
if result.stderr:
    print(f"\nXFOIL error: {result.stderr}")
if OUTPUT_FILE.exists():
    print(f"\nPolar data saved to: {OUTPUT_FILE}")
    print(OUTPUT_FILE.read_text(errors="replace")[-2000:])
else:
    print(f"\nPolar data file not created. Inspect XFOIL output for errors.")