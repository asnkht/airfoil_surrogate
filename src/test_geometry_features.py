
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from geometry_features import (
    load_airfoil_coordinates,
    resample_airfoil,
    validate_single_element_airfoil,
)


project_root = Path(__file__).resolve().parent.parent

airfoil_file = (
    project_root
    / "data"
    / "raw"
    / "airfoils"
    / "coord_seligFmt"
    / "ag03.dat"
)

coordinates = load_airfoil_coordinates(airfoil_file)

is_valid, reason = validate_single_element_airfoil(coordinates)

print("Airfoil:", airfoil_file.name)
print("Coordinate array shape:", coordinates.shape)
print("Geometry valid:", is_valid)
print("Validation result:", reason)

print("First five coordinate points:")
print(coordinates[:5])

if is_valid:
    features = resample_airfoil(coordinates)

    print("Feature vector shape:", features.shape)
    print("First five upper-surface heights:", features[:5])
    print("First five lower-surface heights:", features[100:105])

    x_grid = np.linspace(0.0, 1.0, 100)
    upper_y = features[:100]
    lower_y = features[100:]

    plt.plot(x_grid, upper_y, label="Upper surface")
    plt.plot(x_grid, lower_y, label="Lower surface")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"Resampled {airfoil_file.stem} Airfoil")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("Skipping this geometry. No resampling or plot generated.")
