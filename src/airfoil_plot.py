import numpy as np
import matplotlib.pyplot as plt

# Load airfoil coordinates from the project root
coordinates = np.loadtxt(
    "data/raw/n0012.dat",
    skiprows=2
)

# Extract x and y coordinates
x = coordinates[:, 0]
y = coordinates[:, 1]

# Plot the airfoil
plt.figure(figsize=(10, 4))
plt.plot(x, y, "-o", markersize=2)

plt.xlabel("x / chord")
plt.ylabel("y / chord")
plt.title("NACA 0012 Airfoil")
plt.axis("equal")
plt.grid(True)

plt.show()