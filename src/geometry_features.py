
from pathlib import Path

import numpy as np


def load_airfoil_coordinates(file_path):
    """Load an airfoil coordinate file as an N x 2 array."""
    file_path = Path(file_path)

    coordinates = np.loadtxt(
        file_path,
        skiprows=1,
        usecols=(0, 1),
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Invalid coordinate format: {file_path.name}")

    if len(coordinates) < 10:
        raise ValueError(f"Too few coordinate points: {file_path.name}")

    if not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite coordinates: {file_path.name}")

    return coordinates


def validate_single_element_airfoil(coordinates):
    """Apply conservative checks for an ordered, single-element airfoil."""
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        return False, "Expected an N x 2 coordinate array"

    if len(coordinates) < 10:
        return False, "Too few coordinate points"

    if not np.isfinite(coordinates).all():
        return False, "Coordinates contain non-finite values"

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    chord = np.ptp(x)
    if chord <= 0:
        return False, "Zero chord length"

    # Normalize the chord to avoid dependence on coordinate scale.
    x = (x - np.min(x)) / chord
    y = (y - np.mean(y)) / chord

    if np.ptp(y) > 0.5:
        return False, "Unusually large thickness or multiple elements"

    # The conventional coordinate order goes from trailing edge
    # to leading edge, then back to the trailing edge.
    le_index = int(np.argmin(x))

    if le_index < 3 or len(x) - le_index < 4:
        return False, "Leading edge is too close to an endpoint"

    first_dx = np.diff(x[: le_index + 1])
    second_dx = np.diff(x[le_index:])

    # Permit tiny coordinate irregularities, but reject substantial
    # reversals that suggest disconnected or nonstandard geometry.
    tolerance = 0.02

    first_reversals = np.count_nonzero(first_dx > tolerance)
    second_reversals = np.count_nonzero(second_dx < -tolerance)

    if first_reversals > max(1, int(0.05 * len(first_dx))):
        return False, "Coordinate order reverses before the leading edge"

    if second_reversals > max(1, int(0.05 * len(second_dx))):
        return False, "Coordinate order reverses after the leading edge"

    if np.ptp(x[: le_index + 1]) < 0.8:
        return False, "First surface does not span the chord"

    if np.ptp(x[le_index:]) < 0.8:
        return False, "Second surface does not span the chord"

    return True, "Passed basic ordered-airfoil checks"


def resample_airfoil(coordinates, num_points=100):
    """Return 100 upper and 100 lower surface heights."""
    coordinates = np.asarray(coordinates, dtype=float)

    valid, reason = validate_single_element_airfoil(coordinates)
    if not valid:
        raise ValueError(f"Unsupported geometry: {reason}")

    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    x_min = np.min(coordinates[:, 0])
    x_max = np.max(coordinates[:, 0])
    chord = x_max - x_min

    # Normalize x and y by the chord.
    x = (coordinates[:, 0] - x_min) / chord
    y = coordinates[:, 1] / chord

    le_index = int(np.argmin(x))

    first_surface = np.column_stack((x[: le_index + 1], y[: le_index + 1]))
    second_surface = np.column_stack((x[le_index:], y[le_index:]))

    # Sort from leading edge to trailing edge.
    first_surface = first_surface[np.argsort(first_surface[:, 0])]
    second_surface = second_surface[np.argsort(second_surface[:, 0])]

    def interpolate_surface(surface):
        unique_x, indices = np.unique(
            surface[:, 0], return_index=True
        )
        unique_y = surface[indices, 1]

        if len(unique_x) < 3:
            raise ValueError("Too few unique chordwise coordinates")

        grid = np.linspace(0.0, 1.0, num_points)

        # Do not extrapolate if a surface does not cover the chord.
        if unique_x[0] > 1e-3 or unique_x[-1] < 1.0 - 1e-3:
            raise ValueError("Surface does not cover the full chord")

        return np.interp(grid, unique_x, unique_y)

    first_y = interpolate_surface(first_surface)
    second_y = interpolate_surface(second_surface)

    # Determine which curve is upper based on their average separation.
    difference = first_y - second_y

    if np.mean(difference) >= 0:
        upper_y, lower_y = first_y, second_y
    else:
        upper_y, lower_y = second_y, first_y

    # Reject substantial crossings instead of hiding them with max/min.
    crossing_fraction = np.mean(upper_y < lower_y - 1e-4)

    if crossing_fraction > 0.05:
        raise ValueError("Surfaces cross excessively after resampling")

    return np.concatenate((upper_y, lower_y))
