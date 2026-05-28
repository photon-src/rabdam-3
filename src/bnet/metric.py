"""Bnet metric calculation.

Bnet compares the BDamage distribution of selected damage-sensitive sites
against the median BDamage value for all scored atoms in the structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import gaussian_kde


DEFAULT_TRAPEZIUM_COUNT = 99
DEFAULT_KDE_TAIL_WIDTHS = 3.0


@dataclass(frozen=True, slots=True)
class BnetResult:
    """Raw Bnet calculation result."""

    bnet: float
    median_bdamage: float
    left_area: float
    right_area: float
    site_count: int


def calculate_bnet(
    *,
    all_bdamage_values: Sequence[float],
    bnet_site_bdamage_values: Sequence[float],
    trapezium_count: int = DEFAULT_TRAPEZIUM_COUNT,
) -> BnetResult:
    """Calculate raw Bnet.

    Bnet is the ratio of the area under the KDE of Asp/Glu carboxyl oxygen
    BDamage values to the right of the all-atom median BDamage value, divided
    by the area to the left of that median.

    The KDE is evaluated on an evenly spaced, tail-padded grid. Trapeziums are
    assigned to the below-median area when their left x value is below the
    median; all remaining trapeziums are assigned to the above-median area.

    Parameters
    ----------
    all_bdamage_values
        BDamage values for all scored atoms in the structure.
    bnet_site_bdamage_values
        BDamage values for Asp/Glu side-chain carboxyl oxygen atoms.
    trapezium_count
        Number of trapeziums used for numerical integration. The default uses
        100 grid points and 99 trapeziums.

    Returns
    -------
    BnetResult
        Raw Bnet value plus supporting integration values.

    Raises
    ------
    ValueError
        If the inputs are empty, non-finite, too small for KDE estimation, or
        lead to zero below-median area.
    """

    all_values = _as_finite_1d_array(
        all_bdamage_values,
        name="all_bdamage_values",
    )
    site_values = _as_finite_1d_array(
        bnet_site_bdamage_values,
        name="bnet_site_bdamage_values",
    )

    if site_values.size < 2:
        raise ValueError(
            "Cannot calculate Bnet from fewer than two Bnet-site BDamage values."
        )

    if type(trapezium_count) is not int or trapezium_count < 1:
        raise ValueError("trapezium_count must be a positive integer.")

    median_bdamage = float(np.median(all_values))

    x_values = _bnet_kde_x_values(
        site_values,
        trapezium_count=trapezium_count,
    )
    y_values = _gaussian_kde_scott_1d(site_values, x_values)

    left_area, right_area = _split_trapezium_areas(
        x_values=x_values,
        y_values=y_values,
        split_value=median_bdamage,
    )

    if left_area <= 0.0:
        raise ValueError(
            "Cannot calculate Bnet because the below-median KDE area is zero."
        )

    return BnetResult(
        bnet=right_area / left_area,
        median_bdamage=median_bdamage,
        left_area=left_area,
        right_area=right_area,
        site_count=int(site_values.size),
    )


def _as_finite_1d_array(values: Sequence[float], *, name: str) -> np.ndarray:
    """Convert values to a finite one-dimensional float array."""

    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _gaussian_kde_scott_1d(
    sample_values: np.ndarray,
    evaluation_points: np.ndarray,
) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian KDE using Scott's bandwidth rule."""

    try:
        kde = gaussian_kde(dataset=sample_values, bw_method="scott")
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Cannot calculate KDE because Bnet-site values have zero variance."
        ) from error

    return kde(evaluation_points)


def _bnet_kde_x_values(
    sample_values: np.ndarray,
    *,
    trapezium_count: int,
) -> np.ndarray:
    """Return tail-padded x values for evaluating the Bnet KDE."""

    sample_standard_deviation = float(np.std(sample_values, ddof=0))

    if sample_standard_deviation <= 0.0:
        raise ValueError(
            "Cannot calculate KDE because Bnet-site values have zero variance."
        )

    scott_factor = float(gaussian_kde(dataset=sample_values).scotts_factor())
    tail_width = scott_factor * sample_standard_deviation

    x_min = float(np.min(sample_values) - (DEFAULT_KDE_TAIL_WIDTHS * tail_width))
    x_max = float(np.max(sample_values) + (DEFAULT_KDE_TAIL_WIDTHS * tail_width))

    return np.linspace(x_min, x_max, trapezium_count + 1)


def _split_trapezium_areas(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    split_value: float,
) -> tuple[float, float]:
    """Split trapezoidal integration area around a threshold.

    A trapezium is below the split value when its left x value is below the
    split. This includes the split-containing trapezium in the left area,
    except when the split falls exactly on that trapezium's left edge.
    """

    left_area = 0.0
    right_area = 0.0

    for left_x, right_x, left_y, right_y in zip(
        x_values[:-1],
        x_values[1:],
        y_values[:-1],
        y_values[1:],
        strict=True,
    ):
        trapezium_area = float((right_x - left_x) * (left_y + right_y) / 2.0)

        if left_x < split_value:
            left_area += trapezium_area
        else:
            right_area += trapezium_area

    return left_area, right_area
