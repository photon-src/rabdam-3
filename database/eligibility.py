"""Eligibility checks for Bnet reference database workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import SupportsFloat, SupportsIndex, cast


DEFAULT_MIN_TEMPERATURE_K = 80.0
DEFAULT_MAX_TEMPERATURE_K = 120.0
DEFAULT_MAX_RESOLUTION_ANGSTROM = 3.5
DEFAULT_MAX_RFREE = 0.4
DEFAULT_MIN_ASP_GLU_CARBOXYL_OXYGEN_COUNT = 20

_Floatable = str | bytes | bytearray | SupportsFloat | SupportsIndex


class BnetEligibilityReason(str, Enum):
    """Machine-readable reasons a structure is not eligible."""

    ELIGIBLE = "eligible"
    MISSING_RESOLUTION = "missing_resolution"
    INVALID_RESOLUTION = "invalid_resolution"
    RESOLUTION_TOO_LOW = "resolution_too_low"

    MISSING_RFREE = "missing_rfree"
    INVALID_RFREE = "invalid_rfree"
    RFREE_TOO_HIGH = "rfree_too_high"

    MISSING_TEMPERATURE = "missing_temperature"
    INVALID_TEMPERATURE = "invalid_temperature"
    TEMPERATURE_OUTSIDE_CRYO_RANGE = "temperature_outside_cryo_range"

    TOO_FEW_ASP_GLU_CARBOXYL_OXYGENS = "too_few_asp_glu_carboxyl_oxygens"
    ASP_GLU_OCCUPANCY_LESS_THAN_ONE = "asp_glu_occupancy_less_than_one"

    NOT_PER_ATOM_B_FACTOR_MODEL = "not_per_atom_b_factor_model"

    MISSING_BNET = "missing_bnet"
    INVALID_BNET = "invalid_bnet"


@dataclass(frozen=True, slots=True)
class BnetEligibilityIssue:
    """One eligibility issue."""

    reason: BnetEligibilityReason
    message: str
    value: object | None = None


@dataclass(frozen=True, slots=True)
class BnetEligibilityResult:
    """Result of checking Bnet reference database eligibility."""

    is_eligible: bool
    issues: tuple[BnetEligibilityIssue, ...] = field(default_factory=tuple)

    @property
    def primary_reason(self) -> BnetEligibilityReason:
        """Return the first failure reason, or ELIGIBLE."""

        if not self.issues:
            return BnetEligibilityReason.ELIGIBLE
        return self.issues[0].reason

    @property
    def primary_message(self) -> str:
        """Return the first failure message, or a success message."""

        if not self.issues:
            return "Structure is eligible for Bnet reference database inclusion."
        return self.issues[0].message


@dataclass(frozen=True, slots=True)
class BnetEligibilityContext:
    """Inputs needed to assess Bnet reference database eligibility.

    This object deliberately avoids any PDB-REDO-specific fields. Database
    builders can construct it from PDB-REDO metadata, mmCIF metadata, or RABDAM
    workflow outputs.
    """

    resolution_angstrom: float | None
    r_free: float | None
    temperature_k: float | None
    asp_glu_carboxyl_oxygen_count: int
    has_asp_glu_residue_with_total_occupancy_below_one: bool
    uses_per_atom_b_factors: bool
    bnet: float | None = None


def check_bnet_reference_eligibility(
    context: BnetEligibilityContext,
    *,
    require_bnet: bool = True,
    min_temperature_k: float = DEFAULT_MIN_TEMPERATURE_K,
    max_temperature_k: float = DEFAULT_MAX_TEMPERATURE_K,
    max_resolution_angstrom: float = DEFAULT_MAX_RESOLUTION_ANGSTROM,
    max_r_free: float = DEFAULT_MAX_RFREE,
    min_asp_glu_carboxyl_oxygen_count: int = (
        DEFAULT_MIN_ASP_GLU_CARBOXYL_OXYGEN_COUNT
    ),
) -> BnetEligibilityResult:
    """Check whether a structure is eligible for Bnet reference database inclusion."""

    issues: list[BnetEligibilityIssue] = []

    resolution = context.resolution_angstrom
    if resolution is None:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.MISSING_RESOLUTION,
                "Resolution is missing.",
            )
        )
    elif not _is_finite_number(resolution) or resolution <= 0.0:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_RESOLUTION,
                "Resolution must be a finite positive number.",
                resolution,
            )
        )
    elif resolution > max_resolution_angstrom:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.RESOLUTION_TOO_LOW,
                (
                    f"Resolution is {resolution:.3g} Å, which is worse than the "
                    f"{max_resolution_angstrom:.3g} Å Bnet reference database limit."
                ),
                resolution,
            )
        )

    r_free = context.r_free
    if r_free is None:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.MISSING_RFREE,
                "Rfree is missing.",
            )
        )
    elif not _is_finite_number(r_free) or r_free < 0.0:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_RFREE,
                "Rfree must be a finite non-negative number.",
                r_free,
            )
        )
    elif r_free > max_r_free:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.RFREE_TOO_HIGH,
                (
                    f"Rfree is {r_free:.3g}, which is above the "
                    f"{max_r_free:.3g} Bnet reference database limit."
                ),
                r_free,
            )
        )

    temperature = context.temperature_k
    if temperature is None:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.MISSING_TEMPERATURE,
                "Collection temperature is missing.",
            )
        )
    elif not _is_finite_number(temperature) or temperature <= 0.0:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_TEMPERATURE,
                "Collection temperature must be a finite positive number.",
                temperature,
            )
        )
    elif not min_temperature_k <= temperature <= max_temperature_k:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.TEMPERATURE_OUTSIDE_CRYO_RANGE,
                (
                    f"Collection temperature is {temperature:.3g} K, outside the "
                    f"{min_temperature_k:.3g}–{max_temperature_k:.3g} K "
                    "Bnet reference database range."
                ),
                temperature,
            )
        )

    if context.asp_glu_carboxyl_oxygen_count < min_asp_glu_carboxyl_oxygen_count:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.TOO_FEW_ASP_GLU_CARBOXYL_OXYGENS,
                (
                    "Too few Asp/Glu side-chain carboxyl oxygen atoms for "
                    "Bnet reference database inclusion: "
                    f"{context.asp_glu_carboxyl_oxygen_count} found, "
                    f"{min_asp_glu_carboxyl_oxygen_count} required."
                ),
                context.asp_glu_carboxyl_oxygen_count,
            )
        )

    if context.has_asp_glu_residue_with_total_occupancy_below_one:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.ASP_GLU_OCCUPANCY_LESS_THAN_ONE,
                (
                    "At least one Asp/Glu residue has total side-chain occupancy "
                    "below one across listed conformers."
                ),
            )
        )

    if not context.uses_per_atom_b_factors:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.NOT_PER_ATOM_B_FACTOR_MODEL,
                "Structure does not appear to use a per-atom B-factor model.",
            )
        )

    if require_bnet:
        bnet = context.bnet
        if bnet is None:
            issues.append(
                BnetEligibilityIssue(
                    BnetEligibilityReason.MISSING_BNET,
                    "Raw Bnet value is missing.",
                )
            )
        elif not _is_finite_number(bnet) or bnet < 0.0:
            issues.append(
                BnetEligibilityIssue(
                    BnetEligibilityReason.INVALID_BNET,
                    "Raw Bnet must be a finite non-negative number.",
                    bnet,
                )
            )

    return BnetEligibilityResult(
        is_eligible=not issues,
        issues=tuple(issues),
    )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False

    if not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        return False

    try:
        number = float(cast(_Floatable, value))
    except (TypeError, ValueError):
        return False

    return math.isfinite(number)


__all__ = [
    "BnetEligibilityContext",
    "BnetEligibilityIssue",
    "BnetEligibilityReason",
    "BnetEligibilityResult",
    "DEFAULT_MAX_RESOLUTION_ANGSTROM",
    "DEFAULT_MAX_RFREE",
    "DEFAULT_MAX_TEMPERATURE_K",
    "DEFAULT_MIN_ASP_GLU_CARBOXYL_OXYGEN_COUNT",
    "DEFAULT_MIN_TEMPERATURE_K",
    "check_bnet_reference_eligibility",
]
