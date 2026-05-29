import unittest

from database.eligibility import (
    BnetEligibilityContext,
    BnetEligibilityReason,
    check_bnet_percentile_eligibility,
)


def make_eligible_context() -> BnetEligibilityContext:
    return BnetEligibilityContext(
        resolution_angstrom=1.5,
        r_free=0.2,
        temperature_k=100.0,
        asp_glu_carboxyl_oxygen_count=40,
        has_asp_glu_residue_with_total_occupancy_below_one=False,
        uses_per_atom_b_factors=True,
        bnet=1.5,
    )


class BnetEligibilityTests(unittest.TestCase):
    def test_accepts_eligible_context(self) -> None:
        result = check_bnet_percentile_eligibility(make_eligible_context())

        self.assertTrue(result.is_eligible)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.primary_reason, BnetEligibilityReason.ELIGIBLE)

    def test_reports_all_relevant_failure_reasons(self) -> None:
        context = BnetEligibilityContext(
            resolution_angstrom=4.0,
            r_free=0.45,
            temperature_k=200.0,
            asp_glu_carboxyl_oxygen_count=10,
            has_asp_glu_residue_with_total_occupancy_below_one=True,
            uses_per_atom_b_factors=False,
            bnet=None,
        )

        result = check_bnet_percentile_eligibility(context)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            tuple(issue.reason for issue in result.issues),
            (
                BnetEligibilityReason.RESOLUTION_TOO_LOW,
                BnetEligibilityReason.RFREE_TOO_HIGH,
                BnetEligibilityReason.TEMPERATURE_OUTSIDE_CRYO_RANGE,
                BnetEligibilityReason.TOO_FEW_ASP_GLU_CARBOXYL_OXYGENS,
                BnetEligibilityReason.ASP_GLU_OCCUPANCY_LESS_THAN_ONE,
                BnetEligibilityReason.NOT_PER_ATOM_B_FACTOR_MODEL,
                BnetEligibilityReason.MISSING_BNET,
            ),
        )

    def test_rejects_non_finite_and_non_numeric_values(self) -> None:
        context = BnetEligibilityContext(
            resolution_angstrom=object(),  # type: ignore[arg-type]
            r_free=True,  # type: ignore[arg-type]
            temperature_k=float("nan"),
            asp_glu_carboxyl_oxygen_count=40,
            has_asp_glu_residue_with_total_occupancy_below_one=False,
            uses_per_atom_b_factors=True,
            bnet=object(),  # type: ignore[arg-type]
        )

        result = check_bnet_percentile_eligibility(context)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            tuple(issue.reason for issue in result.issues),
            (
                BnetEligibilityReason.INVALID_RESOLUTION,
                BnetEligibilityReason.INVALID_RFREE,
                BnetEligibilityReason.INVALID_TEMPERATURE,
                BnetEligibilityReason.INVALID_BNET,
            ),
        )

    def test_bnet_can_be_optional(self) -> None:
        context = make_eligible_context()
        context_without_bnet = BnetEligibilityContext(
            resolution_angstrom=context.resolution_angstrom,
            r_free=context.r_free,
            temperature_k=context.temperature_k,
            asp_glu_carboxyl_oxygen_count=context.asp_glu_carboxyl_oxygen_count,
            has_asp_glu_residue_with_total_occupancy_below_one=(
                context.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            uses_per_atom_b_factors=context.uses_per_atom_b_factors,
            bnet=None,
        )

        result = check_bnet_percentile_eligibility(
            context_without_bnet,
            require_bnet=False,
        )

        self.assertTrue(result.is_eligible)
