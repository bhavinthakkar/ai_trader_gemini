"""
Unit Test Suite for Formalized Signal Strength & Confidence Engine (Step 5.5).

Verifies:
1. Score Decisiveness D = |S - 50| / 50:
   - Boundary conditions (0, 50, 100)
   - Symmetry: D(50 + x) == D(50 - x)
   - Clamping to [0.0, 1.0] and non-numeric fallbacks
2. Pillar Agreement A & Dispersion:
   - Perfect consensus (dispersion = 0, A = 1.0)
   - Contradictory dispersion (dispersion = 1.0, A = 0.0)
   - Metrics: MAD, Standard Deviation, Range
   - Sparse pillar handling (K < 3 flagged)
3. Data Completeness C:
   - Percentage of required inputs that are available, valid, and fresh
   - Stale inputs excluded from numerator
   - Weighted input records vs unweighted
   - Direct scalar clamping
4. Combination Normalization & Weights:
   - Hierarchical Gated model (production standard)
   - Direct Linear Weighted model
   - Geometric Multiplicative model
   - HOLD capping at 0.60
5. Epistemic Labeling & Calibration Safety:
   - Labeled 'SIGNAL_STRENGTH', NOT 'probability of profit'
   - is_calibrated_probability is False
   - Calibration status tracked as UNCALIBRATED_HEURISTIC_INDEX
"""

import math
import unittest
from quant.signal_strength import (
    DispersionMetric,
    CombinationMethod,
    SignalStrengthWeights,
    RequiredInputRecord,
    SignalStrengthResult,
    compute_score_decisiveness,
    compute_pillar_agreement,
    compute_data_completeness,
    SignalStrengthEngine,
    calculate_signal_strength,
)


class TestSignalStrengthFormalization(unittest.TestCase):
    """Exhaustive tests for Step 5.5 Signal Strength formalization."""

    def setUp(self):
        self.weights = SignalStrengthWeights()
        self.engine = SignalStrengthEngine(self.weights)

    # -------------------------------------------------------------------------
    # 1. Score Decisiveness (D)
    # -------------------------------------------------------------------------
    def test_score_decisiveness_neutral_boundary(self):
        """Score 50.0 represents zero distance from neutral (D = 0.0)."""
        d = compute_score_decisiveness(50.0)
        self.assertEqual(d, 0.0)

    def test_score_decisiveness_graduated_values(self):
        """D measures normalized distance from neutral anchor 50.0."""
        # S = 60 -> |60 - 50| / 50 = 0.20
        self.assertAlmostEqual(compute_score_decisiveness(60.0), 0.20, places=4)
        # S = 70 -> |70 - 50| / 50 = 0.40
        self.assertAlmostEqual(compute_score_decisiveness(70.0), 0.40, places=4)
        # S = 85 -> |85 - 50| / 50 = 0.70
        self.assertAlmostEqual(compute_score_decisiveness(85.0), 0.70, places=4)
        # S = 100 -> |100 - 50| / 50 = 1.00
        self.assertAlmostEqual(compute_score_decisiveness(100.0), 1.00, places=4)
        # S = 0 -> |0 - 50| / 50 = 1.00
        self.assertAlmostEqual(compute_score_decisiveness(0.0), 1.00, places=4)

    def test_score_decisiveness_symmetry(self):
        """D is strictly symmetric around neutral 50.0: D(50 + x) == D(50 - x)."""
        for delta in [5.0, 10.0, 20.0, 35.0, 50.0]:
            d_bull = compute_score_decisiveness(50.0 + delta)
            d_bear = compute_score_decisiveness(50.0 - delta)
            self.assertAlmostEqual(d_bull, d_bear, places=4)
            self.assertAlmostEqual(d_bull, delta / 50.0, places=4)

    def test_score_decisiveness_clamping_and_fallbacks(self):
        """Out-of-bound scores clamp to [0, 1]; invalid/None fallback to 0.0."""
        self.assertEqual(compute_score_decisiveness(150.0), 1.0)
        self.assertEqual(compute_score_decisiveness(-25.0), 1.0)
        self.assertEqual(compute_score_decisiveness(None), 0.0)
        self.assertEqual(compute_score_decisiveness("invalid"), 0.0)

    # -------------------------------------------------------------------------
    # 2. Pillar Agreement (A) & Dispersion
    # -------------------------------------------------------------------------
    def test_pillar_agreement_perfect_consensus(self):
        """When all pillars agree identically, dispersion is 0 and agreement is 1.0."""
        pillars = {
            "trend": 80.0,
            "sector": 80.0,
            "alpha": 80.0,
            "valuation_history": 80.0,
            "peer_valuation": 80.0,
        }
        agreement, dispersion, count, mean = compute_pillar_agreement(pillars)
        self.assertEqual(count, 5)
        self.assertEqual(mean, 80.0)
        self.assertEqual(dispersion, 0.0)
        self.assertEqual(agreement, 1.0)

    def test_pillar_agreement_moderate_dispersion(self):
        """Moderate divergence yields realistic, normalized agreement."""
        # Mean = (70 + 75 + 80) / 3 = 75.0
        # MAD = (|70-75| + |75-75| + |80-75|) / 3 = (5 + 0 + 5) / 3 = 3.3333
        # Dispersion = 3.3333 / 50.0 = 0.0667
        # Agreement = 1.0 - 0.0667 = 0.9333
        pillars = {"trend": 70.0, "sector": 75.0, "alpha": 80.0}
        agreement, dispersion, count, mean = compute_pillar_agreement(pillars, dispersion_metric=DispersionMetric.MAD)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(mean, 75.0, places=4)
        self.assertAlmostEqual(dispersion, 10.0 / 150.0, places=4)
        self.assertAlmostEqual(agreement, 1.0 - (10.0 / 150.0), places=4)

    def test_pillar_agreement_severe_dispersion(self):
        """Maximal divergence (contradicting pillars: two at 0, two at 100) yields A = 0.0."""
        # Scores: [0, 0, 100, 100]
        # Mean = 50.0, MAD = (50 + 50 + 50 + 50) / 4 = 50.0
        # Dispersion = 50.0 / 50.0 = 1.0 -> Agreement = 0.0
        pillars = {"trend": 0.0, "sector": 0.0, "alpha": 100.0, "valuation_history": 100.0}
        agreement, dispersion, count, mean = compute_pillar_agreement(pillars, dispersion_metric=DispersionMetric.MAD)
        self.assertEqual(count, 4)
        self.assertEqual(mean, 50.0)
        self.assertEqual(dispersion, 1.0)
        self.assertEqual(agreement, 0.0)

    def test_pillar_agreement_dispersion_metric_options(self):
        """Verifies standard deviation and range dispersion alternatives."""
        pillars = {"trend": 60.0, "sector": 70.0, "alpha": 80.0}
        # Standard deviation: mean = 70, var = (100 + 0 + 100)/3 = 66.67, std = 8.165
        a_std, d_std, _, _ = compute_pillar_agreement(pillars, dispersion_metric=DispersionMetric.STANDARD_DEVIATION)
        self.assertAlmostEqual(d_std, math.sqrt(200.0 / 3.0) / 50.0, places=3)
        self.assertAlmostEqual(a_std, 1.0 - (math.sqrt(200.0 / 3.0) / 50.0), places=3)

        # Range: (80 - 60) / 2 = 10, normalized / 50 = 0.20
        a_range, d_range, _, _ = compute_pillar_agreement(pillars, dispersion_metric=DispersionMetric.RANGE)
        self.assertAlmostEqual(d_range, 0.20, places=3)
        self.assertAlmostEqual(a_range, 0.80, places=3)

    def test_pillar_agreement_sparse_pillars(self):
        """Single or empty pillar sets default to neutral 0.50 agreement."""
        a_zero, d_zero, c_zero, _ = compute_pillar_agreement({})
        self.assertEqual(c_zero, 0)
        self.assertEqual(a_zero, 0.50)

        a_one, d_one, c_one, _ = compute_pillar_agreement({"trend": 75.0})
        self.assertEqual(c_one, 1)
        self.assertEqual(a_one, 0.50)

    # -------------------------------------------------------------------------
    # 3. Data Completeness (C)
    # -------------------------------------------------------------------------
    def test_data_completeness_from_fresh_input_records(self):
        """Completeness equals percentage of required inputs that are available, valid, and fresh."""
        records = [
            RequiredInputRecord(name="price", available=True, valid=True, fresh=True),
            RequiredInputRecord(name="atr", available=True, valid=True, fresh=True),
            RequiredInputRecord(name="rsi", available=True, valid=True, fresh=True),
            RequiredInputRecord(name="ml_score", available=True, valid=True, fresh=False),  # Stale!
        ]
        comp, audit = compute_data_completeness(input_records=records)
        # 3 out of 4 are valid and fresh -> 75%
        self.assertEqual(comp, 0.75)
        self.assertEqual(len(audit), 4)
        self.assertFalse(audit[3]["is_valid_and_fresh"])

    def test_data_completeness_weighted_inputs(self):
        """Completeness accounts for importance weights across inputs."""
        records = [
            RequiredInputRecord(name="technical_snapshot", available=True, valid=True, fresh=True, weight=0.50),
            RequiredInputRecord(name="macro_data", available=True, valid=True, fresh=True, weight=0.25),
            RequiredInputRecord(name="sec_filings", available=True, valid=False, fresh=True, weight=0.25),  # Invalid!
        ]
        comp, _ = compute_data_completeness(input_records=records)
        # 0.50 + 0.25 = 0.75 / 1.00 = 75%
        self.assertEqual(comp, 0.75)

    def test_data_completeness_direct_scalar_clamping(self):
        """Direct scalar completeness values clamp to [0.0, 1.0]."""
        self.assertEqual(compute_data_completeness(direct_completeness=0.85)[0], 0.85)
        self.assertEqual(compute_data_completeness(direct_completeness=1.25)[0], 1.0)
        self.assertEqual(compute_data_completeness(direct_completeness=-0.20)[0], 0.0)
        self.assertEqual(compute_data_completeness(direct_completeness=None)[0], 0.0)

    # -------------------------------------------------------------------------
    # 4. Combination Normalization & Weights
    # -------------------------------------------------------------------------
    def test_combination_hierarchical_gated_exact_math(self):
        """
        Verifies exact calculation under production Hierarchical Gated model:
        S = 80.0 -> D = 0.60
        All pillars 80.0 -> Dispersion = 0, Agreement = 1.0
        Completeness = 1.0
        Edge = 0.70 * 0.60 + 0.30 * (1.0 - 0.50) = 0.42 + 0.15 = 0.57
        Base = 0.50 + 0.50 * 0.57 = 0.785
        Conf = 0.785 * (0.50 + 0.50 * 1.0) = 0.785 -> rounds to 0.78
        """
        pillars = {k: 80.0 for k in ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]}
        res = self.engine.calculate(
            composite_score=80.0,
            pillar_scores=pillars,
            data_completeness=1.0,
            decision="BUY",
            method=CombinationMethod.HIERARCHICAL_GATED,
        )
        self.assertEqual(res.score_decisiveness, 0.60)
        self.assertEqual(res.pillar_agreement, 1.0)
        self.assertEqual(res.data_completeness, 1.0)
        self.assertEqual(res.signal_strength, 0.78)
        self.assertEqual(res.raw_signal_strength, 0.785)

    def test_combination_linear_weighted_exact_math(self):
        """
        Verifies Direct Linear Weighted model:
        w_D = 0.50, w_A = 0.25, w_C = 0.25
        D = 0.60, A = 1.0, C = 1.0
        Signal = 0.50 * 0.60 + 0.25 * 1.0 + 0.25 * 1.0 = 0.30 + 0.25 + 0.25 = 0.80
        """
        pillars = {k: 80.0 for k in ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]}
        res = self.engine.calculate(
            composite_score=80.0,
            pillar_scores=pillars,
            data_completeness=1.0,
            decision="BUY",
            method=CombinationMethod.LINEAR_WEIGHTED,
        )
        self.assertEqual(res.signal_strength, 0.80)
        self.assertEqual(res.combination_method, CombinationMethod.LINEAR_WEIGHTED)

    def test_combination_geometric_multiplicative(self):
        """
        Verifies Geometric Multiplicative model:
        D^0.5 * A^0.25 * C^0.25
        """
        pillars = {k: 80.0 for k in ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]}
        res = self.engine.calculate(
            composite_score=80.0,
            pillar_scores=pillars,
            data_completeness=1.0,
            decision="BUY",
            method=CombinationMethod.GEOMETRIC,
        )
        # 0.60^0.5 * 1.0^0.25 * 1.0^0.25 = sqrt(0.60) ≈ 0.7746 -> rounds to 0.77
        expected = round(math.sqrt(0.60), 2)
        self.assertEqual(res.signal_strength, expected)

    def test_neutral_hold_decision_capping(self):
        """Neutral HOLD decisions are capped at 0.60 regardless of high underlying score."""
        pillars = {k: 90.0 for k in ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]}
        res = self.engine.calculate(
            composite_score=90.0,
            pillar_scores=pillars,
            data_completeness=1.0,
            decision="HOLD",
        )
        self.assertEqual(res.signal_strength, 0.60)

    # -------------------------------------------------------------------------
    # 5. Epistemic Labeling & Calibration Safeguards
    # -------------------------------------------------------------------------
    def test_epistemic_labeling_and_calibration_distinction(self):
        """
        Step 5.5 Core Invariant:
        Result must be explicitly designated 'Signal Strength', NOT 'probability of profit'.
        is_calibrated_probability must be False until calibrated against realized return distributions.
        """
        res = calculate_signal_strength(
            composite_score=75.0,
            pillar_scores={"trend": 75.0, "sector": 70.0, "alpha": 72.0},
            data_completeness=0.90,
            decision="BUY",
        )
        self.assertEqual(res.label, "SIGNAL_STRENGTH")
        self.assertFalse(res.is_calibrated_probability)
        self.assertEqual(res.calibration_status, "UNCALIBRATED_HEURISTIC_INDEX")
        dict_rep = res.to_dict()
        self.assertEqual(dict_rep["label"], "SIGNAL_STRENGTH")
        self.assertFalse(dict_rep["is_calibrated_probability"])


if __name__ == "__main__":
    unittest.main()
