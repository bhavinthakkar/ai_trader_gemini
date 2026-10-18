"""Unit and Integration Tests for Portfolio-Level Controls & Concentration Management (Step 7.2).

Validates:
1. Maximum Single-Stock Exposure:
   - Single-stock cap bounds allocation to avoid single-company catastrophe.
   - Incremental adds to existing positions respect cumulative limit.
2. Sector Concentration Limit:
   - Enforces maximum aggregate exposure in any single GICS sector (e.g. 25% in Technology).
   - Throttles or rejects candidate trades when sector budget is depleted.
3. Correlated-Factor Exposure Limit (Focused 25-Stock Universe Analysis):
   - Monitors combined exposures across positions driven by similar alpha factors
     (e.g., AI semiconductors: NVDA, AMD, AVGO; mega-cap growth: MSFT, AAPL, GOOGL).
   - Prevents the portfolio from becoming an unbalanced single-factor bet.
4. Total Gross and Net Exposure Limits:
   - Gross exposure capped at max_gross_exposure_pct (e.g. 100% unleveraged).
   - Net exposure (|Long - Short| / Equity) capped at max_net_exposure_pct (e.g. 80%).
5. Cash Reserve & Short Margin Collateral Requirements:
   - Maintains unencumbered cash buffer (e.g. 10%).
   - Verifies short margin collateral requirements do not invade the cash reserve.
6. Portfolio Drawdown Governor & Circuit Breakers:
   - NORMAL: DD < 5% -> full risk budget.
   - DE_RISKING: 5% <= DD < 10% -> risk fraction r dampened by 50%.
   - CIRCUIT_BREAKER_HALT: DD >= 10% -> 0 shares, freezing all new risk entries.
7. Portfolio Liquidity & Liquidation Horizon:
   - Evaluates weighted portfolio liquidation days at 1.0% ADV participation.
8. Short Position Structural Eligibility & Financing Validator:
   - Low composite score does NOT itself justify opening a short.
   - Rejects shorts without structural trend breakdown (price > EMA20 or EMA50).
   - Rejects shorts into exhausted oversold climax (RSI < 25).
   - Rejects shorts on fortress-quality, historically cheap stocks (value trap short squeeze).
   - Rejects shorts with RR < 1.50.
   - Rejects shorts when borrow is unavailable or HTB without locate.
   - Rejects shorts with prohibitive borrow fees (> 500 bps).
   - Rejects shorts within ex-dividend blackout window (<= 3 sessions).
   - Successfully certifies valid short when all structural and financing checks pass.
9. End-to-End Integration with `calculate_risk_based_position_size`.
"""

import math
import unittest

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import (
    DrawdownRegime,
    PortfolioBindingConstraint,
    DEFAULT_UNIVERSE_TAXONOMY,
    resolve_instrument_taxonomy,
    Position,
    PortfolioSnapshot,
    ShortEligibilityResult,
    ShortEligibilityValidator,
    PortfolioComplianceReport,
    PortfolioRiskEngine,
)
from quant.position_sizing import (
    BindingConstraint,
    InvestmentMandate,
    PositionSizingResult,
    calculate_risk_based_position_size,
)


class TestPortfolioConcentrationControls(unittest.TestCase):
    """Verifies single-stock, sector, and correlated-factor concentration limits."""

    def setUp(self):
        self.policy = get_default_policy()
        # Custom mandate: Equity $100k, Single-stock 10% ($10k), Sector 25% ($25k), Factor 30% ($30k)
        self.equity = 100_000.0
        self.snapshot = PortfolioSnapshot(
            portfolio_equity=self.equity,
            cash=30_000.0,
            peak_equity=self.equity,
            positions={
                "NVDA": Position(
                    ticker="NVDA",
                    direction="LONG",
                    shares=80,
                    entry_price=100.0,
                    current_price=100.0,  # $8,000 (8% of equity)
                    sector="Technology",
                    factor_tags=["AI_SEMICONDUCTORS", "HIGH_BETA_MOMENTUM"],
                    adv_shares_20d=50_000_000.0,
                ),
                "MSFT": Position(
                    ticker="MSFT",
                    direction="LONG",
                    shares=100,
                    entry_price=100.0,
                    current_price=100.0,  # $10,000 (10% of equity)
                    sector="Technology",
                    factor_tags=["MEGA_CAP_GROWTH", "CLOUD"],
                    adv_shares_20d=25_000_000.0,
                ),
            },
        )

    def test_single_stock_exposure_cap(self):
        """Single-stock cap limits allocation to max_single_stock_pct (10% = $10,000)."""
        # Candidate JPM (Financials - $0 existing in snapshot). Risk budget = $1,000 / $5.12 = 195 shares.
        # But Single Stock Cap = $10,000 / $100 = 100 shares.
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="JPM",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=self.snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 100)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.SINGLE_STOCK_CAP.value)
        self.assertEqual(report.n_single_stock_cap, 100)

    def test_single_stock_incremental_add_respects_remaining_headroom(self):
        """Adding to an existing position respects remaining headroom (NVDA has $8k existing)."""
        # NVDA already has $8,000. Max 10% = $10,000. Headroom = $2,000.
        # At $100/share, only 20 shares allowed.
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="NVDA",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=self.snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 20)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.SINGLE_STOCK_CAP.value)
        self.assertEqual(report.single_stock_headroom_dollars, 2000.0)

    def test_sector_concentration_cap_throttles_shares(self):
        """Sector concentration limit (25% = $25,000). Tech already has NVDA($8k) + MSFT($10k) = $18k."""
        # Remaining Tech headroom = $25,000 - $18,000 = $7,000.
        # Candidate QCOM (Sector Technology) at $100/share:
        # N_risk = 163, N_single_stock = 100, N_sector = floor(7000 / 100) = 70.
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="QCOM",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=self.snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 70)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.SECTOR_CAP.value)
        self.assertEqual(report.sector_headroom_dollars, 7000.0)

    def test_sector_concentration_zero_when_budget_exhausted(self):
        """Sector cap rejects trade with 0 shares when sector budget is completely exhausted."""
        full_tech_snapshot = PortfolioSnapshot(
            portfolio_equity=self.equity,
            cash=20_000.0,
            peak_equity=self.equity,
            positions={
                "NVDA": Position("NVDA", "LONG", 100, 100.0, 100.0, sector="Technology"),  # $10k
                "MSFT": Position("MSFT", "LONG", 100, 100.0, 100.0, sector="Technology"),  # $10k
                "AAPL": Position("AAPL", "LONG", 50, 100.0, 100.0, sector="Technology"),   # $5k -> Total $25k (25%)
            },
        )
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="AMD",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=full_tech_snapshot,
            policy=self.policy,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.allowed_shares, 0)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.SECTOR_CAP.value)
        self.assertEqual(report.sector_headroom_dollars, 0.0)

    def test_25_stock_universe_correlated_factor_cluster_cap(self):
        """
        Monitors combined exposure of positions driven by similar factors in the 25-stock universe.
        Policy max_correlated_factor_pct = 30% ($30,000).
        Portfolio already holds NVDA ($10,000) and AVGO ($10,000) in AI_SEMICONDUCTORS ($20,000).
        Adding AMD (also in AI_SEMICONDUCTORS) is capped at remaining factor budget:
        $30,000 - $20,000 = $10,000 headroom -> max 100 shares.
        If portfolio holds $28,000 in AI_SEMICONDUCTORS, headroom is only $2,000 -> 20 shares!
        """
        semi_heavy_snapshot = PortfolioSnapshot(
            portfolio_equity=self.equity,
            cash=50_000.0,
            peak_equity=self.equity,
            positions={
                "NVDA": Position(
                    "NVDA", "LONG", 140, 100.0, 100.0,
                    sector="Technology", factor_tags=["AI_SEMICONDUCTORS", "AI_TECH"]
                ),  # $14,000
                "AVGO": Position(
                    "AVGO", "LONG", 140, 100.0, 100.0,
                    sector="Technology", factor_tags=["AI_SEMICONDUCTORS", "AI_TECH"]
                ),  # $14,000 -> Total AI_SEMICONDUCTORS = $28,000
            },
        )
        # Sector Technology has $28,000 (if sector limit were 40%, sector allows $12,000).
        custom_policy = get_default_policy()
        custom_policy.portfolio_risk.max_sector_pct = 0.50  # Relax sector limit to isolate factor limit
        custom_policy.portfolio_risk.max_correlated_factor_pct = 0.30  # Max 30% in any factor cluster ($30k)

        # Candidate AMD (Price $100, Stop $95).
        # AI_SEMICONDUCTORS has $28k deployed. Factor headroom = $30,000 - $28,000 = $2,000.
        # N_factor_cap = floor(2000 / 100) = 20 shares.
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="AMD",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=semi_heavy_snapshot,
            policy=custom_policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 20)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.FACTOR_CAP.value)
        self.assertEqual(report.factor_headroom_dollars, 2000.0)


class TestTotalExposureAndCashReserve(unittest.TestCase):
    """Verifies Gross/Net exposure and Cash Reserve / Collateral rules."""

    def setUp(self):
        self.policy = get_default_policy()
        self.equity = 100_000.0

    def test_gross_exposure_cap(self):
        """Gross exposure (|Long| + |Short|) cannot exceed max_gross_exposure_pct (100% = $100k)."""
        snapshot = PortfolioSnapshot(
            portfolio_equity=self.equity,
            cash=25_000.0,
            peak_equity=self.equity,
            positions={
                "MSFT": Position("MSFT", "LONG", 550, 100.0, 100.0, sector="Technology"),   # Long $55k
                "XOM": Position("XOM", "SHORT", 400, 100.0, 100.0, sector="Energy"),        # Short $40k -> Gross $95k, Net $15k
            },
        )
        # Gross deployed = $95,000. Remaining gross headroom = $5,000.
        # Candidate JPM at $100: N_gross_cap = 50 shares.
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="JPM",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 50)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.GROSS_EXPOSURE_CAP.value)
        self.assertEqual(report.gross_headroom_dollars, 5000.0)

    def test_net_exposure_cap(self):
        """Net exposure (|Long - Short| / Equity) cannot exceed max_net_exposure_pct (80% = $80k)."""
        snapshot = PortfolioSnapshot(
            portfolio_equity=self.equity,
            cash=25_000.0,
            peak_equity=self.equity,
            positions={
                "MSFT": Position("MSFT", "LONG", 400, 100.0, 100.0, sector="Technology"),  # Long $40k
                "NVDA": Position("NVDA", "LONG", 350, 100.0, 100.0, sector="Technology"),  # Long $35k -> Total Net Long $75k
            },
        )
        # Net long = $75,000. Headroom to 80% ($80,000) is $5,000 -> 50 shares at $100.
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="LLY",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 50)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.NET_EXPOSURE_CAP.value)

    def test_cash_reserve_requirement_protects_buffer(self):
        """Cash reserve requirement (10% = $10,000) prevents deploying unencumbered cash buffer."""
        # Equity = $100k, Cash = $12,000. Required cash reserve = $10,000.
        # Free cash for new long purchase = $12,000 - $10,000 = $2,000.
        # At $100/share, N_cash_cap = 20 shares.
        snapshot = PortfolioSnapshot(
            portfolio_equity=self.equity,
            cash=12_000.0,
            peak_equity=self.equity,
            positions={},
        )
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="AAPL",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.allowed_shares, 20)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.CASH_RESERVE_CAP.value)
        self.assertEqual(report.cash_headroom_dollars, 2000.0)


class TestDrawdownGovernorAndCircuitBreakers(unittest.TestCase):
    """Verifies multi-tier drawdown management: Normal, De-risking, and Circuit Breaker Halt."""

    def setUp(self):
        self.policy = get_default_policy()
        self.peak_equity = 100_000.0

    def test_normal_drawdown_regime(self):
        """DD < 5% operates under NORMAL regime with full risk budget."""
        # Equity $97,000 -> DD = 3.0% < 5.0%
        snapshot = PortfolioSnapshot(
            portfolio_equity=97_000.0,
            cash=50_000.0,
            peak_equity=self.peak_equity,
            positions={},
        )
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="AAPL",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertEqual(report.drawdown_regime, DrawdownRegime.NORMAL.value)
        self.assertEqual(report.effective_risk_fraction, 0.010)
        self.assertTrue(report.passed)

    def test_de_risking_regime_dampens_risk_budget(self):
        """5% <= DD < 10% operates under DE_RISKING regime, scaling risk budget fraction r by 0.50x."""
        # Equity $93,000 -> DD = 7.0% in [5%, 10%)
        # Normal r = 0.010 -> Dampened r = 0.005 (0.50%)
        # Risk dollars = $93,000 * 0.005 = $465.
        # Per-share risk = 5.0 + 0.12 + 1.0 = 6.12.
        # N_risk = floor(465 / 6.12) = 75 shares (vs 151 normal).
        snapshot = PortfolioSnapshot(
            portfolio_equity=93_000.0,
            cash=50_000.0,
            peak_equity=self.peak_equity,
            positions={},
        )
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="AAPL",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=snapshot,
            policy=self.policy,
            cost_per_share=0.12,
            gap_risk_reserve=1.0,
        )
        self.assertEqual(report.drawdown_regime, DrawdownRegime.DE_RISKING.value)
        self.assertEqual(report.effective_risk_fraction, 0.005)
        self.assertEqual(report.allowed_shares, 75)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.RISK_BUDGET.value)

    def test_circuit_breaker_halt_freezes_all_new_entries(self):
        """DD >= 10% triggers CIRCUIT_BREAKER_HALT, freezing all new risk entries (0 shares)."""
        # Equity $89,000 -> DD = 11.0% >= 10.0%
        snapshot = PortfolioSnapshot(
            portfolio_equity=89_000.0,
            cash=50_000.0,
            peak_equity=self.peak_equity,
            positions={},
        )
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker="AAPL",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            snapshot=snapshot,
            policy=self.policy,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.allowed_shares, 0)
        self.assertEqual(report.binding_constraint, PortfolioBindingConstraint.DRAWDOWN_CIRCUIT_BREAKER.value)
        self.assertEqual(report.drawdown_regime, DrawdownRegime.CIRCUIT_BREAKER_HALT.value)
        self.assertIn("circuit breaker halt", report.rejection_reason.lower())


class TestShortPositionEligibilityAndFinancing(unittest.TestCase):
    """
    Verifies that low composite score does NOT itself justify opening a short.
    Tests structural breakdown, borrow availability, financing fee, and margin collateral.
    """

    def setUp(self):
        self.policy = get_default_policy()

    def test_short_rejected_without_structural_trend_breakdown(self):
        """A low score (35.0) on a stock trading above EMA20 is rejected: trend not broken."""
        res = ShortEligibilityValidator.validate(
            ticker="INTC",
            entry_price=30.0,
            stop_loss=33.0,
            composite_score=35.0,  # Below 45 ceiling
            policy=self.policy,
            ema_20=28.0,  # Price $30 is ABOVE 20 EMA ($28) -> not in structural breakdown
            ema_50=27.0,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason_code, "SHORT_TREND_NOT_BROKEN")
        self.assertIn("structural technical breakdown", res.message)

    def test_short_rejected_when_oversold_exhaustion_active(self):
        """Shorting into capitulation panic (RSI < 25) is rejected as mean-reversion squeeze risk."""
        res = ShortEligibilityValidator.validate(
            ticker="INTC",
            entry_price=25.0,
            stop_loss=28.0,
            composite_score=30.0,
            policy=self.policy,
            ema_20=27.0,
            ema_50=29.0,
            rsi=21.5,  # Extreme oversold exhaustion
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason_code, "SHORT_OVERSOLD_SQUEEZE_RISK")
        self.assertIn("squeeze risk", res.message)

    def test_short_rejected_on_fortress_quality_depressed_value(self):
        """Shorting fortress balance sheet with depressed valuation history is forbidden (squeeze trap)."""
        res = ShortEligibilityValidator.validate(
            ticker="000660.KS",  # SK Hynix: fortress quality, cheap valuation
            entry_price=100.0,
            stop_loss=110.0,
            composite_score=32.0,
            policy=self.policy,
            ema_20=105.0,
            ema_50=110.0,
            rsi=35.0,
            financial_quality_score=95.0,   # Fortress balance sheet
            valuation_history_score=85.0,   # Historically cheap multiples
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason_code, "SHORT_VALUE_TRAP_SQUEEZE_RISK")
        self.assertIn("fortress quality", res.message)

    def test_short_rejected_when_borrow_unavailable(self):
        """Borrow unavailable or pool empty rejects short recommendation."""
        res = ShortEligibilityValidator.validate(
            ticker="HARD_STOCK",
            entry_price=50.0,
            stop_loss=55.0,
            composite_score=30.0,
            policy=self.policy,
            ema_20=52.0,
            ema_50=54.0,
            rsi=35.0,
            borrow_available=False,
            borrow_status="UNAVAILABLE",
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason_code, "BORROW_UNAVAILABLE")

    def test_short_rejected_when_borrow_fee_is_prohibitive(self):
        """Borrow fee > 500 bps (e.g. 800 bps) makes trade economically unviable."""
        res = ShortEligibilityValidator.validate(
            ticker="EXPENSIVE_BORROW",
            entry_price=50.0,
            stop_loss=55.0,
            composite_score=30.0,
            policy=self.policy,
            ema_20=52.0,
            ema_50=54.0,
            rsi=35.0,
            borrow_fee_bps=800.0,  # 8.0% annualized carry drag
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason_code, "BORROW_FEE_PROHIBITIVE")
        self.assertIn("exceeds maximum permitted", res.message)

    def test_short_rejected_during_ex_dividend_blackout(self):
        """Short seller liable for dividend payment within 3 sessions of ex-dividend date."""
        res = ShortEligibilityValidator.validate(
            ticker="DIV_PAYER",
            entry_price=50.0,
            stop_loss=55.0,
            composite_score=30.0,
            policy=self.policy,
            ema_20=52.0,
            ema_50=54.0,
            rsi=35.0,
            days_to_ex_dividend=1,  # 1 day to ex-dividend
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason_code, "EX_DIVIDEND_BLACKOUT")

    def test_short_certified_when_all_structural_and_financing_criteria_pass(self):
        """Full structural breakdown + cheap borrow + adequate financing passes with certification."""
        res = ShortEligibilityValidator.validate(
            ticker="WEAK_TECH",
            entry_price=50.0,
            stop_loss=55.0,
            composite_score=32.0,
            policy=self.policy,
            ema_20=53.0,
            ema_50=56.0,
            rsi=35.0,
            financial_quality_score=40.0,
            valuation_history_score=30.0,
            reward_risk_ratio=1.80,
            borrow_available=True,
            borrow_status="ETB",
            borrow_fee_bps=45.0,  # Cheap 45 bps borrow
            days_to_ex_dividend=45,
            available_cash=50_000.0,
            portfolio_equity=100_000.0,
        )
        self.assertTrue(res.is_eligible)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(res.reason_code, "SHORT_ELIGIBILITY_VERIFIED")


class TestEndToEndPositionSizingIntegration(unittest.TestCase):
    """Verifies end-to-end integration between calculate_risk_based_position_size and portfolio controls."""

    def test_integration_with_portfolio_snapshot(self):
        """Passing PortfolioSnapshot into calculate_risk_based_position_size triggers portfolio controls."""
        equity = 100_000.0
        snapshot = PortfolioSnapshot(
            portfolio_equity=equity,
            cash=25_000.0,
            peak_equity=equity,
            positions={
                "NVDA": Position("NVDA", "LONG", 100, 100.0, 100.0, sector="Technology"),  # $10k
                "MSFT": Position("MSFT", "LONG", 100, 100.0, 100.0, sector="Technology"),  # $10k -> Total Tech $20k
            },
        )
        # Adding AMD (Tech sector headroom = $25k - $20k = $5,000 -> 50 shares at $100).
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            ticker="AMD",
            direction="LONG",
            portfolio_snapshot=snapshot,
        )
        self.assertTrue(res.is_tradeable)
        self.assertEqual(res.shares_allowed, 50)
        self.assertEqual(res.binding_constraint, BindingConstraint.SECTOR_CAP.value)
        self.assertIn("portfolio_report", res.metrics)

    def test_integration_short_rejected_via_position_sizing(self):
        """Short trade rejected via calculate_risk_based_position_size due to prohibitive borrow fee."""
        res = calculate_risk_based_position_size(
            entry_price=50.0,
            stop_loss=55.0,
            ticker="HARD_BORROW",
            direction="SHORT",
            composite_score=30.0,
            borrow_fee_bps=950.0,  # 950 bps > 500 bps limit
        )
        self.assertFalse(res.is_tradeable)
        self.assertEqual(res.shares_allowed, 0)
        self.assertEqual(res.binding_constraint, BindingConstraint.SHORT_FINANCING_PROHIBITIVE.value)
        self.assertIn("borrow fee", res.rejection_reason.lower())


if __name__ == "__main__":
    unittest.main()
