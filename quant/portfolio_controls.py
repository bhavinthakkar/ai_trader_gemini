"""Portfolio-Level Risk Controls & Concentration Management (Step 7.2).

Implements institutional portfolio-level risk controls for candidate trade certification:
1. Maximum Single-Stock Exposure:
   Limits single-instrument allocation to avoid single-company catastrophe.
2. Sector Concentration Limit:
   Limits aggregate exposure in any single GICS sector (e.g. max 25% in Technology).
3. Correlated-Factor Exposure Limit (Focused 25-Stock Universe):
   Monitors combined exposures across positions driven by similar alpha factors
   (e.g., AI semiconductors, mega-cap growth, high-beta cyclical).
4. Total Gross and Net Exposure Limits:
   Gross exposure = (|Long| + |Short|) / Equity <= max_gross_exposure_pct.
   Net exposure = |Long - Short| / Equity <= max_net_exposure_pct.
5. Cash Reserve & Short Margin Collateral:
   Maintains minimum unencumbered cash reserve buffer (e.g. 10%).
   Guarantees short margin requirements (Reg-T collateral) do not deplete cash reserves.
6. Portfolio Drawdown Governor & Circuit Breakers:
   Multi-tier drawdown management based on portfolio high-water mark:
   - NORMAL: DD < warning threshold (5%). Full risk budget.
   - DE_RISKING: warning (5%) <= DD < halt (10%). Scales down risk fraction r by dampener (0.5x).
   - CIRCUIT_BREAKER_HALT: DD >= halt (10%). Freezes all new risk entries (0 shares).
7. Portfolio-Level Liquidity & Liquidation Horizon:
   Weighted liquidation horizon at 1% ADV participation <= max_portfolio_liquidation_days.
8. Short Position Structural Eligibility & Financing Validator:
   "A low composite score does not itself justify opening a short."
   Enforces structural trend breakdown (price < EMA20 and < EMA50), oversold exhaustion guard
   (RSI >= 25), quality-value squeeze trap guard, borrow availability/locate, and borrow fee limits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from quant.policy_config import TradingPolicy, get_default_policy


# =============================================================================
# ENUMERATIONS & BINDING CONSTRAINTS
# =============================================================================

class DrawdownRegime(str, Enum):
    """Portfolio drawdown protection regime."""
    NORMAL = "NORMAL"
    DE_RISKING = "DE_RISKING"
    CIRCUIT_BREAKER_HALT = "CIRCUIT_BREAKER_HALT"


class PortfolioBindingConstraint(str, Enum):
    """Specific portfolio-level constraint limiting executable share size."""
    RISK_BUDGET = "RISK_BUDGET"
    SINGLE_STOCK_CAP = "SINGLE_STOCK_CAP"
    SECTOR_CAP = "SECTOR_CAP"
    FACTOR_CAP = "FACTOR_CAP"
    GROSS_EXPOSURE_CAP = "GROSS_EXPOSURE_CAP"
    NET_EXPOSURE_CAP = "NET_EXPOSURE_CAP"
    CASH_RESERVE_CAP = "CASH_RESERVE_CAP"
    LIQUIDITY_CAP = "LIQUIDITY_CAP"
    DRAWDOWN_CIRCUIT_BREAKER = "DRAWDOWN_CIRCUIT_BREAKER"
    SHORT_BORROW_UNAVAILABLE = "SHORT_BORROW_UNAVAILABLE"
    SHORT_FINANCING_PROHIBITIVE = "SHORT_FINANCING_PROHIBITIVE"
    SHORT_STRUCTURAL_INVALID = "SHORT_STRUCTURAL_INVALID"
    ZERO_ALLOCATION = "ZERO_ALLOCATION"
    INVALID_INPUTS = "INVALID_INPUTS"


# =============================================================================
# 25-STOCK ETF LEADER UNIVERSE TAXONOMY
# =============================================================================

DEFAULT_UNIVERSE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "NVDA": {
        "sector": "Technology",
        "industry": "Semiconductors",
        "factor_tags": ["AI_SEMICONDUCTORS", "HIGH_BETA_MOMENTUM", "AI_TECH"],
    },
    "AMD": {
        "sector": "Technology",
        "industry": "Semiconductors",
        "factor_tags": ["AI_SEMICONDUCTORS", "HIGH_BETA_MOMENTUM"],
    },
    "AVGO": {
        "sector": "Technology",
        "industry": "Semiconductors",
        "factor_tags": ["AI_SEMICONDUCTORS", "DIVIDEND_GROWTH", "AI_TECH"],
    },
    "QCOM": {
        "sector": "Technology",
        "industry": "Semiconductors",
        "factor_tags": ["AI_SEMICONDUCTORS", "MOBILE_TECH"],
    },
    "INTC": {
        "sector": "Technology",
        "industry": "Semiconductors",
        "factor_tags": ["AI_SEMICONDUCTORS", "TURNAROUND_VALUE"],
    },
    "TSM": {
        "sector": "Technology",
        "industry": "Semiconductors",
        "factor_tags": ["AI_SEMICONDUCTORS", "GLOBAL_FOUNDRY"],
    },
    "ASML": {
        "sector": "Technology",
        "industry": "Semiconductor Equipment",
        "factor_tags": ["AI_SEMICONDUCTORS", "LITHOGRAPHY_MONOPOLY"],
    },
    "MSFT": {
        "sector": "Technology",
        "industry": "Software Infrastructure",
        "factor_tags": ["MEGA_CAP_GROWTH", "AI_TECH", "CLOUD"],
    },
    "AAPL": {
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "factor_tags": ["MEGA_CAP_GROWTH", "HARDWARE_ECOSYSTEM"],
    },
    "GOOGL": {
        "sector": "Communication Services",
        "industry": "Internet Content",
        "factor_tags": ["MEGA_CAP_GROWTH", "AI_TECH", "DIGITAL_AD"],
    },
    "META": {
        "sector": "Communication Services",
        "industry": "Internet Content",
        "factor_tags": ["MEGA_CAP_GROWTH", "AI_TECH", "DIGITAL_AD"],
    },
    "AMZN": {
        "sector": "Consumer Discretionary",
        "industry": "Internet Retail",
        "factor_tags": ["MEGA_CAP_GROWTH", "CLOUD", "CONSUMER_DISCRETIONARY"],
    },
    "TSLA": {
        "sector": "Consumer Discretionary",
        "industry": "Auto Manufacturers",
        "factor_tags": ["HIGH_BETA_MOMENTUM", "CONSUMER_DISCRETIONARY", "EV_TECH"],
    },
    "JPM": {
        "sector": "Financials",
        "industry": "Banks - Diversified",
        "factor_tags": ["FINANCIAL_RATE_SENSITIVE", "VALUE_QUALITY"],
    },
    "BAC": {
        "sector": "Financials",
        "industry": "Banks - Diversified",
        "factor_tags": ["FINANCIAL_RATE_SENSITIVE", "VALUE_FINANCIAL"],
    },
    "GS": {
        "sector": "Financials",
        "industry": "Capital Markets",
        "factor_tags": ["FINANCIAL_RATE_SENSITIVE", "HIGH_BETA_FINANCIAL"],
    },
    "V": {
        "sector": "Financials",
        "industry": "Credit Services",
        "factor_tags": ["FINANCIAL_RATE_SENSITIVE", "PAYMENTS_TECH"],
    },
    "MA": {
        "sector": "Financials",
        "industry": "Credit Services",
        "factor_tags": ["FINANCIAL_RATE_SENSITIVE", "PAYMENTS_TECH"],
    },
    "XOM": {
        "sector": "Energy",
        "industry": "Oil & Gas Integrated",
        "factor_tags": ["ENERGY_CYCLICAL", "VALUE_DIVIDEND"],
    },
    "CVX": {
        "sector": "Energy",
        "industry": "Oil & Gas Integrated",
        "factor_tags": ["ENERGY_CYCLICAL", "VALUE_DIVIDEND"],
    },
    "LLY": {
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "factor_tags": ["DEFENSIVE_HEALTHCARE", "GROWTH_MOMENTUM"],
    },
    "UNH": {
        "sector": "Healthcare",
        "industry": "Healthcare Plans",
        "factor_tags": ["DEFENSIVE_HEALTHCARE", "VALUE_QUALITY"],
    },
    "JNJ": {
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "factor_tags": ["DEFENSIVE_HEALTHCARE", "DIVIDEND_VALUE"],
    },
    "COST": {
        "sector": "Consumer Staples",
        "industry": "Discount Stores",
        "factor_tags": ["DEFENSIVE_STAPLES", "QUALITY_COMPOUNDER"],
    },
    "WMT": {
        "sector": "Consumer Staples",
        "industry": "Discount Stores",
        "factor_tags": ["DEFENSIVE_STAPLES", "DIVIDEND_VALUE"],
    },
}


def resolve_instrument_taxonomy(
    ticker: str,
    custom_sector: Optional[str] = None,
    custom_factors: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Resolves sector and factor cluster tags for a ticker.
    Prefers explicit custom values, falling back to canonical 25-stock universe taxonomy,
    or general market defaults.
    """
    sym = (ticker or "").upper().strip()
    entry = DEFAULT_UNIVERSE_TAXONOMY.get(sym, {})
    sector = custom_sector or entry.get("sector", "General")
    factors = custom_factors if custom_factors is not None else entry.get("factor_tags", ["BROAD_MARKET"])
    return sector, list(factors)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Represents an active long or short holding in the portfolio."""
    ticker: str
    direction: str  # "LONG" or "SHORT"
    shares: int
    entry_price: float
    current_price: float
    sector: str = "General"
    factor_tags: List[str] = field(default_factory=list)
    borrow_fee_bps: float = 0.0
    adv_shares_20d: float = 1_000_000.0

    @property
    def market_value(self) -> float:
        """Unsigned gross market value in portfolio base currency."""
        return abs(self.shares) * max(0.0, self.current_price)

    @property
    def signed_market_value(self) -> float:
        """Signed market value (+ for long, - for short)."""
        return self.market_value if self.direction.upper() == "LONG" else -self.market_value

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized dollar profit or loss."""
        if self.direction.upper() == "LONG":
            return (self.current_price - self.entry_price) * abs(self.shares)
        else:
            return (self.entry_price - self.current_price) * abs(self.shares)

    @property
    def liquidation_days(self) -> float:
        """Estimated days to liquidate at 1.0% ADV participation."""
        if self.adv_shares_20d <= 0.0:
            return 999.0
        daily_cap = self.adv_shares_20d * 0.01
        return abs(self.shares) / daily_cap if daily_cap > 0 else 999.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["market_value"] = round(self.market_value, 2)
        d["signed_market_value"] = round(self.signed_market_value, 2)
        d["unrealized_pnl"] = round(self.unrealized_pnl, 2)
        d["liquidation_days"] = round(self.liquidation_days, 2)
        return d


@dataclass
class PortfolioSnapshot:
    """Complete Point-in-Time snapshot of portfolio equity, cash, and active holdings."""
    portfolio_equity: float
    cash: float
    peak_equity: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def __post_init__(self):
        if self.peak_equity <= 0.0 and self.portfolio_equity > 0.0:
            self.peak_equity = self.portfolio_equity

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from portfolio high-water mark."""
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.portfolio_equity) / self.peak_equity)

    @property
    def total_long_market_value(self) -> float:
        """Total market value of all long holdings."""
        return sum(p.market_value for p in self.positions.values() if p.direction.upper() == "LONG")

    @property
    def total_short_market_value(self) -> float:
        """Total gross market value of all short holdings."""
        return sum(p.market_value for p in self.positions.values() if p.direction.upper() == "SHORT")

    @property
    def gross_exposure_dollars(self) -> float:
        """Total gross capital deployed: sum(|Long|) + sum(|Short|)."""
        return self.total_long_market_value + self.total_short_market_value

    @property
    def gross_exposure_pct(self) -> float:
        """Gross exposure as percentage of portfolio equity."""
        return (self.gross_exposure_dollars / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    @property
    def net_exposure_dollars(self) -> float:
        """Net market exposure: sum(Long) - sum(Short)."""
        return self.total_long_market_value - self.total_short_market_value

    @property
    def net_exposure_pct(self) -> float:
        """Absolute net exposure percentage: |Long - Short| / Equity."""
        return (abs(self.net_exposure_dollars) / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    @property
    def signed_net_exposure_pct(self) -> float:
        """Signed net exposure percentage: (Long - Short) / Equity."""
        return (self.net_exposure_dollars / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    @property
    def cash_reserve_pct(self) -> float:
        """Unencumbered cash as fraction of portfolio equity."""
        return (self.cash / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    def get_sector_exposure_dollars(self, sector: str) -> float:
        """Calculates current gross dollar exposure in a specific sector."""
        sec_norm = (sector or "").strip().lower()
        return sum(
            p.market_value
            for p in self.positions.values()
            if (p.sector or "").strip().lower() == sec_norm
        )

    def get_sector_exposure_pct(self, sector: str) -> float:
        """Calculates current sector exposure as fraction of portfolio equity."""
        if self.portfolio_equity <= 0.0:
            return 0.0
        return self.get_sector_exposure_dollars(sector) / self.portfolio_equity

    def get_factor_cluster_exposure_dollars(self, factor_tag: str) -> float:
        """Calculates current gross dollar exposure in a correlated factor cluster."""
        tag_norm = (factor_tag or "").strip().upper()
        return sum(
            p.market_value
            for p in self.positions.values()
            if any((t or "").strip().upper() == tag_norm for t in p.factor_tags)
        )

    def get_factor_cluster_exposure_pct(self, factor_tag: str) -> float:
        """Calculates current factor cluster exposure as fraction of portfolio equity."""
        if self.portfolio_equity <= 0.0:
            return 0.0
        return self.get_factor_cluster_exposure_dollars(factor_tag) / self.portfolio_equity

    def get_all_sector_exposures_pct(self) -> Dict[str, float]:
        """Returns exposure percentages across all present sectors."""
        sectors: Set[str] = {p.sector for p in self.positions.values() if p.sector}
        return {s: round(self.get_sector_exposure_pct(s), 4) for s in sorted(sectors)}

    def get_all_factor_exposures_pct(self) -> Dict[str, float]:
        """Returns exposure percentages across all present factor clusters."""
        factors: Set[str] = set()
        for p in self.positions.values():
            factors.update(p.factor_tags)
        return {f: round(self.get_factor_cluster_exposure_pct(f), 4) for f in sorted(factors)}

    def get_portfolio_liquidation_horizon_days(self) -> float:
        """
        Calculates weighted average liquidation days for the entire portfolio
        assuming 1% ADV participation per day.
        """
        gross = self.gross_exposure_dollars
        if gross <= 0.0 or not self.positions:
            return 0.0
        weighted_days = sum(p.liquidation_days * (p.market_value / gross) for p in self.positions.values())
        return round(weighted_days, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_equity": round(self.portfolio_equity, 2),
            "cash": round(self.cash, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "gross_exposure_dollars": round(self.gross_exposure_dollars, 2),
            "gross_exposure_pct": round(self.gross_exposure_pct, 4),
            "net_exposure_dollars": round(self.net_exposure_dollars, 2),
            "net_exposure_pct": round(self.net_exposure_pct, 4),
            "signed_net_exposure_pct": round(self.signed_net_exposure_pct, 4),
            "cash_reserve_pct": round(self.cash_reserve_pct, 4),
            "portfolio_liquidation_days": self.get_portfolio_liquidation_horizon_days(),
            "positions_count": len(self.positions),
            "sector_exposures_pct": self.get_all_sector_exposures_pct(),
            "factor_exposures_pct": self.get_all_factor_exposures_pct(),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }


# =============================================================================
# SHORT POSITION ELIGIBILITY & FINANCING VALIDATOR
# =============================================================================

@dataclass
class ShortEligibilityResult:
    """Result of structural short eligibility and financing verification."""
    is_eligible: bool
    status: str  # "PASS" or "FAIL"
    reason_code: Optional[str] = None
    message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShortEligibilityValidator:
    """
    Validates whether a candidate short trade satisfies all structural and financing prerequisites.

    Mandate Invariant:
    A low composite score (e.g. <= 45.0) is a necessary condition, but does NOT by itself
    justify opening a short position.
    """

    @classmethod
    def validate(
        cls,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        composite_score: Optional[float],
        policy: Optional[TradingPolicy] = None,
        borrow_available: bool = True,
        borrow_status: str = "ETB",  # "ETB" (Easy to Borrow), "LOCATED", "HTB", "UNAVAILABLE"
        borrow_fee_bps: float = 0.0,
        available_borrow_shares: Optional[int] = None,
        shares_requested: int = 1,
        rsi: Optional[float] = None,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        trend_score: Optional[float] = None,
        financial_quality_score: Optional[float] = None,
        valuation_history_score: Optional[float] = None,
        reward_risk_ratio: Optional[float] = None,
        days_to_ex_dividend: Optional[int] = None,
        available_cash: Optional[float] = None,
        portfolio_equity: Optional[float] = None,
        is_certified_extension_sell: bool = False,
    ) -> ShortEligibilityResult:
        active_policy = policy or get_default_policy()
        pr = active_policy.portfolio_risk

        metrics: Dict[str, Any] = {
            "ticker": ticker,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "composite_score": composite_score,
            "borrow_status": borrow_status,
            "borrow_fee_bps": borrow_fee_bps,
            "rsi": rsi,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "trend_score": trend_score,
            "financial_quality_score": financial_quality_score,
            "valuation_history_score": valuation_history_score,
            "reward_risk_ratio": reward_risk_ratio,
            "days_to_ex_dividend": days_to_ex_dividend,
            "is_certified_extension_sell": is_certified_extension_sell,
        }

        # 1. Composite Score Breakdown Ceiling Check
        ceiling = 65.0 if is_certified_extension_sell else active_policy.scores.short_maximum
        if composite_score is None:
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="SHORT_COMPOSITE_MISSING",
                message="Short entry rejected: Deterministic composite score is undefined.",
                metrics=metrics,
            )
        if float(composite_score) > ceiling:
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="SHORT_CEILING_EXCEEDED",
                message=f"Short entry rejected: Composite {float(composite_score):.1f} > {ceiling:.0f} ceiling.",
                metrics=metrics,
            )

        # 2. Structural Trend Breakdown Verification (Not Just Score Alone)
        # Price must trade below key EMAs, confirming established downward trajectory
        if pr.short_require_trend_breakdown and not is_certified_extension_sell:
            trend_broken = True
            fail_reason = ""
            if ema_20 is not None and entry_price > ema_20:
                trend_broken = False
                fail_reason = f"Entry price ${entry_price:.2f} is above 20 EMA (${ema_20:.2f})"
            elif ema_50 is not None and entry_price > ema_50:
                trend_broken = False
                fail_reason = f"Entry price ${entry_price:.2f} is above 50 EMA (${ema_50:.2f})"
            elif trend_score is not None and trend_score > 40.0:
                trend_broken = False
                fail_reason = f"Pillar trend score {trend_score:.1f} > 40.0 (no structural downtrend)"

            if not trend_broken:
                return ShortEligibilityResult(
                    is_eligible=False,
                    status="FAIL",
                    reason_code="SHORT_TREND_NOT_BROKEN",
                    message=(
                        f"Short entry rejected: Low composite score alone does not justify shorting without "
                        f"structural technical breakdown ({fail_reason})."
                    ),
                    metrics=metrics,
                )

        # 3. Oversold Climax / Short Squeeze Exhaustion Guard
        # Shorting into panic exhaustion (RSI < 25) carries severe short-squeeze risk
        if rsi is not None and rsi < pr.short_min_rsi_threshold:
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="SHORT_OVERSOLD_SQUEEZE_RISK",
                message=(
                    f"Short entry rejected: RSI {rsi:.1f} < {pr.short_min_rsi_threshold:.1f}. "
                    f"Shorting into an exhausted oversold climax presents extreme mean-reversion squeeze risk."
                ),
                metrics=metrics,
            )

        # 4. Fortress-Quality Value Trap Short Squeeze Guard
        # Shorting a fortress balance sheet with depressed valuation history is hazardous
        if pr.short_forbid_high_quality_value:
            is_fortress_quality = (financial_quality_score is not None and financial_quality_score >= 80.0)
            is_depressed_val = (valuation_history_score is not None and valuation_history_score >= 70.0)
            if is_fortress_quality and is_depressed_val:
                return ShortEligibilityResult(
                    is_eligible=False,
                    status="FAIL",
                    reason_code="SHORT_VALUE_TRAP_SQUEEZE_RISK",
                    message=(
                        f"Short entry rejected: Stock has fortress quality ({financial_quality_score:.1f}) "
                        f"and historically cheap valuation ({valuation_history_score:.1f}). "
                        f"High risk of violent dividend/earnings short squeeze."
                    ),
                    metrics=metrics,
                )

        # 5. Higher Reward-to-Risk Hurdle for Shorts (1.50 vs 1.20)
        min_rr = pr.short_min_rr_threshold
        if reward_risk_ratio is not None and reward_risk_ratio < min_rr:
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="SHORT_RR_TOO_LOW",
                message=f"Short entry rejected: Setup RR {reward_risk_ratio:.2f} < short threshold {min_rr:.2f}.",
                metrics=metrics,
            )

        # 6. Borrow Availability & Locate Verification
        b_stat = (borrow_status or "").strip().upper()
        if not borrow_available or b_stat in ("UNAVAILABLE", "HARD_TO_BORROW_UNLOCATED"):
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="BORROW_UNAVAILABLE",
                message=f"Short entry rejected: Shares are not borrowable (Borrow Status: {b_stat}).",
                metrics=metrics,
            )

        if b_stat == "HTB":
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="BORROW_LOCATE_REQUIRED",
                message="Short entry rejected: Stock is Hard-to-Borrow (HTB); active locate is required.",
                metrics=metrics,
            )

        if available_borrow_shares is not None and available_borrow_shares < shares_requested:
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="INSUFFICIENT_BORROW_SHARES",
                message=(
                    f"Short entry rejected: Available borrow pool {available_borrow_shares} shares "
                    f"< requested trade size {shares_requested} shares."
                ),
                metrics=metrics,
            )

        # 7. Borrow Financing Fee Sanity Check
        max_fee = pr.short_borrow_fee_max_bps
        if borrow_fee_bps > max_fee:
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="BORROW_FEE_PROHIBITIVE",
                message=(
                    f"Short entry rejected: Annualized borrow fee {borrow_fee_bps:.1f} bps "
                    f"exceeds maximum permitted {max_fee:.1f} bps (negative carrying drag)."
                ),
                metrics=metrics,
            )

        # 20-session borrow financing drag per share
        financing_drag_per_share = (borrow_fee_bps / 10000.0) * (20.0 / 252.0) * entry_price
        metrics["financing_drag_per_share_20d"] = round(financing_drag_per_share, 4)

        # 8. Ex-Dividend Blackout Check (Short seller liable for dividend payment)
        if (
            days_to_ex_dividend is not None
            and days_to_ex_dividend <= active_policy.events.ex_dividend_short_blackout_days
        ):
            return ShortEligibilityResult(
                is_eligible=False,
                status="FAIL",
                reason_code="EX_DIVIDEND_BLACKOUT",
                message=(
                    f"Short entry rejected: Ex-dividend date in {days_to_ex_dividend} day(s). "
                    f"Short sellers must pay dividend in cash on ex-date."
                ),
                metrics=metrics,
            )

        # 9. Cash Margin Collateral Requirement
        if available_cash is not None and portfolio_equity is not None:
            # Short position collateral: 100% position value + short_margin_requirement_pct (e.g. 50%)
            margin_collateral_needed = shares_requested * entry_price * (1.0 + pr.short_margin_requirement_pct)
            min_cash_required = portfolio_equity * pr.min_cash_reserve_pct
            free_cash_for_margin = max(0.0, available_cash - min_cash_required)

            if margin_collateral_needed > free_cash_for_margin:
                return ShortEligibilityResult(
                    is_eligible=False,
                    status="FAIL",
                    reason_code="SHORT_COLLATERAL_INSUFFICIENT",
                    message=(
                        f"Short entry rejected: Required short margin collateral ${margin_collateral_needed:,.2f} "
                        f"exceeds free cash ${free_cash_for_margin:,.2f} (protecting {pr.min_cash_reserve_pct*100:.0f}% cash reserve)."
                    ),
                    metrics=metrics,
                )

        return ShortEligibilityResult(
            is_eligible=True,
            status="PASS",
            reason_code="SHORT_ELIGIBILITY_VERIFIED",
            message="Short eligibility and financing checks passed.",
            metrics=metrics,
        )


# =============================================================================
# PORTFOLIO COMPLIANCE REPORT & ENGINE
# =============================================================================

@dataclass
class PortfolioComplianceReport:
    """Detailed audit report of portfolio-level controls and capacity evaluation."""
    passed: bool
    is_tradeable: bool
    allowed_shares: int
    binding_constraint: str
    rejection_reason: Optional[str] = None
    drawdown_regime: str = DrawdownRegime.NORMAL.value
    drawdown_pct: float = 0.0
    effective_risk_fraction: float = 0.010
    single_stock_headroom_dollars: float = 0.0
    sector_headroom_dollars: float = 0.0
    factor_headroom_dollars: float = 0.0
    gross_headroom_dollars: float = 0.0
    net_headroom_dollars: float = 0.0
    cash_headroom_dollars: float = 0.0
    n_risk: int = 0
    n_single_stock_cap: int = 0
    n_sector_cap: int = 0
    n_factor_cap: int = 0
    n_gross_cap: int = 0
    n_net_cap: int = 0
    n_cash_cap: int = 0
    n_liquidity_cap: int = 0
    portfolio_liquidation_days: float = 0.0
    short_validation: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioRiskEngine:
    """
    Evaluates institutional portfolio-level controls against a proposed trade and portfolio snapshot.
    """

    @classmethod
    def evaluate_portfolio_controls(
        cls,
        ticker: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        snapshot: PortfolioSnapshot,
        policy: Optional[TradingPolicy] = None,
        sector: Optional[str] = None,
        factor_tags: Optional[List[str]] = None,
        adv_shares_20d: Optional[float] = None,
        cost_per_share: float = 0.0,
        gap_risk_reserve: float = 0.0,
        composite_score: Optional[float] = None,
        borrow_fee_bps: float = 0.0,
        borrow_available: bool = True,
        borrow_status: str = "ETB",
        available_borrow_shares: Optional[int] = None,
        rsi: Optional[float] = None,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        trend_score: Optional[float] = None,
        financial_quality_score: Optional[float] = None,
        valuation_history_score: Optional[float] = None,
        reward_risk_ratio: Optional[float] = None,
        days_to_ex_dividend: Optional[int] = None,
    ) -> PortfolioComplianceReport:
        active_policy = policy or get_default_policy()
        pr = active_policy.portfolio_risk
        dir_norm = (direction or "").strip().upper()
        sym = (ticker or "").strip().upper()

        E = snapshot.portfolio_equity
        metrics: Dict[str, Any] = {
            "ticker": sym,
            "direction": dir_norm,
            "portfolio_equity": E,
            "cash": snapshot.cash,
            "peak_equity": snapshot.peak_equity,
        }

        # 1. Fail-Closed Boundary Validation
        if E <= 0.0 or math.isnan(E) or math.isinf(E):
            return PortfolioComplianceReport(
                passed=False,
                is_tradeable=False,
                allowed_shares=0,
                binding_constraint=PortfolioBindingConstraint.INVALID_INPUTS.value,
                rejection_reason="Portfolio equity must be strictly positive and finite.",
                metrics=metrics,
            )

        if entry_price <= 0.0 or stop_loss <= 0.0:
            return PortfolioComplianceReport(
                passed=False,
                is_tradeable=False,
                allowed_shares=0,
                binding_constraint=PortfolioBindingConstraint.INVALID_INPUTS.value,
                rejection_reason="Entry price and stop loss must be strictly positive.",
                metrics=metrics,
            )

        # 2. Drawdown Governor & Circuit Breakers
        dd = snapshot.drawdown_pct
        metrics["drawdown_pct"] = round(dd, 4)

        if dd >= pr.drawdown_halt_pct:
            regime = DrawdownRegime.CIRCUIT_BREAKER_HALT
            return PortfolioComplianceReport(
                passed=False,
                is_tradeable=False,
                allowed_shares=0,
                binding_constraint=PortfolioBindingConstraint.DRAWDOWN_CIRCUIT_BREAKER.value,
                rejection_reason=(
                    f"Portfolio drawdown {dd:.1%} exceeds circuit breaker halt threshold "
                    f"{pr.drawdown_halt_pct:.1%}. All new risk entries frozen."
                ),
                drawdown_regime=regime.value,
                drawdown_pct=dd,
                metrics=metrics,
            )
        elif dd >= pr.drawdown_warning_pct:
            regime = DrawdownRegime.DE_RISKING
            # Scale down risk fraction r by dampener factor
            r_effective = pr.max_risk_fraction_per_trade * pr.drawdown_dampener_factor
        else:
            regime = DrawdownRegime.NORMAL
            r_effective = pr.max_risk_fraction_per_trade

        metrics["drawdown_regime"] = regime.value
        metrics["effective_risk_fraction"] = r_effective

        # 3. Short Eligibility & Financing Verification (If SHORT)
        short_val_report: Optional[Dict[str, Any]] = None
        if dir_norm == "SHORT":
            short_res = ShortEligibilityValidator.validate(
                ticker=sym,
                entry_price=entry_price,
                stop_loss=stop_loss,
                composite_score=composite_score,
                policy=active_policy,
                borrow_available=borrow_available,
                borrow_status=borrow_status,
                borrow_fee_bps=borrow_fee_bps,
                available_borrow_shares=available_borrow_shares,
                shares_requested=1,  # preliminary validation
                rsi=rsi,
                ema_20=ema_20,
                ema_50=ema_50,
                trend_score=trend_score,
                financial_quality_score=financial_quality_score,
                valuation_history_score=valuation_history_score,
                reward_risk_ratio=reward_risk_ratio,
                days_to_ex_dividend=days_to_ex_dividend,
                available_cash=snapshot.cash,
                portfolio_equity=E,
            )
            short_val_report = short_res.to_dict()
            if not short_res.is_eligible:
                binding_map = {
                    "BORROW_UNAVAILABLE": PortfolioBindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                    "BORROW_LOCATE_REQUIRED": PortfolioBindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                    "INSUFFICIENT_BORROW_SHARES": PortfolioBindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                    "BORROW_FEE_PROHIBITIVE": PortfolioBindingConstraint.SHORT_FINANCING_PROHIBITIVE.value,
                    "SHORT_COLLATERAL_INSUFFICIENT": PortfolioBindingConstraint.CASH_RESERVE_CAP.value,
                }
                b_constraint = binding_map.get(
                    short_res.reason_code or "", PortfolioBindingConstraint.SHORT_STRUCTURAL_INVALID.value
                )
                return PortfolioComplianceReport(
                    passed=False,
                    is_tradeable=False,
                    allowed_shares=0,
                    binding_constraint=b_constraint,
                    rejection_reason=short_res.message,
                    drawdown_regime=regime.value,
                    drawdown_pct=dd,
                    effective_risk_fraction=r_effective,
                    short_validation=short_val_report,
                    metrics=metrics,
                )

        # 4. Resolve Instrument Sector & Factor Cluster Tags (25-Stock Universe)
        resolved_sector, resolved_factors = resolve_instrument_taxonomy(
            ticker=sym, custom_sector=sector, custom_factors=factor_tags
        )
        metrics["resolved_sector"] = resolved_sector
        metrics["resolved_factor_tags"] = resolved_factors

        # 5. Risk-Based Preliminary Sizing (N_risk)
        stop_dist = abs(entry_price - stop_loss)
        eff_risk_per_share = stop_dist + cost_per_share + gap_risk_reserve
        if eff_risk_per_share <= 0.0:
            return PortfolioComplianceReport(
                passed=False,
                is_tradeable=False,
                allowed_shares=0,
                binding_constraint=PortfolioBindingConstraint.INVALID_INPUTS.value,
                rejection_reason="Effective per-share risk denominator is non-positive.",
                metrics=metrics,
            )
        equity_at_risk = E * r_effective
        n_risk = math.floor(equity_at_risk / eff_risk_per_share)

        # 6. Limit Caps & Headroom Calculations
        # 6A. Maximum Single-Stock Exposure Cap
        current_ticker_val = 0.0
        if sym in snapshot.positions:
            current_ticker_val = snapshot.positions[sym].market_value
        max_single_stock_dollars = E * pr.max_single_stock_pct
        single_stock_headroom = max(0.0, max_single_stock_dollars - current_ticker_val)
        n_single_stock_cap = math.floor(single_stock_headroom / entry_price)

        # 6B. Sector Concentration Cap
        current_sector_val = snapshot.get_sector_exposure_dollars(resolved_sector)
        max_sector_dollars = E * pr.max_sector_pct
        sector_headroom = max(0.0, max_sector_dollars - current_sector_val)
        n_sector_cap = math.floor(sector_headroom / entry_price)

        # 6C. Correlated-Factor Exposure Cap (25-Stock Universe Analysis)
        factor_caps: List[int] = []
        min_factor_headroom = 1e12
        for ftag in resolved_factors:
            current_fval = snapshot.get_factor_cluster_exposure_dollars(ftag)
            max_fdollars = E * pr.max_correlated_factor_pct
            fheadroom = max(0.0, max_fdollars - current_fval)
            min_factor_headroom = min(min_factor_headroom, fheadroom)
            factor_caps.append(math.floor(fheadroom / entry_price))
        n_factor_cap = min(factor_caps) if factor_caps else 10_000_000_000
        factor_headroom = min_factor_headroom if factor_caps else max_single_stock_dollars

        # 6D. Total Gross Exposure Cap
        max_gross_dollars = E * pr.max_gross_exposure_pct
        gross_headroom = max(0.0, max_gross_dollars - snapshot.gross_exposure_dollars)
        n_gross_cap = math.floor(gross_headroom / entry_price)

        # 6E. Total Net Exposure Cap
        max_net_dollars = E * pr.max_net_exposure_pct
        curr_net = snapshot.net_exposure_dollars
        if dir_norm == "LONG":
            net_headroom = max(0.0, max_net_dollars - curr_net)
        else:
            net_headroom = max(0.0, max_net_dollars + curr_net)
        n_net_cap = math.floor(net_headroom / entry_price)

        # 6F. Cash Reserve & Margin Collateral Cap
        min_cash_required = E * pr.min_cash_reserve_pct
        available_free_cash = max(0.0, snapshot.cash - min_cash_required)
        if dir_norm == "LONG":
            n_cash_cap = math.floor(available_free_cash / entry_price)
        else:
            # Short margin collateral per share = P * (1 + margin_pct)
            per_share_collateral = entry_price * (1.0 + pr.short_margin_requirement_pct)
            n_cash_cap = math.floor(available_free_cash / per_share_collateral)

        # 6G. Liquidity Participation Cap (1% of 20d ADV)
        if adv_shares_20d is not None and adv_shares_20d > 0.0:
            n_liquidity_cap = math.floor(adv_shares_20d * pr.max_adv_participation_pct)
        else:
            n_liquidity_cap = 10_000_000_000

        # 7. Permissible Share Count Resolution
        cap_candidates = [
            (n_risk, PortfolioBindingConstraint.RISK_BUDGET),
            (n_single_stock_cap, PortfolioBindingConstraint.SINGLE_STOCK_CAP),
            (n_sector_cap, PortfolioBindingConstraint.SECTOR_CAP),
            (n_factor_cap, PortfolioBindingConstraint.FACTOR_CAP),
            (n_gross_cap, PortfolioBindingConstraint.GROSS_EXPOSURE_CAP),
            (n_net_cap, PortfolioBindingConstraint.NET_EXPOSURE_CAP),
            (n_cash_cap, PortfolioBindingConstraint.CASH_RESERVE_CAP),
            (n_liquidity_cap, PortfolioBindingConstraint.LIQUIDITY_CAP),
        ]

        min_shares = min(s for s, _ in cap_candidates)
        n_allowed = max(0, min_shares)

        # Identify binding constraint
        if n_allowed == 0:
            # Find the zero-forcing constraint
            zero_reasons = [c for s, c in cap_candidates if s <= 0]
            binding_enum = zero_reasons[0] if zero_reasons else PortfolioBindingConstraint.ZERO_ALLOCATION
        else:
            for s, c in cap_candidates:
                if s == n_allowed:
                    binding_enum = c
                    break
            else:
                binding_enum = PortfolioBindingConstraint.RISK_BUDGET

        binding_str = binding_enum.value

        # 8. Portfolio Liquidation Horizon Evaluation
        port_liq_days = snapshot.get_portfolio_liquidation_horizon_days()
        metrics["portfolio_liquidation_days"] = port_liq_days

        is_tradeable = n_allowed >= pr.min_trade_size_shares
        rejection_reason = None
        if not is_tradeable:
            if n_allowed == 0:
                rejection_reason = f"Portfolio controls constrained position size to 0 shares (limited by {binding_str})."
            else:
                rejection_reason = (
                    f"Allowed shares {n_allowed} below minimum trade size {pr.min_trade_size_shares} (limited by {binding_str})."
                )

        return PortfolioComplianceReport(
            passed=is_tradeable,
            is_tradeable=is_tradeable,
            allowed_shares=n_allowed,
            binding_constraint=binding_str,
            rejection_reason=rejection_reason,
            drawdown_regime=regime.value,
            drawdown_pct=round(dd, 4),
            effective_risk_fraction=round(r_effective, 4),
            single_stock_headroom_dollars=round(single_stock_headroom, 2),
            sector_headroom_dollars=round(sector_headroom, 2),
            factor_headroom_dollars=round(factor_headroom, 2),
            gross_headroom_dollars=round(gross_headroom, 2),
            net_headroom_dollars=round(net_headroom, 2),
            cash_headroom_dollars=round(available_free_cash, 2),
            n_risk=n_risk,
            n_single_stock_cap=n_single_stock_cap,
            n_sector_cap=n_sector_cap,
            n_factor_cap=n_factor_cap,
            n_gross_cap=n_gross_cap,
            n_net_cap=n_net_cap,
            n_cash_cap=n_cash_cap,
            n_liquidity_cap=n_liquidity_cap,
            portfolio_liquidation_days=port_liq_days,
            short_validation=short_val_report,
            metrics=metrics,
        )
