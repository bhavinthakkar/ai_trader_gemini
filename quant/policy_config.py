"""Centralized Deterministic Trading Compliance Policy Configuration (Step 5.1).

Defines the single source of truth for:
- Composite score floors and ceilings (BUY floor 60.0, Dip-Buy floor 52.0, Short ceiling 45.0)
- Distinct catalyst dip-buy branch specification (dislocation vs. momentum)
- Trade quality and setup geometry constraints (Reward:Risk >= 1.20)
- Event blackouts (earnings 3 sessions, ex-dividend 2 days)
- Volatility regime and option expected move noise boundaries
- Data quality and point-in-time freshness requirements
- Holding-thesis guard preventing premature long exits (Composite >= 70.0)
- Policy governance (fail-closed, LLM has no execution authority)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ScoresConfig:
    """Composite score floors and ceilings (0-100)."""
    buy_minimum: float = 60.0
    dip_buy_minimum: float = 52.0
    short_maximum: float = 45.0
    holding_thesis_minimum: float = 70.0

    def validate(self) -> None:
        if self.dip_buy_minimum > self.buy_minimum:
            raise ValueError(
                f"Contradictory policy: dip_buy_minimum ({self.dip_buy_minimum}) "
                f"cannot be greater than standard buy_minimum ({self.buy_minimum})"
            )
        if self.short_maximum >= self.buy_minimum:
            raise ValueError(
                f"Contradictory policy: short_maximum ({self.short_maximum}) "
                f"must be strictly less than buy_minimum ({self.buy_minimum}) to prevent overlapping signals"
            )


@dataclass
class DipBuyConfig:
    """Distinct catalyst dip-buy branch specification."""
    enabled: bool = True
    score_minimum: float = 52.0
    min_drop_pct_5d: float = 5.0
    max_drop_pct_5d: float = 15.0
    min_rsi: float = 30.0
    max_rsi: float = 45.0
    min_valuation_history: float = 60.0
    min_peer_valuation: float = 55.0
    min_rvol: float = 1.2
    max_catalyst_age_days: int = 3
    forbid_value_trap: bool = True
    forbid_collapse: bool = True

    def validate(self) -> None:
        if self.min_drop_pct_5d > self.max_drop_pct_5d:
            raise ValueError(f"Invalid dip-buy drop bounds: min {self.min_drop_pct_5d}% > max {self.max_drop_pct_5d}%")
        if self.min_rsi > self.max_rsi:
            raise ValueError(f"Invalid dip-buy RSI bounds: min {self.min_rsi} > max {self.max_rsi}")


@dataclass
class TradeQualityConfig:
    """Trade quality, setup geometry, execution assumptions, and liquidity constraints."""
    reward_risk_minimum: float = 1.20
    min_data_completeness: float = 0.80
    min_rvol_20d: float = 0.50
    min_avg_dollar_volume: float = 1_000_000.0
    max_distance_to_resistance_atr: float = 0.80
    min_wall_street_rr: float = 1.0
    # Step 5.2 & 5.3: Execution assumptions and validated risk bounds
    max_stop_distance_pct: float = 0.30
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    min_edge_pct: float = 0.02
    require_positive_expectancy: bool = True
    # Step 5.3: Universal liquidity controls & market impact sizing
    max_participation_rate: float = 0.01  # Max 1.0% of 20d ADV
    max_market_impact_bps: float = 15.0  # Max 15 bps square-root impact
    min_confidence_threshold: float = 0.65  # Empirically justified Gate 5 threshold

    def validate(self) -> None:
        if self.reward_risk_minimum <= 0.0:
            raise ValueError(f"reward_risk_minimum must be positive, got {self.reward_risk_minimum}")
        if not (0.0 <= self.min_data_completeness <= 1.0):
            raise ValueError(f"min_data_completeness must be in [0.0, 1.0], got {self.min_data_completeness}")
        if self.max_stop_distance_pct <= 0.0:
            raise ValueError(f"max_stop_distance_pct must be positive, got {self.max_stop_distance_pct}")
        if self.slippage_bps < 0.0:
            raise ValueError(f"slippage_bps cannot be negative, got {self.slippage_bps}")
        if self.commission_bps < 0.0:
            raise ValueError(f"commission_bps cannot be negative, got {self.commission_bps}")
        if not (0.0 < self.max_participation_rate <= 0.10):
            raise ValueError("max_participation_rate must be in (0.0, 0.10]")


@dataclass
class EventsConfig:
    """Scheduled events, earnings blackouts, and corporate actions."""
    earnings_blackout_sessions: int = 3
    event_risk_blackout: bool = True
    ex_dividend_short_blackout_days: int = 2

    def validate(self) -> None:
        if self.earnings_blackout_sessions < 0:
            raise ValueError("earnings_blackout_sessions cannot be negative")


@dataclass
class VolatilityConfig:
    """Volatility noise, stop placement vs options expected move, and regime safety."""
    min_vol_factor: float = 0.80
    max_stop_expected_move_ratio: float = 0.85
    min_stop_noise_multiple: float = 0.85
    option_horizon_days: int = 20
    realized_vol_fallback_window: int = 20
    reject_extreme_regime: bool = True

    def validate(self) -> None:
        if not (0.0 < self.min_vol_factor <= 1.0):
            raise ValueError("min_vol_factor must be in (0.0, 1.0]")
        if self.option_horizon_days <= 0:
            raise ValueError("option_horizon_days must be positive")


@dataclass
class DataQualityConfig:
    """Data quality and Point-in-Time freshness contracts (Step 5.4)."""
    require_fresh_market_data: bool = True
    require_fresh_ml_data_for_in_scope_tickers: bool = True
    max_market_data_staleness_days: int = 5
    max_ml_score_staleness_days: int = 5
    block_new_entries_on_outage: bool = True
    allow_neutral_score_substitution: bool = False
    enable_open_position_failsafe: bool = True
    max_outage_tolerance_hours: float = 2.0


@dataclass
class HoldingGuardConfig:
    """Holding thesis guard to prevent premature long exits on noise."""
    enabled: bool = True
    thesis_intact_score_minimum: float = 70.0
    prevent_premature_exit: bool = True
    never_block_stop_breach: bool = True
    never_block_critical_evidence_veto: bool = True
    never_block_earnings_blackout: bool = True
    catastrophic_composite_floor: float = 50.0


@dataclass
class PolicyGovernanceConfig:
    """Execution authority and safety defaults."""
    llm_has_execution_authority: bool = False
    fail_closed_for_new_entries: bool = True
    enable_evidence_veto: bool = True
    enable_value_trap_veto: bool = True


@dataclass
class SignalStrengthConfig:
    """Formalized signal strength and confidence calculation parameters (Step 5.5)."""
    neutral_score_anchor: float = 50.0
    dispersion_metric: str = "mad"
    max_dispersion_scale: float = 50.0
    min_valid_pillars: int = 3
    fallback_agreement: float = 0.50
    combination_method: str = "hierarchical_gated"
    weight_decisiveness: float = 0.70
    weight_agreement: float = 0.30
    completeness_blend_weight: float = 0.50
    linear_weight_decisiveness: float = 0.50
    linear_weight_agreement: float = 0.25
    linear_weight_completeness: float = 0.25
    hold_ceiling: float = 0.60
    min_signal_strength_threshold: float = 0.65
    is_calibrated_probability: bool = False


@dataclass
class TradingPolicy:
    """Root Trading Compliance Policy Container."""
    version: str = "1.0.0"
    policy_name: str = "Institutional Deterministic Compliance Policy"
    scores: ScoresConfig = field(default_factory=ScoresConfig)
    dip_buy: DipBuyConfig = field(default_factory=DipBuyConfig)
    trade_quality: TradeQualityConfig = field(default_factory=TradeQualityConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    holding_guard: HoldingGuardConfig = field(default_factory=HoldingGuardConfig)
    policy: PolicyGovernanceConfig = field(default_factory=PolicyGovernanceConfig)
    signal_strength: SignalStrengthConfig = field(default_factory=SignalStrengthConfig)

    def validate(self) -> None:
        """Validates all sub-configurations and cross-rule consistency."""
        self.scores.validate()
        self.dip_buy.validate()
        self.trade_quality.validate()
        self.events.validate()
        self.volatility.validate()

        # Cross-rule validation between scores and dip_buy branch
        if self.dip_buy.enabled and self.dip_buy.score_minimum > self.scores.buy_minimum:
            raise ValueError(
                f"Contradictory policy: dip_buy.score_minimum ({self.dip_buy.score_minimum}) "
                f"exceeds standard scores.buy_minimum ({self.scores.buy_minimum})"
            )

    # -------------------------------------------------------------------------
    # Precedence & Conflict Resolution Helpers
    # -------------------------------------------------------------------------
    def resolve_buy_score_floor(self, is_certified_dip_buy: bool) -> Tuple[float, str]:
        """
        Resolves the precedence between standard BUY floor (60.0) and Dip-Buy floor (52.0).

        Returns:
            (effective_floor, rule_name)
        """
        if is_certified_dip_buy and self.dip_buy.enabled:
            return self.dip_buy.score_minimum, "RULE_DIP_BUY_DISCOUNT_FLOOR"
        return self.scores.buy_minimum, "RULE_STANDARD_BUY_FLOOR"

    def is_holding_thesis_intact(
        self,
        is_holding: bool,
        composite_score: Optional[float],
        has_critical_risk: bool = False,
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        is_earnings_blackout: bool = False,
    ) -> bool:
        """
        Evaluates the Holding-Thesis Guard (Step 5.3):
        Prevents premature liquidation if position is held, composite score >= 70,
        and no fatal thesis-breaking risk is present.

        MANDATORY FORCED EXITS:
        The holding guard NEVER blocks exits under:
        1. Stop loss breach (current_price <= stop_loss)
        2. Critical evidence veto (has_critical_risk)
        3. Scheduled earnings blackout (is_earnings_blackout)
        4. Catastrophic composite collapse (composite < catastrophic_composite_floor)
        """
        if not (self.holding_guard.enabled and self.holding_guard.prevent_premature_exit):
            return False
        if not is_holding:
            return False

        # Forced Exit Invariants: Must never be blocked by thesis guard
        if self.holding_guard.never_block_critical_evidence_veto and has_critical_risk:
            return False
        if self.holding_guard.never_block_stop_breach and (
            current_price is not None and stop_loss is not None and current_price <= stop_loss
        ):
            return False
        if self.holding_guard.never_block_earnings_blackout and is_earnings_blackout:
            return False
        if composite_score is None:
            return False
        if float(composite_score) < self.holding_guard.catastrophic_composite_floor:
            return False

        return float(composite_score) >= self.holding_guard.thesis_intact_score_minimum

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_default_policy() -> TradingPolicy:
    """Returns the canonical institutional policy with verified default parameters."""
    policy = TradingPolicy()
    policy.validate()
    return policy


def load_policy_config(config_path: Optional[str] = None) -> TradingPolicy:
    """
    Loads and validates the trading compliance policy from a YAML configuration file.

    Parameters:
        config_path: Optional path to policy YAML file. Defaults to 'config/trading_policy.yaml'.

    Returns:
        Validated TradingPolicy instance.
    """
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "trading_policy.yaml",
    )
    target_path = config_path or default_path

    if not os.path.isfile(target_path) or yaml is None:
        policy = get_default_policy()
        _apply_env_overrides(policy)
        return policy

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # Parse sections with safe defaults
        scores_data = raw_data.get("scores", {})
        dip_buy_data = raw_data.get("dip_buy", {})
        tq_data = raw_data.get("trade_quality", {})
        events_data = raw_data.get("events", {})
        vol_data = raw_data.get("volatility", {})
        dq_data = raw_data.get("data_quality", {})
        hg_data = raw_data.get("holding_guard", {})
        pol_data = raw_data.get("policy", {})
        ss_data = raw_data.get("signal_strength", {})

        policy = TradingPolicy(
            version=str(raw_data.get("version", "1.0.0")),
            policy_name=str(raw_data.get("policy_name", "Institutional Policy")),
            scores=ScoresConfig(**scores_data),
            dip_buy=DipBuyConfig(**dip_buy_data),
            trade_quality=TradeQualityConfig(**tq_data),
            events=EventsConfig(**events_data),
            volatility=VolatilityConfig(**vol_data),
            data_quality=DataQualityConfig(**dq_data),
            holding_guard=HoldingGuardConfig(**hg_data),
            policy=PolicyGovernanceConfig(**pol_data),
            signal_strength=SignalStrengthConfig(**ss_data),
        )
        policy.validate()
        _apply_env_overrides(policy)
        return policy

    except Exception as e:
        # Fallback to defaults with environmental overrides to guarantee fail-safe availability
        fallback = get_default_policy()
        _apply_env_overrides(fallback)
        return fallback


def _apply_env_overrides(policy: TradingPolicy) -> None:
    """Applies environment variable overrides if present."""
    env_buy_floor = os.getenv("BUY_COMPOSITE_FLOOR")
    if env_buy_floor:
        try:
            policy.scores.buy_minimum = float(env_buy_floor)
        except ValueError:
            pass

    env_rr = os.getenv("MIN_REWARD_RISK_RATIO")
    if env_rr:
        try:
            policy.trade_quality.reward_risk_minimum = float(env_rr)
        except ValueError:
            pass
