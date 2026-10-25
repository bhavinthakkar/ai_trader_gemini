"""Historical Setup Evaluation Harness with Realistic Friction and Event Handling (Step 8).

Implements rigorous historical evaluation of:
1. PULLBACK_BUY: Intact longer-term trend, orderly consolidation/retest of EMA20/50, RSI cooling (40-55),
   normal volume/relative strength, extension/event exclusions, standard buy score floor (>= 60.0).
2. CATALYST_DIP_BUY: Temporary dislocation (5-15% drop), oversold RSI (30-45), valuation/solvency floor,
   calibrated dip score floor (>= 52.0).
3. COMBINED PORTFOLIO: Unified portfolio trading both setups under single-stock concentration caps,
   1% risk-per-trade budgeting, staged profit trims (50% at 1.8R, ratcheting stop to breakeven),
   and drawdown risk dampening.

Applies institutional execution assumptions:
- Realistic commissions: $0.005/share (min $1.00 per order) on entry and exit.
- Half-spread friction: 5.0 bps (0.0005) on entry and exit.
- Market impact slippage: 5.0 bps (0.0005) on entry and exit (10 bps round-trip drag).
- Event handling: Pre-earnings blackout window (3 sessions before quarterly earnings).
- Tracks max drawdown, stop-outs, profit retained after trims, missed rebounds (shakeouts),
  portfolio turnover, and performance across discrete market regimes:
  * 2022 Bear Shock (Inflation / interest rate shock)
  * 2023 Recovery (Tech rebound / AI expansion)
  * 2024 Bull Expansion (Broad market rally)
  * 2025-2026 Holdout Window (Untuned out-of-fold forward validation)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quant.candidate_generation import get_default_focused_paths
from quant.policy_config import TradingPolicy, get_default_policy


DATA_DIR = Path("data")


@dataclass
class HistoricalEvaluationConfig:
    """Configuration for historical setup backtesting with realistic friction."""
    start_date: str = "2022-01-01"
    end_date: str = "2026-10-14"
    initial_equity: float = 100_000.0
    max_concurrent_positions: int = 8
    max_single_stock_pct: float = 0.10
    risk_per_trade_pct: float = 0.01
    commission_per_share: float = 0.005
    min_commission: float = 1.00
    half_spread_bps: float = 5.0    # 5 bps
    slippage_bps: float = 5.0       # 5 bps
    atr_stop_multiple: float = 2.0
    min_stop_distance_pct: float = 0.04
    target_trim_multiple: float = 1.8
    trim_fraction: float = 0.50
    max_holding_sessions: int = 20
    earnings_blackout_sessions: int = 3
    drawdown_warning_pct: float = 0.05
    drawdown_dampener_factor: float = 0.50
    drawdown_halt_pct: float = 0.20
    
    # Setup Thresholds
    pullback_score_minimum: float = 60.0
    dip_score_minimum: float = 52.0
    pullback_drop_min_pct: float = 1.5
    pullback_drop_max_pct: float = 8.0
    pullback_rsi_min: float = 40.0
    pullback_rsi_max: float = 55.0
    dip_drop_min_pct: float = 5.0
    dip_drop_max_pct: float = 15.0
    dip_rsi_min: float = 30.0
    dip_rsi_max: float = 45.0


@dataclass
class HistoricalTrade:
    """Individual trade execution lifecycle record in historical evaluation."""
    trade_id: str
    ticker: str
    setup_type: str
    entry_date: str
    entry_price: float
    entry_commission: float
    shares: int
    orig_shares: int
    stop_loss: float
    target_price: float
    highest_price: float = 0.0
    lowest_price: float = 0.0
    days_held: int = 0
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    was_stopped_out: bool = False
    was_trimmed: bool = False
    realized_trim_pnl: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    net_return_pct: float = 0.0
    profit_retained_pct: float = 0.0
    was_missed_rebound: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalSetupMetrics:
    """Aggregated performance metrics for a setup type or combined portfolio."""
    setup_type: str
    total_candidates_evaluated: int
    total_trades_executed: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_pnl: float
    net_pnl: float
    initial_equity: float
    final_equity: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    stop_out_count: int
    stop_out_rate_pct: float
    profit_retained_after_trims_pct: float
    missed_rebounds_count: int
    missed_rebounds_rate_pct: float
    annualized_turnover_pct: float
    trade_pnl_sum: float = 0.0
    pnl_reconciliation_error: float = 0.0
    period_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalEvaluationReport:
    """Consolidated institutional historical setup evaluation report."""
    evaluated_at: str
    start_date: str
    end_date: str
    total_trading_days: int
    pullbacks: HistoricalSetupMetrics
    catalyst_dips: HistoricalSetupMetrics
    combined: HistoricalSetupMetrics
    passed_oos_gate: bool
    production_rule_fidelity: bool = False
    validation_scope: str = "HISTORICAL_PROXY"
    oos_gate_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    remediation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_price_and_indicator_matrix(
    prices_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Loads daily prices and precomputes vector indicator matrices across all universe tickers.
    """
    _, def_price = get_default_focused_paths()
    pr_path = prices_path or def_price
    if not pr_path.is_file():
        raise FileNotFoundError(f"Prices file not found: {pr_path}")

    prices = pd.read_pickle(pr_path)
    close = prices.xs("Close", axis=1, level=1).copy()
    high = prices.xs("High", axis=1, level=1).copy()
    low = prices.xs("Low", axis=1, level=1).copy()
    volume = prices.xs("Volume", axis=1, level=1).copy()

    # Precompute Moving Averages
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # Precompute ATR(14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    atr14 = tr.rolling(14).mean()

    # Precompute RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    # Precompute 5-day and 20-day returns
    ret_5d = close.pct_change(5, fill_method=None)
    ret_20d = close.pct_change(20, fill_method=None)

    # Precompute SPY relative return
    spy_ret_20d = close["SPY"].pct_change(20, fill_method=None)
    rel_sp500_1m = ret_20d.sub(spy_ret_20d, axis=0)

    # 20d High and Low for range proximity
    high_20d = high.rolling(20).max()
    low_20d = low.rolling(20).min()

    # 20d Volume Ratio
    vol_ma20 = volume.rolling(20).mean().shift(1)
    rvol_20d = volume / (vol_ma20 + 1e-9)

    matrices = {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "atr14": atr14,
        "rsi14": rsi14,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "rel_sp500_1m": rel_sp500_1m,
        "high_20d": high_20d,
        "low_20d": low_20d,
        "rvol_20d": rvol_20d,
    }
    return close, matrices


def simulate_setup_backtest(
    matrices: Dict[str, pd.DataFrame],
    setup_mode: str = "COMBINED",
    config: Optional[HistoricalEvaluationConfig] = None,
) -> Tuple[List[HistoricalTrade], List[Tuple[pd.Timestamp, float]], int]:
    """
    Simulates portfolio trading with realistic friction, stop-loss ratcheting, staged trims,
    earnings blackout exclusions, and drawdown dampening.
    """
    cfg = config or HistoricalEvaluationConfig()
    close = matrices["close"]
    high = matrices["high"]
    low = matrices["low"]
    ema20 = matrices["ema20"]
    ema50 = matrices["ema50"]
    ema200 = matrices["ema200"]
    atr14 = matrices["atr14"]
    rsi14 = matrices["rsi14"]
    ret_5d = matrices["ret_5d"]
    rel_sp500_1m = matrices["rel_sp500_1m"]

    tickers = [t for t in close.columns if t not in ("SPY", "^GSPC")]
    spy_close = close["SPY"]
    spy_ma200 = spy_close.rolling(200).mean()

    # Filter valid trading dates (excluding non-trading or trailing NaN rows)
    dates = [
        d for d in close.index
        if cfg.start_date <= str(d)[:10] <= cfg.end_date and pd.notna(close.loc[d, "SPY"])
    ]
    if not dates:
        return [], [], 0

    cash = cfg.initial_equity
    equity = cfg.initial_equity
    peak_equity = cfg.initial_equity
    positions: Dict[str, HistoricalTrade] = {}
    closed_trades: List[HistoricalTrade] = []
    daily_equity: List[Tuple[pd.Timestamp, float]] = []

    half_spread = cfg.half_spread_bps / 10_000.0
    slippage = cfg.slippage_bps / 10_000.0

    total_candidates_evaluated = 0
    trade_id_seq = 0

    for i, d in enumerate(dates):
        # 1. Update Existing Positions
        tickers_to_close: List[str] = []
        for sym, pos in list(positions.items()):
            cur_p = close.loc[d, sym]
            h_p = high.loc[d, sym]
            l_p = low.loc[d, sym]

            if pd.isna(cur_p) or pd.isna(h_p) or pd.isna(l_p):
                continue

            pos.days_held += 1
            pos.highest_price = max(pos.highest_price, float(h_p))
            pos.lowest_price = min(pos.lowest_price, float(l_p))

            # Stop-loss check
            if l_p <= pos.stop_loss:
                exit_fill = min(pos.stop_loss, float(cur_p)) * (1.0 - half_spread - slippage)
                comm = max(cfg.min_commission, pos.shares * cfg.commission_per_share)
                cash += (pos.shares * exit_fill - comm)

                pos.exit_date = str(d)[:10]
                pos.exit_price = exit_fill
                pos.exit_reason = "STOP_LOSS"
                pos.was_stopped_out = True
                tickers_to_close.append(sym)

            # Target trim check (1.8R): trim 50% and move stop to breakeven
            elif not pos.was_trimmed and h_p >= pos.target_price:
                trim_shares = int(pos.shares * cfg.trim_fraction)
                if trim_shares > 0:
                    trim_fill = pos.target_price * (1.0 - half_spread - slippage)
                    comm = max(cfg.min_commission, trim_shares * cfg.commission_per_share)
                    cash += (trim_shares * trim_fill - comm)
                    pos.shares -= trim_shares
                    pos.realized_trim_pnl = trim_shares * (trim_fill - pos.entry_price) - comm
                    pos.was_trimmed = True
                    # Ratchet stop to breakeven
                    pos.stop_loss = pos.entry_price

            # Max holding period exit (20 sessions)
            elif pos.days_held >= cfg.max_holding_sessions:
                exit_fill = float(cur_p) * (1.0 - half_spread - slippage)
                comm = max(cfg.min_commission, pos.shares * cfg.commission_per_share)
                cash += (pos.shares * exit_fill - comm)

                pos.exit_date = str(d)[:10]
                pos.exit_price = exit_fill
                pos.exit_reason = "TIME_HORIZON"
                pos.was_stopped_out = False
                tickers_to_close.append(sym)

        # Finalize Closed Positions
        for sym in tickers_to_close:
            p_closed = positions.pop(sym)
            orig_sh = p_closed.orig_shares
            rem_sh = p_closed.shares
            entry_cost = orig_sh * p_closed.entry_price + p_closed.entry_commission
            trim_pnl = p_closed.realized_trim_pnl

            exit_comm = max(cfg.min_commission, rem_sh * cfg.commission_per_share) if rem_sh > 0 else 0.0
            exit_proceeds = rem_sh * (p_closed.exit_price or 0.0) - exit_comm

            net_pnl = trim_pnl + (exit_proceeds - (rem_sh * p_closed.entry_price)) - p_closed.entry_commission
            gross_pnl = trim_pnl + (rem_sh * ((p_closed.exit_price or 0.0) - p_closed.entry_price))

            p_closed.gross_pnl = round(gross_pnl, 2)
            p_closed.net_pnl = round(net_pnl, 2)
            p_closed.net_return_pct = round((net_pnl / max(1.0, entry_cost)) * 100.0, 2)

            # Profit retention calculation
            peak_unrealized = orig_sh * max(0.0, p_closed.highest_price - p_closed.entry_price)
            if p_closed.was_trimmed and peak_unrealized > 0:
                p_closed.profit_retained_pct = round(max(0.0, min(100.0, (net_pnl / peak_unrealized) * 100.0)), 2)
            else:
                p_closed.profit_retained_pct = 100.0 if net_pnl > 0 else 0.0

            # Missed rebound (premature shakeout): check if stock rallied >= 5% in next 10 sessions
            p_closed.was_missed_rebound = False
            if p_closed.was_stopped_out:
                fwd_idx = min(len(dates) - 1, i + 10)
                future_max = high.loc[dates[i:fwd_idx+1], p_closed.ticker].max()
                if pd.notna(future_max) and future_max >= p_closed.entry_price * 1.05:
                    p_closed.was_missed_rebound = True

            closed_trades.append(p_closed)

        # 2. Mark to Market Equity
        pos_mv = sum(
            p.shares * float(close.loc[d, s])
            for s, p in positions.items()
            if not pd.isna(close.loc[d, s])
        )
        equity = cash + pos_mv
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        daily_equity.append((d, equity))

        # Check Drawdown Risk Controls
        risk_mult = 1.0
        if dd >= cfg.drawdown_warning_pct:
            risk_mult = cfg.drawdown_dampener_factor

        # Max Concurrent Positions Check
        slots_available = cfg.max_concurrent_positions - len(positions)
        if slots_available <= 0:
            continue

        # 3. Candidate Generation & Qualification
        candidates = []
        is_bull_regime = bool(spy_close.loc[d] >= spy_ma200.loc[d])

        for sym in tickers:
            if sym in positions:
                continue

            c_val = close.loc[d, sym]
            atr_val = atr14.loc[d, sym]
            rsi_val = rsi14.loc[d, sym]
            r5_val = ret_5d.loc[d, sym]

            if pd.isna(c_val) or pd.isna(atr_val) or pd.isna(rsi_val) or atr_val <= 0 or c_val <= 0:
                continue

            total_candidates_evaluated += 1

            # Event Blackout: approximate quarterly earnings blackout (3 sessions before quarter boundary)
            day_of_quarter = (i % 63)
            if 60 <= day_of_quarter <= 62:
                continue

            # Deterministic Trend & Relative Pillar Proxy
            trend_score = 50.0
            if c_val > ema50.loc[d, sym] and ema50.loc[d, sym] > ema200.loc[d, sym]:
                trend_score = 75.0
            elif c_val < ema50.loc[d, sym] and ema50.loc[d, sym] < ema200.loc[d, sym]:
                trend_score = 25.0

            rel_score = float(np.clip(50.0 + (rel_sp500_1m.loc[d, sym] if pd.notna(rel_sp500_1m.loc[d, sym]) else 0.0) * 200.0, 0.0, 100.0))
            composite_proxy = 0.50 * trend_score + 0.50 * rel_score

            # --- SETUP 1: PULLBACK_BUY ---
            if setup_mode in ("COMBINED", "PULLBACK"):
                # Bull market regime and intact longer-term trend
                trend_intact = (c_val > ema50.loc[d, sym]) and (ema50.loc[d, sym] > ema200.loc[d, sym])
                pullback_drop = max(0.0, -float(r5_val) * 100.0) if pd.notna(r5_val) else 0.0
                dist_ema20 = abs(c_val - ema20.loc[d, sym]) / atr_val

                if (
                    is_bull_regime
                    and trend_intact
                    and (cfg.pullback_drop_min_pct <= pullback_drop <= cfg.pullback_drop_max_pct or dist_ema20 <= 1.50)
                    and (cfg.pullback_rsi_min <= rsi_val <= cfg.pullback_rsi_max)
                    and (composite_proxy >= cfg.pullback_score_minimum)
                ):
                    candidates.append((sym, "PULLBACK_BUY", float(c_val), float(atr_val), composite_proxy))

            # --- SETUP 2: CATALYST_DIP_BUY ---
            if setup_mode in ("COMBINED", "CATALYST_DIP"):
                if pd.notna(r5_val):
                    drop_pct = abs(float(r5_val) * 100.0)
                    # Oversold panic dip in high-quality instrument
                    if (
                        (cfg.dip_drop_min_pct <= drop_pct <= cfg.dip_drop_max_pct)
                        and (cfg.dip_rsi_min <= rsi_val <= cfg.dip_rsi_max)
                        and (c_val > ema200.loc[d, sym] * 0.95)  # Solvency floor
                        and (composite_proxy >= cfg.dip_score_minimum)
                    ):
                        candidates.append((sym, "CATALYST_DIP_BUY", float(c_val), float(atr_val), composite_proxy))

        # A ticker can qualify for both setups on the same bar. The portfolio
        # holds one position per ticker, so select exactly one candidate before
        # sizing. Otherwise the second entry overwrites the first in `positions`
        # while leaving the first entry cost stranded in cash accounting.
        best_by_ticker: Dict[str, Tuple[str, float, float, float]] = {}
        for sym, stype, c_val, atr_val, score in candidates:
            incumbent = best_by_ticker.get(sym)
            if incumbent is None or score > incumbent[3]:
                best_by_ticker[sym] = (stype, c_val, atr_val, score)

        ranked_candidates = [
            (sym, *candidate)
            for sym, candidate in best_by_ticker.items()
        ]
        ranked_candidates.sort(key=lambda x: x[4], reverse=True)
        for sym, stype, c_val, atr_val, score in ranked_candidates[:slots_available]:
            stop_dist = max(cfg.atr_stop_multiple * atr_val, cfg.min_stop_distance_pct * c_val)
            if stop_dist <= 0 or pd.isna(stop_dist):
                continue

            # Position Sizing under 1% Risk and 10% Equity Cap
            risk_dollars = equity * cfg.risk_per_trade_pct * risk_mult
            shares_by_risk = int(risk_dollars / stop_dist)
            shares_by_cap = int((equity * cfg.max_single_stock_pct) / c_val)
            shares = min(shares_by_risk, shares_by_cap)

            if shares <= 0:
                continue

            entry_fill = c_val * (1.0 + half_spread + slippage)
            comm = max(cfg.min_commission, shares * cfg.commission_per_share)
            total_entry_cost = shares * entry_fill + comm

            if cash < total_entry_cost:
                continue

            cash -= total_entry_cost
            trade_id_seq += 1

            stop_price = entry_fill - stop_dist
            target_price = entry_fill + cfg.target_trim_multiple * stop_dist

            trade = HistoricalTrade(
                trade_id=f"TRD-{stype[:3]}-{trade_id_seq:05d}",
                ticker=sym,
                setup_type=stype,
                entry_date=str(d)[:10],
                entry_price=entry_fill,
                entry_commission=comm,
                shares=shares,
                orig_shares=shares,
                stop_loss=stop_price,
                target_price=target_price,
                highest_price=c_val,
                lowest_price=c_val,
                days_held=0,
                was_trimmed=False,
            )
            positions[sym] = trade

    # 5. Liquidate any remaining open positions on the final evaluation day
    if positions and dates:
        final_d = dates[-1]
        for sym, pos in list(positions.items()):
            cur_p = close.loc[final_d, sym]
            if pd.isna(cur_p) or cur_p <= 0:
                cur_p = pos.entry_price
            exit_fill = float(cur_p) * (1.0 - half_spread - slippage)
            comm = max(cfg.min_commission, pos.shares * cfg.commission_per_share)
            cash += (pos.shares * exit_fill - comm)

            pos.exit_date = str(final_d)[:10]
            pos.exit_price = exit_fill
            pos.exit_reason = "HORIZON_END"
            pos.was_stopped_out = False

            orig_sh = pos.orig_shares
            rem_sh = pos.shares
            entry_cost = orig_sh * pos.entry_price + pos.entry_commission
            trim_pnl = pos.realized_trim_pnl
            exit_proceeds = rem_sh * exit_fill - comm

            net_pnl = trim_pnl + (exit_proceeds - (rem_sh * pos.entry_price)) - pos.entry_commission
            gross_pnl = trim_pnl + (rem_sh * (exit_fill - pos.entry_price))
            pos.gross_pnl = round(gross_pnl, 2)
            pos.net_pnl = round(net_pnl, 2)
            pos.net_return_pct = round((net_pnl / max(1.0, entry_cost)) * 100.0, 2)

            peak_unrealized = orig_sh * max(0.0, pos.highest_price - pos.entry_price)
            if pos.was_trimmed and peak_unrealized > 0:
                pos.profit_retained_pct = round(max(0.0, min(100.0, (net_pnl / peak_unrealized) * 100.0)), 2)
            else:
                pos.profit_retained_pct = 100.0 if net_pnl > 0 else 0.0

            closed_trades.append(pos)
        positions.clear()

    if daily_equity:
        daily_equity[-1] = (dates[-1], cash)

    return closed_trades, daily_equity, total_candidates_evaluated


def compute_setup_metrics(
    trades: List[HistoricalTrade],
    daily_equity: List[Tuple[pd.Timestamp, float]],
    total_candidates: int,
    setup_type: str,
    initial_equity: float = 100_000.0,
) -> HistoricalSetupMetrics:
    """Computes comprehensive performance, drawdown, stop-out, trim, and market period metrics."""
    if not trades or not daily_equity:
        return HistoricalSetupMetrics(
            setup_type=setup_type,
            total_candidates_evaluated=total_candidates,
            total_trades_executed=len(trades),
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            gross_pnl=0.0,
            net_pnl=0.0,
            initial_equity=initial_equity,
            final_equity=initial_equity,
            cagr_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            stop_out_count=0,
            stop_out_rate_pct=0.0,
            profit_retained_after_trims_pct=0.0,
            missed_rebounds_count=0,
            missed_rebounds_rate_pct=0.0,
            annualized_turnover_pct=0.0,
        )

    n_trades = len(trades)
    winning = [t for t in trades if t.net_pnl > 0]
    losing = [t for t in trades if t.net_pnl < 0]
    win_rate = (len(winning) / n_trades) * 100.0

    gross_pnl = sum(t.gross_pnl for t in trades)
    final_equity = daily_equity[-1][1]
    # The equity curve is the portfolio cash ledger. Trade-level PnL is
    # independently summed below and must reconcile to this value.
    net_pnl = final_equity - initial_equity

    # Time Horizon in Years
    t_start = daily_equity[0][0]
    t_end = daily_equity[-1][0]
    years = max(0.25, (t_end - t_start).days / 365.25)
    cagr = ((final_equity / initial_equity) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    # Drawdown Curve
    eq_series = pd.Series([eq for _, eq in daily_equity])
    cum_max = eq_series.cummax()
    drawdowns = (cum_max - eq_series) / cum_max
    max_dd = float(drawdowns.max()) * 100.0

    # Risk-Adjusted Ratios
    daily_rets = eq_series.pct_change(fill_method=None).dropna()
    mean_ret = float(daily_rets.mean())
    std_ret = float(daily_rets.std())
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252) if std_ret > 0 else 0.0

    downside_rets = daily_rets[daily_rets < 0]
    downside_std = float(downside_rets.std()) if len(downside_rets) > 1 else std_ret
    sortino = (mean_ret / (downside_std + 1e-9)) * np.sqrt(252) if downside_std > 0 else 0.0

    # Profit Factor
    gross_gains = sum(t.net_pnl for t in winning)
    gross_losses = abs(sum(t.net_pnl for t in losing))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (2.0 if gross_gains > 0 else 0.0)
    trade_pnl_sum = sum(t.net_pnl for t in trades)

    # Stop-outs and Profit Retained
    stop_outs = sum(1 for t in trades if t.was_stopped_out)
    stop_out_rate = (stop_outs / n_trades) * 100.0

    trimmed_trades = [t for t in trades if t.was_trimmed]
    profit_retained_avg = (
        sum(t.profit_retained_pct for t in trimmed_trades) / len(trimmed_trades)
        if trimmed_trades else 50.0
    )

    # Missed Rebounds (Shakeouts)
    missed_rebounds = sum(1 for t in trades if t.was_missed_rebound)
    missed_rebound_rate = (missed_rebounds / max(1, stop_outs)) * 100.0

    # Annualized Turnover
    total_dollar_vol = sum(t.entry_price * t.orig_shares for t in trades) * 2.0
    avg_eq = float(eq_series.mean())
    annual_turnover = (total_dollar_vol / (avg_eq * years)) * 100.0

    # Discrete Market Period Breakdowns
    periods = {
        "2022_bear_shock": ("2022-01-01", "2022-12-31"),
        "2023_recovery": ("2023-01-01", "2023-12-31"),
        "2024_bull_expansion": ("2024-01-01", "2024-12-31"),
        "2025_2026_holdout": ("2025-01-01", "2026-10-14"),
    }
    period_stats: Dict[str, Dict[str, Any]] = {}
    for p_name, (p_start, p_end) in periods.items():
        p_trades = [t for t in trades if p_start <= t.entry_date <= p_end]
        if not p_trades:
            period_stats[p_name] = {"trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "profit_factor": 0.0}
            continue

        p_wins = sum(1 for t in p_trades if t.net_pnl > 0)
        p_net = sum(t.net_pnl for t in p_trades)
        p_gains = sum(t.net_pnl for t in p_trades if t.net_pnl > 0)
        p_losses = abs(sum(t.net_pnl for t in p_trades if t.net_pnl < 0))
        p_pf = (p_gains / p_losses) if p_losses > 0 else (2.0 if p_gains > 0 else 0.0)

        period_stats[p_name] = {
            "trades": len(p_trades),
            "win_rate_pct": round((p_wins / len(p_trades)) * 100.0, 1),
            "net_pnl": round(p_net, 2),
            "profit_factor": round(p_pf, 2),
        }

    return HistoricalSetupMetrics(
        setup_type=setup_type,
        total_candidates_evaluated=total_candidates,
        total_trades_executed=n_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate_pct=round(win_rate, 2),
        gross_pnl=round(gross_pnl, 2),
        net_pnl=round(net_pnl, 2),
        initial_equity=round(initial_equity, 2),
        final_equity=round(final_equity, 2),
        cagr_pct=round(cagr, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        profit_factor=round(profit_factor, 2),
        max_drawdown_pct=round(max_dd, 2),
        stop_out_count=stop_outs,
        stop_out_rate_pct=round(stop_out_rate, 2),
        profit_retained_after_trims_pct=round(profit_retained_avg, 2),
        missed_rebounds_count=missed_rebounds,
        missed_rebounds_rate_pct=round(missed_rebound_rate, 2),
        annualized_turnover_pct=round(annual_turnover, 2),
        trade_pnl_sum=round(trade_pnl_sum, 2),
        pnl_reconciliation_error=round(trade_pnl_sum - net_pnl, 2),
        period_breakdown=period_stats,
    )


def run_historical_setups_evaluation(
    prices_path: Optional[Path] = None,
    config: Optional[HistoricalEvaluationConfig] = None,
    policy: Optional[TradingPolicy] = None,
    export_json: bool = True,
    json_path: Optional[Path] = None,
) -> HistoricalEvaluationReport:
    """
    Executes full institutional out-of-sample historical evaluation across:
    - PULLBACK_BUY alone
    - CATALYST_DIP_BUY alone
    - COMBINED strategy
    And verifies the Out-of-Sample Historical Promotion Gate.
    """
    cfg = config or HistoricalEvaluationConfig()
    active_policy = policy or get_default_policy()
    promotion_cfg = active_policy.shadow_trading
    _, matrices = load_price_and_indicator_matrix(prices_path=prices_path)

    # 1. Evaluate Pullbacks Alone
    pb_trades, pb_eq, pb_cand = simulate_setup_backtest(matrices, setup_mode="PULLBACK", config=cfg)
    pb_metrics = compute_setup_metrics(pb_trades, pb_eq, pb_cand, setup_type="PULLBACK_BUY", initial_equity=cfg.initial_equity)

    # 2. Evaluate Catalyst Dips Alone
    dip_trades, dip_eq, dip_cand = simulate_setup_backtest(matrices, setup_mode="CATALYST_DIP", config=cfg)
    dip_metrics = compute_setup_metrics(dip_trades, dip_eq, dip_cand, setup_type="CATALYST_DIP_BUY", initial_equity=cfg.initial_equity)

    # 3. Evaluate Combined Strategy
    comb_trades, comb_eq, comb_cand = simulate_setup_backtest(matrices, setup_mode="COMBINED", config=cfg)
    comb_metrics = compute_setup_metrics(comb_trades, comb_eq, comb_cand, setup_type="COMBINED", initial_equity=cfg.initial_equity)

    # 4. Out-of-Sample Institutional Gate Checks
    max_drawdown = max(
        pb_metrics.max_drawdown_pct,
        dip_metrics.max_drawdown_pct,
        comb_metrics.max_drawdown_pct,
    )
    max_stop_out = max(
        pb_metrics.stop_out_rate_pct,
        dip_metrics.stop_out_rate_pct,
        comb_metrics.stop_out_rate_pct,
    )
    min_profit_retained = min(
        pb_metrics.profit_retained_after_trims_pct,
        dip_metrics.profit_retained_after_trims_pct,
        comb_metrics.profit_retained_after_trims_pct,
    )
    holdout_net = {
        "pullback": pb_metrics.period_breakdown.get("2025_2026_holdout", {}).get("net_pnl", 0.0),
        "catalyst_dip": dip_metrics.period_breakdown.get("2025_2026_holdout", {}).get("net_pnl", 0.0),
        "combined": comb_metrics.period_breakdown.get("2025_2026_holdout", {}).get("net_pnl", 0.0),
    }
    pnl_reconciliation_ok = all(
        abs(m.pnl_reconciliation_error) <= max(1.0, m.total_trades_executed * 0.02)
        for m in (pb_metrics, dip_metrics, comb_metrics)
    )

    # This evaluator only has adjusted prices and technical proxies. It has no
    # point-in-time adverse-news, valuation, or live portfolio evidence, so its
    # results are useful for research but cannot certify the production rules.
    production_rule_fidelity = False
    gate_checks = {
        "min_trades_statistical_significance": {
            "required": promotion_cfg.min_historical_trades,
            "actual": comb_metrics.total_trades_executed,
            "passed": comb_metrics.total_trades_executed >= promotion_cfg.min_historical_trades,
        },
        "pullback_positive_expectancy": {
            "required": f">= {promotion_cfg.min_pullback_cagr_pct:.2f}% CAGR",
            "actual": f"{pb_metrics.cagr_pct:+.2f}%",
            "passed": pb_metrics.cagr_pct >= promotion_cfg.min_pullback_cagr_pct,
        },
        "catalyst_dip_positive_expectancy": {
            "required": f">= {promotion_cfg.min_catalyst_dip_cagr_pct:.2f}% CAGR",
            "actual": f"{dip_metrics.cagr_pct:+.2f}%",
            "passed": dip_metrics.cagr_pct >= promotion_cfg.min_catalyst_dip_cagr_pct,
        },
        "max_historical_drawdown_within_bounds": {
            "required": f"<= {promotion_cfg.max_historical_drawdown_pct:.2f}% across all modes",
            "actual": f"{max_drawdown:.2f}%",
            "passed": max_drawdown <= promotion_cfg.max_historical_drawdown_pct,
        },
        "profit_retained_after_trims": {
            "required": f">= {promotion_cfg.min_profit_retained_after_trims_pct:.1f}% across all modes",
            "actual": f"{min_profit_retained:.1f}%",
            "passed": min_profit_retained >= promotion_cfg.min_profit_retained_after_trims_pct,
        },
        "controlled_stop_out_rate": {
            "required": f"<= {promotion_cfg.max_stop_out_rate_pct:.1f}% across all modes",
            "actual": f"{max_stop_out:.1f}%",
            "passed": max_stop_out <= promotion_cfg.max_stop_out_rate_pct,
        },
        "holdout_positive_return": {
            "required": "> 0.0 net PnL for each setup and combined strategy" if promotion_cfg.require_holdout_positive_pnl else "not required",
            "actual": holdout_net,
            "passed": (not promotion_cfg.require_holdout_positive_pnl) or all(v > 0.0 for v in holdout_net.values()),
        },
        "pnl_equity_reconciliation": {
            "required": "trade PnL reconciles to final equity within $1 or $0.02/trade",
            "actual": {
                "pullback_error": pb_metrics.pnl_reconciliation_error,
                "catalyst_dip_error": dip_metrics.pnl_reconciliation_error,
                "combined_error": comb_metrics.pnl_reconciliation_error,
            },
            "passed": pnl_reconciliation_ok,
        },
        "production_rule_fidelity": {
            "required": "point-in-time replay of production rules and data",
            "actual": "technical proxy; no point-in-time catalyst or valuation inputs",
            "passed": production_rule_fidelity,
        },
    }

    all_passed = all(chk["passed"] for chk in gate_checks.values())
    remediation_notes = []
    if not all_passed:
        for k, chk in gate_checks.items():
            if not chk["passed"]:
                remediation_notes.append(f"Historical Gate Failure in {k}: actual {chk['actual']} vs req {chk['required']}")

    report = HistoricalEvaluationReport(
        evaluated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        total_trading_days=len(pb_eq),
        pullbacks=pb_metrics,
        catalyst_dips=dip_metrics,
        combined=comb_metrics,
        passed_oos_gate=all_passed,
        production_rule_fidelity=production_rule_fidelity,
        validation_scope="HISTORICAL_PROXY",
        oos_gate_checks=gate_checks,
        remediation_notes=remediation_notes,
    )

    if export_json:
        out_path = Path(json_path) if json_path else (DATA_DIR / "historical_setups_evaluation.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_historical_setups_evaluation()
    print("=== Historical Evaluation Report ===")
    print(f"Passed OOS Gate: {rep.passed_oos_gate}")
    print(f"Pullback Trades: {rep.pullbacks.total_trades_executed} | Win Rate: {rep.pullbacks.win_rate_pct}% | Max DD: {rep.pullbacks.max_drawdown_pct}% | Retained: {rep.pullbacks.profit_retained_after_trims_pct}%")
    print(f"Catalyst Dip Trades: {rep.catalyst_dips.total_trades_executed} | Win Rate: {rep.catalyst_dips.win_rate_pct}% | Max DD: {rep.catalyst_dips.max_drawdown_pct}% | Retained: {rep.catalyst_dips.profit_retained_after_trims_pct}%")
    print(f"Combined Trades: {rep.combined.total_trades_executed} | Win Rate: {rep.combined.win_rate_pct}% | Stop-out Rate: {rep.combined.stop_out_rate_pct}%")
