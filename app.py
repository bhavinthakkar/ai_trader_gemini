import streamlit as st
import pandas as pd
import json
from db import init_db, get_latest_signals, get_signal_history, get_summary_stats, get_outcome_performance_stats

# Page Configuration
st.set_page_config(
    page_title="AI Trader - Swing Trading Analysis Report",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .badge-buy {
        background-color: #0e3a24;
        color: #3dd68c;
        border: 1px solid #1c6b45;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-sell {
        background-color: #3b1719;
        color: #f87171;
        border: 1px solid #7f1d1d;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-hold {
        background-color: #27272a;
        color: #a1a1aa;
        border: 1px solid #3f3f46;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Ensure DB is initialized
init_db()

# Sidebar Setup
st.sidebar.title("⚡ AI Trader Report Viewer")
st.sidebar.markdown("---")

st.sidebar.info("""
💻 **Terminal Trigger Mode (6-Agent System)**
To run a new pipeline analysis, execute in your terminal:
- `python main.py nemotron <TICKER>` (Nemotron-3 Ultra 550B / Super 120B)
- `python main.py kimi <TICKER>` (Moonshot AI Kimi-K3 via NVIDIA)
- `python main.py gemini <TICKER>` (Gemini 3.1 Pro)
- `python main.py openrouter <TICKER>` (OpenRouter Free Models Router - openrouter/free)
- `python main.py twostage <TICKER>` (2-Stage: Qwen2.5 14B + Qwen3 30B)
- `python main.py gemma <TICKER>` (Local gemma4:12b)
- `python main.py qwen <TICKER>` (Local Qwen2.5 14B)

*Example:* `python main.py kimi NVDA` or `python main.py nemotron AAPL`
""")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Report Data", width="stretch"):
    st.rerun()

# Main App Layout
st.title("📈 Autonomous Stock Swing Trading Report")
st.caption("Reporting dashboard reading analysis results from SQLite (`trader.db`)")

# Summary KPI Cards
stats = get_summary_stats()
outcome_stats = get_outcome_performance_stats()
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(label="Tracked Stocks", value=stats["total_tracked"])
with col2:
    st.metric(label="🟢 Buy Signals", value=stats["buy_count"])
with col3:
    st.metric(label="🔴 Sell Signals", value=stats["sell_count"])
with col4:
    st.metric(label="⚪ Hold Signals", value=stats["hold_count"])
with col5:
    win_val = f"{outcome_stats['win_rate_pct']}%" if outcome_stats["total_evaluated"] > 0 else "Pending"
    st.metric(label="🎯 Model Win Rate", value=win_val)
with col6:
    st.metric(label="Last Analysis Run", value=stats["last_run"])

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Latest Signals",
    "🔍 Stock Deep-Dive",
    "📜 Signal History Log",
    "🎯 Model Outcome Tracking"
])

latest_signals = get_latest_signals()

with tab1:
    st.subheader("Latest Swing Trade Recommendations")

    if not latest_signals:
        st.info("No signal data found in database. Run `python main.py` in your terminal to populate analysis reports!")
    else:
        # Filters
        filter_col1, filter_col2 = st.columns([2, 2])
        with filter_col1:
            decision_filter = st.multiselect(
                "Filter by Decision:",
                options=["BUY", "SELL", "HOLD"],
                default=["BUY", "SELL", "HOLD"]
            )
        with filter_col2:
            search_symbol = st.text_input("Search Symbol:", "").strip().upper()

        filtered_signals = [
            s for s in latest_signals
            if s["decision"] in decision_filter and (not search_symbol or search_symbol in s["symbol"])
        ]

        if filtered_signals:
            df_display = pd.DataFrame(filtered_signals)

            cols_to_use = ["symbol", "decision", "confidence"]
            col_renames = {
                "symbol": "Ticker",
                "decision": "Decision",
                "confidence": "Confidence"
            }

            if "quant_score" in df_display.columns and df_display["quant_score"].notna().any():
                cols_to_use.append("quant_score")
                col_renames["quant_score"] = "Quant Score"

            if "entry_price" in df_display.columns and df_display["entry_price"].notna().any():
                cols_to_use.append("entry_price")
                col_renames["entry_price"] = "Entry Price"

            if "rvol_20d" in df_display.columns and df_display["rvol_20d"].notna().any():
                cols_to_use.append("rvol_20d")
                col_renames["rvol_20d"] = "RVOL"

            if "reward_risk_ratio" in df_display.columns and df_display["reward_risk_ratio"].notna().any():
                cols_to_use.append("reward_risk_ratio")
                col_renames["reward_risk_ratio"] = "Reward:Risk"

            if "us_10y_yield" in df_display.columns and df_display["us_10y_yield"].notna().any():
                cols_to_use.append("us_10y_yield")
                col_renames["us_10y_yield"] = "10Y Yield"

            cols_to_use.extend(["pe_and_peg", "model_used", "timestamp"])
            col_renames["pe_and_peg"] = "Fwd Valuation"
            col_renames["model_used"] = "Model"
            col_renames["timestamp"] = "Timestamp"

            df_table = df_display[cols_to_use].rename(columns=col_renames)

            def format_decision(val):
                if val == "BUY":
                    return "🟢 BUY"
                elif val == "SELL":
                    return "🔴 SELL"
                return "⚪ HOLD"

            df_table["Decision"] = df_table["Decision"].apply(format_decision)
            st.dataframe(df_table, width="stretch", hide_index=True)
        else:
            st.warning("No signals match the selected filters.")

with tab2:
    st.subheader("Deterministic 5-Pillar & Quantitative Synthesis Breakdown")

    if not latest_signals:
        st.info("No data available. Run `python main.py` in terminal to generate stock details.")
    else:
        symbol_list = [s["symbol"] for s in latest_signals]
        selected_stock = st.selectbox("Select Ticker Symbol to Inspect:", options=symbol_list)

        stock_data = next((s for s in latest_signals if s["symbol"] == selected_stock), None)

        if stock_data:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                dec = stock_data["decision"]
                badge_class = "badge-buy" if dec == "BUY" else ("badge-sell" if dec == "SELL" else "badge-hold")
                st.markdown(f"### **{stock_data['symbol']}**")
                st.markdown(f"**Decision:** <span class='{badge_class}'>{dec}</span>", unsafe_allow_html=True)
            with c2:
                conf = stock_data.get("confidence", 0.0)
                st.metric("Model Confidence", f"{conf * 100:.0f}%" if conf else "N/A")
                st.progress(min(max(float(conf or 0.0), 0.0), 1.0))
            with c3:
                q_score = stock_data.get("quant_score")
                st.metric("Deterministic Quant Score", f"{q_score}/100" if q_score is not None else "N/A")
            with c4:
                entry_p = stock_data.get("entry_price")
                st.metric("Entry Price", f"${entry_p:.2f}" if entry_p else "N/A")

            # 5-Pillar Quantitative Scores Display
            if any(stock_data.get(k) is not None for k in ["trend_score", "sector_score", "alpha_score", "val_history_score", "peer_val_score"]):
                st.markdown("#### 🧮 5-Pillar Quantitative Scores")
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Trend (25%)", f"{stock_data.get('trend_score', 'N/A')}")
                p2.metric("Sector Rel (20%)", f"{stock_data.get('sector_score', 'N/A')}")
                p3.metric("Market Alpha (20%)", f"{stock_data.get('alpha_score', 'N/A')}")
                p4.metric("Val History (15%)", f"{stock_data.get('val_history_score', 'N/A')}")
                p5.metric("Peer Val (20%)", f"{stock_data.get('peer_val_score', 'N/A')}")

            # Technical & Macro Snapshot Row
            st.markdown("#### 📈 Execution & Macro Parameters")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Stop Loss", f"${stock_data.get('stop_loss_price', 'N/A')}")
            m2.metric("Target Price", f"${stock_data.get('target_price', 'N/A')}")
            m3.metric("RSI14", f"{stock_data.get('rsi14', 'N/A')}")
            m4.metric("RVOL (20d)", f"{stock_data.get('rvol_20d', 'N/A')}x")
            m5.metric("US 10Y Yield", f"{stock_data.get('us_10y_yield', 'N/A')}%")
            m6.metric("Days to Earnings", f"{stock_data.get('days_to_earnings', 'N/A')}d")

            # Reward:Risk setup geometry
            st.markdown("#### ⚖️ Reward:Risk & Setup Geometry")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Reward:Risk", stock_data.get('reward_risk_ratio', 'N/A'))
            g2.metric("Breakeven Win Rate", f"{stock_data.get('breakeven_win_rate', 0) * 100:.0f}%" if stock_data.get("breakeven_win_rate") is not None else "N/A")
            g3.metric("Wall St Target RR", stock_data.get('analyst_target_rr', 'N/A'))
            g4.metric("Structural Stop", f"${stock_data.get('structural_stop_price', 'N/A')}")
            g5.metric("Structural Target", f"${stock_data.get('structural_target_price', 'N/A')}")

            # Volatility risk profile (deterministic ATR dampener)
            st.markdown("#### 🎢 Volatility Risk Profile")
            v1, v2, v3 = st.columns(3)
            vf = stock_data.get("vol_factor") or stock_data.get("vol_factor", 1.0)
            atr_p = stock_data.get("atr_pct")
            v1.metric("Vol Factor", f"{vf}")
            v2.metric("ATR (% of price)", f"{atr_p}%" if atr_p is not None else "N/A")
            v3.metric("Raw Composite", stock_data.get('raw_composite', 'N/A'))

            # Earnings calendar & mechanical gate status
            dte = stock_data.get("days_to_earnings")
            gate_armed = (dte is not None and dte <= 3)
            gate_label = f"ARMED ({dte}d) -- BUY mechanically capped to HOLD" if gate_armed else (
                f"{dte} day(s)" if dte is not None else "N/A"
            )
            st.markdown(
                f"**📅 Earnings Calendar:** `{gate_label}`"
                + (" ⚠️" if gate_armed else "")
            )

            # Decision origin & falsification
            driver = stock_data.get("primary_driver") or "QUANT_STRUCTURE"
            fb = stock_data.get("falsification_bear")
            fbu = stock_data.get("falsification_bull")
            if driver or fb or fbu:
                st.markdown("#### 🧭 Decision Origin & Falsification")
                c_drv, c_conf = st.columns([2, 1])
                c_drv.metric("Primary Driver", f"`{driver}`")
                mc = stock_data.get("model_confidence")
                c_conf.metric(
                    "Mechanistic Conf",
                    f"{stock_data.get('confidence', 'N/A')}",
                    help=f"Model's stated confidence: {mc}" if mc is not None else "Model confidence unavailable"
                )
                if fbu:
                    st.markdown(f"**Falsifies buy-side:** _{fbu}_")
                if fb:
                    st.markdown(f"**Falsifies bear-side:** _{fb}_")

            st.markdown("---")

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("💡 Swing Setup Rationale (Bull Case)")
                st.info(stock_data.get("bull_case") or stock_data.get("reason") or "No rationale provided.")

                st.subheader("🏛️ SEC EDGAR Institutional & Filings")
                st.write(stock_data.get("institutional_data") or "No SEC filing summary available.")

                st.subheader("🌐 Macro Economy (FRED) & CFTC COT")
                st.write(stock_data.get("macro_data") or stock_data.get("marco_data") or "No macro summary available.")

                st.subheader("📰 Today's News Catalysts")
                st.write(stock_data.get("news") or "No news catalyst summary reported.")

            with col_right:
                st.subheader("⚠️ Downside Risks & Bear Case")
                st.warning(stock_data.get("bear_case") or stock_data.get("risk_assessment") or "Low risk profile.")

                st.subheader("🏦 Wall Street Bank Coverage")
                st.write(stock_data.get("investment_bank_coverage") or "No bank rating changes recorded.")

                st.subheader("📊 Forward Valuation")
                st.write(f"Forward P/E Ratio: `{stock_data.get('pe_and_peg', 'N/A')}`")

                if stock_data.get("missing_information"):
                    st.subheader("❓ Missing Information")
                    st.caption(stock_data.get("missing_information"))

            with st.expander("🛠️ View Full JSON Payload"):
                st.json(stock_data.get("raw_json") or json.dumps(stock_data))

with tab3:
    st.subheader("Historical Signal Analysis Log")
    history_records = get_signal_history(limit=200)

    if not history_records:
        st.info("No historical records logged yet.")
    else:
        df_hist = pd.DataFrame(history_records)[[
            "id", "timestamp", "symbol", "decision", "confidence", "model_used", "reason"
        ]].rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "symbol": "Ticker",
            "decision": "Decision",
            "confidence": "Confidence",
            "model_used": "Model Used",
            "reason": "Swing Setup Reason"
        })
        st.dataframe(df_hist, width="stretch", hide_index=True)

with tab4:
    st.subheader("🎯 Model Ground-Truth Outcome Tracking & Evaluation")
    st.caption("Tracks how predictions performed over forward 1-to-10 trading days.")

    from db import update_signal_outcomes, get_connection

    if st.button("⚡ Evaluate Forward Outcomes Now"):
        with st.spinner("Checking market bars and evaluating forward trade outcomes..."):
            count = update_signal_outcomes()
            st.success(f"Evaluated {count} pending signal outcomes!")
            st.rerun()

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Total Trades Evaluated", outcome_stats["total_evaluated"])
    p_col2.metric("Win Rate", f"{outcome_stats['win_rate_pct']}%")
    p_col3.metric("Average Return", f"{outcome_stats['avg_return_pct']:+.2f}%")
    p_col4.metric("Profitable / Losing", f"{outcome_stats['profitable_trades']} / {outcome_stats['losing_trades']}")

    # Table of evaluated outcomes
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT o.id, s.timestamp as signal_date, s.symbol, s.decision,
                   o.horizon_days, o.entry_price, o.exit_price,
                   o.realized_return_pct, o.max_runup_pct, o.max_drawdown_pct,
                   o.hit_target, o.hit_stop, o.is_profitable
            FROM signal_outcomes o
            JOIN signals s ON o.signal_id = s.id
            ORDER BY o.id DESC;
        """)
        outcomes_rows = [dict(r) for r in cursor.fetchall()]

    if outcomes_rows:
        df_outcomes = pd.DataFrame(outcomes_rows)
        df_outcomes["is_profitable"] = df_outcomes["is_profitable"].apply(lambda x: "🟢 Win" if x else "🔴 Loss")
        st.dataframe(df_outcomes, width="stretch", hide_index=True)
    else:
        st.info("No trade outcomes evaluated yet. Signals need at least 1-10 trading days elapsed to compare against historical market bars.")

