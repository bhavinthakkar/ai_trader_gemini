import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import json
from db import (
    init_db,
    get_latest_signals,
    get_signal_history,
    get_summary_stats,
    get_outcome_performance_stats,
    get_latest_portfolio_reviews,
    delete_signals_older_than
)
from dashboard import collect_dashboard_data, format_volume
from ticker_resolver import resolve_symbol, is_european_symbol, get_currency_for_symbol


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Configuration
st.set_page_config(
    page_title="AI Trader - Swing Trading Analysis Report",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom Styling (Responsive & Mobile-Friendly)
st.markdown("""
<style>
    /* Dark Theme Base */
    .main {
        background-color: #0e1117;
    }

    /* Badges */
    .badge-buy {
        background-color: #0e3a24;
        color: #3dd68c;
        border: 1px solid #1c6b45;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-sell {
        background-color: #3b1719;
        color: #f87171;
        border: 1px solid #7f1d1d;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-hold {
        background-color: #27272a;
        color: #a1a1aa;
        border: 1px solid #3f3f46;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Universal Touch-Friendly Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1rem;
        min-height: 44px;
        transition: all 0.2s ease-in-out;
        touch-action: manipulation;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }

    /* Card styling for Metrics across all viewports */
    div[data-testid="stMetric"] {
        background: #111827;
        padding: 0.75rem 0.9rem;
        border-radius: 10px;
        border: 1px solid #1f2937;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 2.5vw, 1.45rem) !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    /* Mobile Swipeable Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        gap: 0.35rem !important;
        padding: 0.25rem 0.1rem 0.6rem 0.1rem !important;
        scrollbar-width: thin !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 3px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        padding: 0.5rem 0.9rem !important;
        font-size: 0.85rem !important;
        border-radius: 8px !important;
    }

    /* Dataframe & Table Horizontal Scroll for Mobile */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* Prominent Sidebar Hamburger Button on Mobile */
    [data-testid="stSidebarCollapsedControl"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.3rem !important;
        top: 0.6rem !important;
        left: 0.6rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Responsive Media Queries for Tablet & Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: 100% !important;
        }

        /* Allow columns in horizontal blocks to wrap on mobile */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }

        /* Columns take 50% width on tablet/mobile (2 per row for metrics) */
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            min-width: 135px !important;
        }

        /* Responsive typography */
        h1 {
            font-size: 1.55rem !important;
            line-height: 1.25 !important;
        }
        h2 {
            font-size: 1.25rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        h4 {
            font-size: 0.98rem !important;
        }

        .hide-on-mobile {
            display: none !important;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        /* Inputs stack full-width on compact phone screens */
        div[data-testid="stTextInput"],
        div[data-testid="stSelectbox"],
        div[data-testid="stMultiSelect"],
        div[data-testid="stSlider"] {
            width: 100% !important;
        }
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
- `python main.py qwen <TICKER>` (Local Qwen 2.5 14B via Vulkan-enabled llama.cpp)
- `python main.py llamacpp <TICKER>` (Alias for the local llama.cpp server)
- `python main.py qwen-llamacpp <TICKER>` (Explicit alias for the local llama.cpp server)

*Example:* `python main.py kimi NVDA` or `python main.py nemotron AAPL`

**Portfolio risk review:** `python main.py portfolio gemini`
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Recency Filter")
recency_options = {
    "Past 7 Days (1 Week)": 7,
    "Past 14 Days (2 Weeks)": 14,
    "Past 30 Days (1 Month)": 30,
    "All Time": None
}
selected_window_label = st.sidebar.selectbox(
    "Signals Recency Window:",
    options=list(recency_options.keys()),
    index=0,
    help="Filter out signals older than the selected timeframe in Latest Signals & KPIs"
)
max_age_days = recency_options[selected_window_label]

with st.sidebar.expander("🗑️ Database Maintenance"):
    st.caption("Permanently purge older historical records from SQLite database (`trader.db`).")
    purge_days = st.number_input("Purge records older than (days):", min_value=1, max_value=365, value=7, step=1)
    if st.button("Permanently Purge Old Signals", type="secondary"):
        deleted_count = delete_signals_older_than(days=int(purge_days))
        st.success(f"Purged {deleted_count} records older than {purge_days} days.")
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Report Data", use_container_width=True):
    st.rerun()

# Main App Layout
st.title("📈 Autonomous Stock Swing Trading Report")
st.caption("Reporting dashboard reading analysis results from SQLite (`trader.db`)")

# Summary KPI Cards
stats = get_summary_stats(max_age_days=max_age_days)
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Latest Signals",
    "🔍 Stock Deep-Dive",
    "🔥 Most Traded Stocks",
    "📜 Signal History Log",
    "🎯 Model Outcome Tracking",
    "📁 Portfolio Risk Review"
])

@st.cache_data(ttl=120)
def get_cached_active_stocks(limit: int = 10, market: str = "US", prefer_exchange: str = "DE"):
    return collect_dashboard_data(limit=limit, market=market, prefer_exchange=prefer_exchange)

latest_signals = get_latest_signals(max_age_days=max_age_days)

with tab1:
    st.subheader("Latest Swing Trade Recommendations")
    if max_age_days:
        st.caption(f"📅 Displaying active recommendations analyzed within the **{selected_window_label}** (older analyses filtered out).")
    else:
        st.caption("📅 Displaying latest recommendation per stock for **All Time**.")

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

    # On-demand ISIN & European Ticker Resolver / Analyzer
    st.markdown("##### 🔎 ISIN / European Ticker Resolver & Deep-Dive")
    lookup_col1, lookup_col2 = st.columns([3, 1])
    with lookup_col1:
        isin_query = st.text_input(
            "Enter ISIN, WKN, or Symbol (e.g. US67066G1040, DE0007164600, 716460, NVD.DE, SAP.DE):",
            placeholder="Paste European ISIN or ticker...",
            key="isin_lookup_input"
        ).strip().upper()
    with lookup_col2:
        st.markdown("<div class='hide-on-mobile' style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_isin_btn = st.button("🚀 Analyze with AI", key="analyze_isin_button", use_container_width=True)

    symbol_list = [s["symbol"] for s in latest_signals] if latest_signals else []

    if isin_query:
        resolved_info = resolve_symbol(isin_query, prefer_exchange="DE", force_european=True)
        res_sym = resolved_info["symbol"]
        curr_code, curr_sym_lookup = get_currency_for_symbol(res_sym)
        isin_str = f" | **ISIN:** `{resolved_info['isin']}`" if resolved_info.get("isin") else ""
        st.info(
            f"**Resolved Symbol:** `{res_sym}` | **Company:** {resolved_info.get('company_name', res_sym)} | "
            f"**Currency:** {curr_code} ({curr_sym_lookup}) | **Exchange:** {resolved_info.get('exchange', 'gettex / XETRA')}{isin_str}"
        )

        if analyze_isin_btn:
            with st.status(f"🚀 Running 6-Agent AI Swing Synthesis on {res_sym}...", expanded=True) as status_box:
                st.write(f"Launching main.py for {res_sym}...")
                main_script = os.path.join(BASE_DIR, "main.py")
                cmd = [sys.executable, main_script, "free", res_sym, "--eu"]
                try:
                    proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
                    if proc.returncode == 0:
                        status_box.update(label=f"✅ Analysis Complete for {res_sym}!", state="complete", expanded=False)
                        st.success(f"Analysis complete for {res_sym}! Signal recorded to database.")
                        latest_signals = get_latest_signals(max_age_days=max_age_days)
                        symbol_list = [s["symbol"] for s in latest_signals]
                    else:
                        status_box.update(label=f"⚠️ Pipeline finished with exit code {proc.returncode}", state="error", expanded=True)
                        st.code(proc.stderr or proc.stdout, language="text")
                except Exception as e:
                    status_box.update(label=f"❌ Failed: {e}", state="error")
                    st.error(f"Execution error: {e}")

        if res_sym not in symbol_list:
            symbol_list.insert(0, res_sym)

    if not symbol_list:
        st.info("No signal data available. Enter an ISIN or ticker above, or run `python main.py` in your terminal.")
    else:
        selected_stock = st.selectbox("Select Ticker Symbol to Inspect:", options=symbol_list)
        curr_code, curr_sym = get_currency_for_symbol(selected_stock)

        stock_data = next((s for s in latest_signals if s["symbol"] == selected_stock), None) if latest_signals else None

        if stock_data:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                dec = stock_data["decision"]
                badge_class = "badge-buy" if dec == "BUY" else ("badge-sell" if dec == "SELL" else "badge-hold")
                st.markdown(f"### **{stock_data['symbol']}**")
                st.markdown(f"**Decision:** <span class='{badge_class}'>{dec}</span>", unsafe_allow_html=True)
                ntr = stock_data.get("no_trade_reason")
                if ntr:
                    st.markdown(f"**No-Trade Reason:** `{ntr}`")
            with c2:
                conf = stock_data.get("confidence", 0.0)
                st.metric("Model Confidence", f"{conf * 100:.0f}%" if conf else "N/A")
                st.progress(min(max(float(conf or 0.0), 0.0), 1.0))
            with c3:
                q_score = stock_data.get("quant_score")
                st.metric("Deterministic Quant Score", f"{q_score}/100" if q_score is not None else "N/A")
            with c4:
                entry_p = stock_data.get("entry_price")
                st.metric("Entry Price", f"{curr_sym}{entry_p:.2f}" if entry_p else "N/A")

            # 5-Pillar Quantitative Scores Display with Visible Missingness
            if any(stock_data.get(k) is not None for k in ["trend_score", "sector_score", "alpha_score", "val_history_score", "peer_val_score"]):
                st.markdown("#### 🧮 5-Pillar Quantitative Scores & Data Availability")
                p_status = {}
                raw_st = stock_data.get("pillar_status")
                if isinstance(raw_st, str):
                    try:
                        p_status = json.loads(raw_st)
                    except Exception:
                        pass
                elif isinstance(raw_st, dict):
                    p_status = raw_st

                def _fmt_pillar(score, st_name):
                    status = p_status.get(st_name, "measured" if score is not None else "unavailable")
                    score_str = f"{score}" if score is not None else "UNAVAILABLE"
                    return f"{score_str} ({status})"

                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Trend (25%)", _fmt_pillar(stock_data.get('trend_score'), "trend"))
                p2.metric("Sector Rel (20%)", _fmt_pillar(stock_data.get('sector_score'), "sector"))
                p3.metric("Market Alpha (20%)", _fmt_pillar(stock_data.get('alpha_score'), "alpha"))
                p4.metric("Val History (15%)", _fmt_pillar(stock_data.get('val_history_score'), "valuation_history"))
                p5.metric("Peer Val (20%)", _fmt_pillar(stock_data.get('peer_val_score'), "peer_valuation"))

            # Financial Quality & Balance-Sheet Solvency
            fq_sc = stock_data.get("financial_quality_score")
            raw_j = stock_data.get("raw_json")
            fq_dict = {}
            if isinstance(raw_j, str):
                try:
                    fq_dict = json.loads(raw_j).get("financial_quality", {}).get("metrics", {})
                except Exception:
                    pass
            elif isinstance(stock_data.get("financial_quality"), dict):
                fq_dict = stock_data.get("financial_quality", {}).get("metrics", {})

            if fq_sc is not None or fq_dict:
                st.markdown("#### 💎 Financial Quality & Balance-Sheet Solvency")
                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("Quality Score", f"{fq_sc}/100" if fq_sc is not None else "N/A")
                fcf_m = fq_dict.get("fcf_margin")
                q2.metric("FCF Margin", f"{fcf_m:.1f}%" if fcf_m is not None else "N/A")
                op_m = fq_dict.get("operating_margin")
                q3.metric("Operating Margin", f"{op_m:.1f}%" if op_m is not None else "N/A")
                lev = fq_dict.get("net_debt_to_ebitda")
                q4.metric("Net Debt/EBITDA", f"{lev:.1f}x" if (lev is not None and lev > 0) else ("Net Cash" if lev == 0 else "N/A"))
                eq_r = fq_dict.get("earnings_quality_ratio")
                q5.metric("OCF/Net Income", f"{eq_r:.2f}x" if eq_r is not None else "N/A")

            # Technical & Macro Snapshot Row
            st.markdown("#### 📈 Execution & Macro Parameters")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Stop Loss", f"{curr_sym}{stock_data.get('stop_loss_price', 'N/A')}")
            m2.metric("Target Price", f"{curr_sym}{stock_data.get('target_price', 'N/A')}")
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
            g4.metric("Structural Stop", f"{curr_sym}{stock_data.get('structural_stop_price', 'N/A')}")
            g5.metric("Structural Target", f"{curr_sym}{stock_data.get('structural_target_price', 'N/A')}")

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
    st.subheader("🔥 Most Traded Stocks Today (Price, Range, Volume & Catalysts)")
    st.caption("Real-time high-volume market movers and the news catalysts driving heavy trading activity (model-free).")

    top_ctrl1, top_ctrl2, top_ctrl3 = st.columns([1.5, 1.2, 1.3])
    with top_ctrl1:
        market_choice = st.radio(
            "Market Selection:",
            options=["🇺🇸 US Markets (NYSE / NASDAQ)", "🇪🇺 Europe (gettex / XETRA)"],
            index=0,
            horizontal=True,
            help="Switch between US high-volume movers ($) and European gettex/XETRA traded NASDAQ leaders (€)"
        )
        is_eu = "Europe" in market_choice
        selected_market = "EU" if is_eu else "US"
        selected_exchange = "DE"
        if is_eu:
            exchange_opt = st.selectbox(
                "Exchange Preference:",
                options=["DE (XETRA Reference)", "MU (gettex / Börse München)", "HA (EIX / Börse Hannover)", "TG (Tradegate)"],
                index=0,
                help="gettex and EIX follow the German Referenzmarkt-Prinzip based on XETRA (.DE) during market hours"
            )
            selected_exchange = exchange_opt.split(" ")[0].strip()

    with top_ctrl2:
        limit_val = st.selectbox("Number of Active Stocks:", options=[10, 15, 20, 25], index=0)
        analyse_count = st.selectbox(
            "AI Analysis Count:",
            options=[3, 5, 10],
            index=1,
            help="Number of top active stocks to sequentially analyze with main.py using AI preset ('free')"
        )
    with top_ctrl3:
        if st.button("🔄 Refresh Active Data", key="refresh_active_stocks", use_container_width=True):
            get_cached_active_stocks.clear()
            st.rerun()

        run_analyse = st.button(
            "🚀 Refresh active data & Analyse",
            key="refresh_and_analyse_stocks",
            help="Refreshes active stocks data and executes master.py using AI preset (free)",
            use_container_width=True,
            type="primary"
        )

    if run_analyse:
        get_cached_active_stocks.clear()
        master_script = os.path.join(BASE_DIR, "master.py")
        cmd = [
            sys.executable,
            master_script,
            "free",
            "--limit", str(analyse_count),
            "--dashboard-limit", str(limit_val),
            "--market", selected_market,
            "--prefer-exchange", selected_exchange,
            "--no-streamlit",
        ]
        status_market_label = f"Europe (gettex / XETRA: {selected_exchange})" if is_eu else "US Markets"
        with st.status(f"🚀 Running Gloomberb Master Orchestration ({status_market_label} | Model: free, Limit: {analyse_count})...", expanded=True) as status_box:
            st.write(f"Initializing active volume screener [{selected_market}] & launching 6-agent AI swing analysis...")
            log_box = st.empty()
            logs = []
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    clean = line.strip()
                    if clean:
                        logs.append(clean)
                        log_box.code("\n".join(logs[-15:]), language="text")
                        if "COMMENCING SEQUENTIAL" in clean:
                            status_box.update(label="⚡ Executing 6-Agent AI Swing Synthesis per stock...")
                        elif "Launching main.py for" in clean:
                            st.write(f"▶ {clean}")
                        elif "Completed" in clean and "Decision:" in clean:
                            st.write(f"✅ {clean}")
                proc.stdout.close()
                rc = proc.wait()
                if rc == 0:
                    status_box.update(label=f"✅ Master Analysis Complete for Top {analyse_count} Stocks!", state="complete", expanded=False)
                    st.success(f"Successfully analyzed top {analyse_count} active stocks! Trade signals saved to database.")
                else:
                    status_box.update(label=f"⚠️ Pipeline finished with exit code {rc}", state="error", expanded=True)
                    st.warning(f"Master script exited with code {rc}. Check details in log above.")
            except Exception as e:
                status_box.update(label=f"❌ Execution failed: {e}", state="error")
                st.error(f"Failed to execute master script: {e}")

    with st.spinner(f"Fetching real-time active stocks [{selected_market}] and news catalysts..."):
        active_stocks = get_cached_active_stocks(limit=limit_val, market=selected_market, prefer_exchange=selected_exchange)

    if not active_stocks:
        st.warning("No active stocks data available at this time.")
    else:
        tot_vol = sum(s.get("volume", 0) for s in active_stocks)
        vol_leader = active_stocks[0]["symbol"] if active_stocks else "N/A"
        top_gainer = max(active_stocks, key=lambda x: x.get("change_pct", 0.0))
        top_loser = min(active_stocks, key=lambda x: x.get("change_pct", 0.0))

        ak1, ak2, ak3, ak4 = st.columns(4)
        ak1.metric("Active Stocks Tracked", f"{len(active_stocks)} Stocks")
        ak2.metric("Combined Volume", format_volume(tot_vol))
        ak3.metric("Top Active Gainer", f"{top_gainer['symbol']} ({top_gainer['change_pct']:+.2f}%)")
        ak4.metric("Top Active Decliner", f"{top_loser['symbol']} ({top_loser['change_pct']:+.2f}%)")

        st.markdown("---")

        table_rows = []
        for s in active_stocks:
            curr_sym = s.get("currency_symbol", "€" if is_eu else "$")
            chg = s.get("change_pct", 0.0)
            sign = "+" if chg >= 0 else ""
            table_rows.append({
                "Ticker": s["symbol"],
                "Company": s["name"],
                "Price": f"{curr_sym}{s['price']:.2f}",
                "Change": f"{sign}{chg:.2f}% ({sign}{curr_sym}{abs(s['change']):.2f})",
                "Day Low": f"{curr_sym}{s['day_low']:.2f}",
                "Day High": f"{curr_sym}{s['day_high']:.2f}",
                "Volume": format_volume(s["volume"]),
                "RVOL": f"{s['rvol']:.1f}x",
                "52W Range": f"{curr_sym}{s['low_52w']:.1f} - {curr_sym}{s['high_52w']:.1f}" if s.get('low_52w') else "N/A",
                "RSI(14)": f"{s['rsi14']:.1f}" if s.get('rsi14') is not None else "N/A",
                "Catalyst Driver": s["catalyst_type"],
                "Trading Reason": s["reason_summary"]
            })

        df_active = pd.DataFrame(table_rows)
        st.dataframe(df_active, width="stretch", hide_index=True)

        st.markdown("#### 📰 Why Are They Trading So Much Today? (News & Catalyst Drilldown)")
        for s in active_stocks:
            curr_sym = s.get("currency_symbol", "€" if is_eu else "$")
            chg_sign = "+" if s.get("change_pct", 0.0) >= 0 else ""
            badge = "🟢" if s.get("change_pct", 0.0) >= 0 else "🔴"
            with st.expander(f"{badge} **{s['symbol']}** ({s['name']}) — {curr_sym}{s['price']:.2f} ({chg_sign}{s['change_pct']:.2f}%) | Vol: {format_volume(s['volume'])} ({s['rvol']:.1f}x 20d avg)"):
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Day Low / High", f"{curr_sym}{s['day_low']:.2f} - {curr_sym}{s['day_high']:.2f}")
                m_c2.metric("Market Cap", s["market_cap_str"])
                m_c3.metric("RVOL (20d avg)", f"{s['rvol']:.1f}x")
                m_c4.metric("RSI14 / ATR", f"{s.get('rsi14', 'N/A')} / {curr_sym}{s.get('atr', 'N/A')}")

                st.markdown(f"**⚡ Catalyst Tag:** `{s['catalyst_type']}`")
                st.info(f"**Reason for High Volume:** {s['reason_summary']}")

                news_list = s.get("news", [])
                if news_list:
                    st.markdown("**Recent News Headlines:**")
                    for n in news_list:
                        pub = f"[{n['publisher']}] " if n.get('publisher') else ""
                        url = n.get('url', '#')
                        st.markdown(f"- {pub}[{n['title']}]({url})")



with tab4:
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

with tab5:
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

with tab6:
    st.subheader("📁 Portfolio Risk Review (Investment-Committee Memo)")
    st.caption("Stored portfolio risk memos generated via `python main.py portfolio <model>`.")

    portfolio_reviews = get_latest_portfolio_reviews(limit=20)

    if not portfolio_reviews:
        st.info("No portfolio reviews found. Run `python main.py portfolio gemini` in your terminal to generate one.")
    else:
        for rev in portfolio_reviews:
            with st.expander(f"🕐 {rev['timestamp']} — Model: {rev.get('model_used', 'N/A')}"):
                try:
                    review = json.loads(rev.get("review_json") or "{}")
                except Exception:
                    review = {}
                portfolio = {}
                try:
                    portfolio = json.loads(rev.get("portfolio_json") or "{}")
                except Exception:
                    pass

                if not review:
                    st.warning("Review payload is empty or unparsable.")
                    if rev.get("raw_json"):
                        st.json(rev["raw_json"])
                    continue

                summary = review.get("portfolio_summary") or "No summary provided."
                st.markdown(f"**Overview:** {summary}")

                if portfolio:
                    st.markdown("#### 💼 Supplied Portfolio")
                    st.json(portfolio)

                c_risks = review.get("concentration_risks")
                if c_risks:
                    st.markdown("#### ⚠️ Concentration Risks")
                    if isinstance(c_risks, list):
                        for r in c_risks:
                            if isinstance(r, dict):
                                sev = str(r.get("severity", ""))
                                emoji = "🔴" if sev == "HIGH" else ("🟡" if sev == "MEDIUM" else "🟢")
                                st.markdown(f"- {emoji} **{r.get('risk', 'N/A')}** `{sev}` — {r.get('evidence', '')}")
                                if r.get("holdings"):
                                    st.caption(f"Holdings: {', '.join(str(h) for h in r['holdings'])}")
                    else:
                        st.write(c_risks)

                exp_map = review.get("exposure_map")
                if exp_map and isinstance(exp_map, dict):
                    st.markdown("#### 🗺️ Exposure Map")
                    for key, val in exp_map.items():
                        if val:
                            st.caption(f"**{key.replace('_', ' ').title()}:** {', '.join(str(x) for x in val) if isinstance(val, list) else val}")

                s_tests = review.get("stress_tests")
                if s_tests:
                    st.markdown("#### 🧪 Stress Tests")
                    if isinstance(s_tests, list):
                        for t in s_tests:
                            if isinstance(t, dict):
                                st.markdown(f"- **{t.get('scenario', 'N/A')}** — {t.get('likely_impact', 'N/A')}")
                                if t.get("most_exposed"):
                                    st.caption(f"Most exposed: {', '.join(str(x) for x in t['most_exposed'])}")
                                if t.get("assumptions_and_limits"):
                                    st.caption(f"Limits: {', '.join(str(x) for x in t['assumptions_and_limits'])}")
                    else:
                        st.write(s_tests)

                d_gaps = review.get("diversification_gaps")
                if d_gaps:
                    st.markdown("#### 🤔 Diversification Gaps")
                    if isinstance(d_gaps, list):
                        for g in d_gaps:
                            st.markdown(f"- {g}")
                    else:
                        st.write(d_gaps)

                r_options = review.get("resilience_options")
                if r_options:
                    st.markdown("#### 🛡️ Resilience Options")
                    if isinstance(r_options, list):
                        for o in r_options:
                            if isinstance(o, dict):
                                st.markdown(f"- **{o.get('possible_change', 'N/A')}** — reduces {o.get('risk_reduced', 'risk')} (trade-off: {o.get('trade_off', 'N/A')})")
                    else:
                        st.write(r_options)

                m_info = review.get("missing_information")
                if m_info:
                    st.markdown("#### ❓ Missing Information")
                    if isinstance(m_info, list):
                        st.caption(", ".join(str(x) for x in m_info))
                    else:
                        st.caption(str(m_info))

                with st.expander("🔍 View Full JSON Review"):
                    st.json(review)

