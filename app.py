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
    delete_signals_older_than,
    get_shadow_decisions
)
from dashboard import collect_dashboard_data, format_volume
from ticker_resolver import resolve_symbol, is_european_symbol, get_currency_for_symbol
from ml_signal_service import classify_ml_signal, get_ml_context, get_all_ml_scores
from quant.order_execution import GLOBAL_KILL_SWITCH, KillSwitchReason
from quant.compliance_gates import (
    is_signal_vetoed,
    format_compliance_gate_action,
    format_model_recommendation,
    format_executed_action,
    NON_VETO_REASON_CODES,
)
from quant.shadow_trading import (
    ShadowDecisionRecord,
    ShadowMetricsAnalyzer,
    ShadowPromotionEvaluator,
)


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

    /* Standalone ML Alpha Badges */
    .badge-ml-bull {
        background-color: #052e16;
        color: #4ade80;
        border: 1px solid #166534;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-ml-neutral {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-ml-bear {
        background-color: #450a0a;
        color: #f87171;
        border: 1px solid #991b1b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-ml-top10 {
        background-color: #422006;
        color: #fde047;
        border: 1px solid #ca8a04;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-open-long {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-no-action {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-gate-pass {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #10b981;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        display: inline-block;
    }
    .badge-gate-fail {
        background-color: #450a0a;
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
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
- `python main.py flint <TICKER>` (Recursive: Flint (free) - default)
- `python main.py openrouter <TICKER>` (Optional: OpenRouter Free Models Router - openrouter/free)
- `python main.py nemotron <TICKER>` (Nemotron-3 Ultra 550B / Super 120B)
- `python main.py kimi <TICKER>` (Moonshot AI Kimi-K3 via NVIDIA)
- `python main.py gemini <TICKER>` (Gemini 3.1 Pro)
- `python main.py qwen <TICKER>` (Local Qwen 2.5 14B via Vulkan-enabled llama.cpp)
- `python main.py llamacpp <TICKER>` (Alias for the local llama.cpp server)
- `python main.py qwen-llamacpp <TICKER>` (Explicit alias for the local llama.cpp server)

*Example:* `python main.py flint NVDA` or `python main.py flint AAPL`

**Portfolio risk review:** `python main.py portfolio flint`
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

# Danelfin API Free Tier Quota Meter
try:
    from danelfin_service import DanelfinService
    dan_usage = DanelfinService().get_usage()
    used = dan_usage.get("call_count", 0)
    limit = dan_usage.get("monthly_limit", 500)
    pct = min(1.0, float(used) / max(1, limit))
    st.sidebar.subheader("🤖 Danelfin API (Free Tier)")
    st.sidebar.progress(pct, text=f"Calls: {used} / {limit} ({dan_usage.get('month')})")
    if not dan_usage.get("is_safe"):
        st.sidebar.warning(f"⚠️ Safe threshold reached ({used}/{dan_usage.get('safe_threshold')}). Safe mode active (serving from cache).")
    else:
        st.sidebar.caption(f"✅ {dan_usage.get('remaining')} calls remaining this month (24h SQLite cached).")
except Exception:
    pass

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
    m_buys = stats.get("model_buy_count", stats["buy_count"])
    st.metric(label="🤖 Model Buys", value=m_buys)
with col3:
    m_sells = stats.get("model_sell_count", stats["sell_count"])
    st.metric(label="🤖 Model Sells", value=m_sells)
with col4:
    veto_cnt = stats.get("vetoed_count", 0)
    st.metric(label="🛡️ Gate Vetoes", value=veto_cnt)
with col5:
    win_val = f"{outcome_stats['win_rate_pct']}%" if outcome_stats["total_evaluated"] > 0 else "Pending"
    st.metric(label="🎯 Model Win Rate", value=win_val)
with col6:
    st.metric(label="Last Analysis Run", value=stats["last_run"])

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Latest Signals",
    "🔍 Stock Deep-Dive",
    "🤖 ML Alpha Pipeline",
    "🔥 Most Traded Stocks",
    "📜 Signal History Log",
    "🎯 Model Outcome Tracking",
    "📁 Portfolio Risk Review",
    "⚡ Shadow Execution Desk",
    "🔬 Baseline Decision Audit",
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
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.5, 1.2, 1.2, 1.5])
        with filter_col1:
            decision_filter = st.multiselect(
                "Filter by Model Recommendation:",
                options=["BUY", "DEEP BUY", "HOLD", "BOOK PROFIT", "SELL"],
                default=["BUY", "DEEP BUY", "HOLD", "BOOK PROFIT", "SELL"]
            )
        with filter_col2:
            gate_filter = st.selectbox(
                "Compliance Safety Gate:",
                options=["All Statuses", "Passed Only", "Vetoed Only"],
                index=0
            )
        with filter_col3:
            ml_filter = st.selectbox(
                "Filter by ML Stance:",
                options=["All Stances", "Bullish (Outperform >= 60)", "Neutral (40-60)", "Bearish (Underperform < 40)"],
                index=0
            )
        with filter_col4:
            search_symbol = st.text_input("Search Symbol:", "").strip().upper()

        def _matches_ml_filter(s, flt):
            if flt == "All Stances":
                return True
            sc = s.get("ml_score")
            if sc is None:
                return False
            if flt == "Bullish (Outperform >= 60)":
                return sc >= 60.0
            if flt == "Neutral (40-60)":
                return 40.0 <= sc < 60.0
            if flt == "Bearish (Underperform < 40)":
                return sc < 40.0
            return True

        def _get_model_rec(s):
            return str(s.get("model_decision") or s.get("decision") or "HOLD").upper()

        def _matches_gate_filter(s, flt):
            if flt == "All Statuses":
                return True
            vetoed = is_signal_vetoed(s)
            if flt == "Passed Only":
                return not vetoed
            if flt == "Vetoed Only":
                return vetoed
            return True

        filtered_signals = [
            s for s in latest_signals
            if _get_model_rec(s) in decision_filter
            and _matches_ml_filter(s, ml_filter)
            and _matches_gate_filter(s, gate_filter)
            and (not search_symbol or search_symbol in s["symbol"] or search_symbol in s.get("company_name", "").upper())
        ]

        if filtered_signals:
            df_display = pd.DataFrame(filtered_signals)

            def format_ml_signal_display(row):
                score = row.get("ml_score")
                ret = row.get("ml_predicted_return")
                if score is None or pd.isna(score):
                    return "—"
                ret_str = f" ({ret * 100:+.1f}%)" if ret is not None and not pd.isna(ret) else ""
                if score >= 80:
                    return f"🟢 Strong Outperform{ret_str}"
                elif score >= 60:
                    return f"🟢 Outperform{ret_str}"
                elif score >= 40:
                    return f"⚪ Neutral{ret_str}"
                elif score >= 20:
                    return f"🔴 Underperform{ret_str}"
                else:
                    return f"🔴 Strong Underperform{ret_str}"

            def format_ml_score_display(row):
                score = row.get("ml_score")
                rank = row.get("ml_rank")
                usize = row.get("ml_universe_size") or 483
                if score is None or pd.isna(score):
                    return "—"
                if rank is not None and not pd.isna(rank):
                    return f"{score:.1f} (#{int(rank)}/{int(usize)})"
                return f"{score:.1f}/100"

            df_display["model_rec_col"] = df_display.apply(format_model_recommendation, axis=1)
            df_display["compliance_gate_col"] = df_display.apply(format_compliance_gate_action, axis=1)
            df_display["final_action_col"] = df_display.apply(format_executed_action, axis=1)
            df_display["decision_reasoning_col"] = df_display.apply(lambda r: r.get("decision_reasoning") or r.get("reason") or "—", axis=1)
            df_display["ml_indicator_col"] = df_display.apply(format_ml_signal_display, axis=1)
            df_display["ml_score_col"] = df_display.apply(format_ml_score_display, axis=1)

            cols_to_use = [
                "symbol",
                "model_rec_col",
                "compliance_gate_col",
                "final_action_col",
                "decision_reasoning_col",
                "ml_indicator_col",
                "ml_score_col",
                "confidence",
            ]
            col_renames = {
                "symbol": "Ticker",
                "model_rec_col": "🤖 Model Recommendation",
                "compliance_gate_col": "🛡️ Safety Gate (Veto Action)",
                "final_action_col": "Final Executed Action",
                "decision_reasoning_col": "💡 Decision Reasoning",
                "ml_indicator_col": "🤖 ML Signal",
                "ml_score_col": "🤖 ML Score (Rank)",
                "confidence": "Confidence",
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

            if "danelfin_score" in df_display.columns and df_display["danelfin_score"].notna().any():
                cols_to_use.append("danelfin_score")
                col_renames["danelfin_score"] = "🧠 Danelfin AI"

            if "candidate_type" in df_display.columns and df_display["candidate_type"].notna().any():
                cols_to_use.append("candidate_type")
                col_renames["candidate_type"] = "Candidate Setup"

            if "model_decision" in df_display.columns and df_display["model_decision"].notna().any():
                cols_to_use.append("model_decision")
                col_renames["model_decision"] = "Model Opinion"

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
        analyze_isin_btn = st.button("🚀 Analyze with Ling 3.1", key="analyze_isin_button", use_container_width=True)

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
            with st.status(f"🚀 Running 6-Agent AI Swing Synthesis on {res_sym} via Recursive: Flint (free)...", expanded=True) as status_box:
                st.write(f"Launching main.py for {res_sym}...")
                main_script = os.path.join(BASE_DIR, "main.py")
                cmd = [sys.executable, main_script, "flint", res_sym, "--eu"]
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
            c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1.2, 1])
            with c1:
                model_rec = format_model_recommendation(stock_data)
                model_dec = str(stock_data.get("model_decision") or stock_data.get("decision") or "HOLD").upper()
                badge_class = "badge-buy" if model_dec in ("BUY", "DEEP BUY") else ("badge-sell" if model_dec in ("SELL", "BOOK PROFIT") else "badge-hold")
                st.markdown(f"### **{stock_data['symbol']}**")
                st.markdown(f"**🤖 Model Rec:** <span class='{badge_class}'>{model_rec}</span>", unsafe_allow_html=True)

                vetoed = is_signal_vetoed(stock_data)
                gate_str = format_compliance_gate_action(stock_data)
                gate_badge = "badge-sell" if vetoed else "badge-buy"
                st.markdown(f"**🛡️ Safety Gate:** <span class='{gate_badge}'>{gate_str}</span>", unsafe_allow_html=True)

                final_act = format_executed_action(stock_data)
                st.markdown(f"**Final Action:** `{final_act}`")

                if vetoed:
                    reasons = stock_data.get("rejection_reasons") or stock_data.get("reason_codes")
                    clean_reasons = []
                    if reasons:
                        if isinstance(reasons, str):
                            try:
                                reasons = json.loads(reasons)
                            except Exception:
                                reasons = [reasons]
                        clean_reasons = [r for r in reasons if r not in NON_VETO_REASON_CODES]
                    if clean_reasons:
                        st.caption(f"Veto: {', '.join(clean_reasons)}")
                elif stock_data.get("no_trade_reason"):
                    st.caption(f"Reason: {stock_data['no_trade_reason']}")
            with c2:
                conf = stock_data.get("confidence", 0.0)
                st.metric("Model Confidence", f"{conf * 100:.0f}%" if conf else "N/A")
                st.progress(min(max(float(conf or 0.0), 0.0), 1.0))
            with c3:
                q_score = stock_data.get("quant_score")
                st.metric("Deterministic Quant", f"{q_score}/100" if q_score is not None else "N/A")
            with c4:
                ml_sc = stock_data.get("ml_score")
                ml_pred = stock_data.get("ml_predicted_return")
                ml_rk = stock_data.get("ml_rank")
                ml_usize = stock_data.get("ml_universe_size") or 483
                if ml_sc is not None:
                    delta_str = f"{ml_pred * 100:+.1f}% 20d" if ml_pred is not None else None
                    st.metric(
                        "🤖 Standalone ML Score",
                        f"{ml_sc:.1f}/100",
                        delta=delta_str,
                        help=f"Rank #{ml_rk} of {ml_usize} in S&P 500"
                    )
                else:
                    st.metric("🤖 Standalone ML Score", "N/A", help="Non-S&P 500 or unranked")
            with c5:
                entry_p = stock_data.get("entry_price")
                st.metric("Entry Price", f"{curr_sym}{entry_p:.2f}" if entry_p else "N/A")

            # =====================================================================
            # 💡 AI Model Decision Reasoning & Rationale
            # =====================================================================
            model_dec_str = str(stock_data.get("model_decision") or stock_data.get("decision") or "HOLD").upper()
            model_reasoning = stock_data.get("decision_reasoning") or stock_data.get("reason")
            if not model_reasoning or model_reasoning == stock_data.get("bull_case"):
                raw_j = stock_data.get("raw_json")
                if raw_j:
                    try:
                        pj = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
                        model_reasoning = (
                            pj.get("decision_reasoning")
                            or pj.get("decision_rationale")
                            or (pj.get("forecast_20d", {}).get("rationale") if isinstance(pj.get("forecast_20d"), dict) else None)
                            or pj.get("reason")
                        )
                    except Exception:
                        pass

            st.markdown("#### 💡 AI Model Decision Reasoning")
            if model_reasoning:
                st.info(f"**Why the Model Recommended {model_dec_str}:**\n\n{model_reasoning}")
            else:
                st.info(f"No specific decision reasoning recorded for `{selected_stock}`.")

            # =====================================================================
            # 🤖 Standalone ML Pipeline Indicator (Point-in-Time XGBoost Model)
            # =====================================================================
            st.markdown("#### 🤖 Standalone ML Pipeline Alpha Indicator (XGBoost 20-Session Model)")
            ml_ctx = None
            if stock_data:
                raw_j = stock_data.get("raw_json")
                if raw_j:
                    try:
                        parsed_pj = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
                        ml_ctx = parsed_pj.get("ml_signal")
                    except Exception:
                        pass
            if not ml_ctx:
                ml_ctx = get_ml_context(selected_stock)

            if ml_ctx and ml_ctx.get("in_scope") and ml_ctx.get("score_0_100") is not None:
                score_val = float(ml_ctx.get("score_0_100") or 0.0)
                cls = classify_ml_signal(score_val)
                ml_badge = ml_ctx.get("badge") or cls["badge"]
                ml_tier = ml_ctx.get("tier") or cls["tier"]
                pred_ret = ml_ctx.get("predicted_20d_excess_return")
                ml_pred_pct = (float(pred_ret) * 100.0) if pred_ret is not None else 0.0
                in_top10 = ml_ctx.get("in_top_10_portfolio", False)

                if in_top10:
                    st.success(f"⭐ **TOP 10 HIGH-CONVICTION ML BASKET** — `{selected_stock}` ranks #{ml_ctx.get('rank')} in the model's highest-conviction long portfolio for the next 20 sessions.")

                ml_k1, ml_k2, ml_k3, ml_k4, ml_k5 = st.columns(5)
                ml_k1.metric("Pure ML Stance", ml_ctx.get("signal") or cls["signal"])
                ml_k2.metric("ML Alpha Score", f"{score_val:.1f}/100")
                ml_k3.metric("S&P 500 Rank", f"#{ml_ctx.get('rank')} of {ml_ctx.get('universe_size', 483)}")
                ml_k4.metric("Pred 20d Excess Return", f"{ml_pred_pct:+.2f}%")
                ml_k5.metric("Alpha Tier", ml_tier.split(" (")[0])

                # Visual Score Gauge Bar
                st.progress(min(max(score_val / 100.0, 0.0), 1.0))
                st.caption("Score Scale: 0–20 Strong Underperform | 20–40 Underperform | 40–60 Neutral | 60–80 Outperform | 80–100 Strong Outperform")

                # Sub-model ranking and metadata
                sub_cols = st.columns(4)
                fund_rk = ml_ctx.get("rank_fundamentals")
                tech_rk = ml_ctx.get("rank_technical")
                sub_cols[0].caption(f"**Fundamentals Sub-Model:** `#{fund_rk}`" if fund_rk else "**Fundamentals Sub-Model:** Unranked")
                sub_cols[1].caption(f"**Technical Sub-Model:** `#{tech_rk}`" if tech_rk else "**Technical Sub-Model:** Unranked")
                sub_cols[2].caption(f"**Model:** `{ml_ctx.get('model_name', 'xgb_pit_core_rank_1m')}`")
                sub_cols[3].caption(f"**Score Date:** `{ml_ctx.get('score_date')}` ({ml_ctx.get('age_days', 0)}d old)")

                # Multi-Agent vs ML Disposition
                raw_assessment = {}
                if stock_data and stock_data.get("raw_json"):
                    try:
                        raw_assessment = json.loads(stock_data["raw_json"]).get("ml_signal_assessment", {})
                    except Exception:
                        pass
                if not raw_assessment and stock_data and isinstance(stock_data.get("ml_signal_assessment"), dict):
                    raw_assessment = stock_data["ml_signal_assessment"]

                if raw_assessment:
                    disp = raw_assessment.get("disposition", "NEUTRAL")
                    reason = raw_assessment.get("reason", "")
                    if disp == "CONFLICTING":
                        st.warning(f"⚠️ **ML Signal vs Multi-Agent Setup (CONFLICTING):** {reason}")
                    elif disp == "SUPPORTIVE":
                        st.success(f"✅ **ML Signal vs Multi-Agent Setup (SUPPORTIVE):** {reason}")
                    else:
                        st.info(f"ℹ️ **ML Signal vs Multi-Agent Setup (NEUTRAL):** {reason}")
            else:
                st.info(f"ℹ️ No point-in-time S&P 500 ML model score available for `{selected_stock}`. The ML alpha pipeline currently evaluates the US S&P 500 universe (~483 components).")

            # =====================================================================
            # 🧠 Danelfin AI Quantitative Intelligence
            # =====================================================================
            danelfin_score = stock_data.get("danelfin_score")
            danelfin_sig = stock_data.get("danelfin_signal")
            danelfin_details_raw = stock_data.get("danelfin_details")
            dan_info = None
            if danelfin_details_raw and isinstance(danelfin_details_raw, str):
                try:
                    dan_info = json.loads(danelfin_details_raw)
                except Exception:
                    dan_info = None
            if not dan_info and stock_data.get("raw_json"):
                try:
                    dan_info = json.loads(stock_data["raw_json"]).get("danelfin")
                except Exception:
                    dan_info = None

            if danelfin_score is not None or (dan_info and dan_info.get("aiscore") is not None):
                d_score = danelfin_score if danelfin_score is not None else dan_info.get("aiscore")
                st.markdown("#### 🧠 Danelfin AI Quantitative Intelligence")
                dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                with dc1:
                    st.metric("Overall AI Score", f"{d_score}/10")
                with dc2:
                    st.metric("Fundamental", f"{dan_info.get('fundamental', '—')}/10" if dan_info else "—")
                with dc3:
                    st.metric("Technical", f"{dan_info.get('technical', '—')}/10" if dan_info else "—")
                with dc4:
                    st.metric("Sentiment", f"{dan_info.get('sentiment', '—')}/10" if dan_info else "—")
                with dc5:
                    st.metric("Low Risk", f"{dan_info.get('low_risk', '—')}/10" if dan_info else "—")

                if danelfin_sig:
                    sig_badge = "🟢 BUY" if str(danelfin_sig).lower() == "buy" else ("🔴 SELL" if str(danelfin_sig).lower() == "sell" else "⚪ HOLD")
                    st.caption(f"Danelfin AI Stance: **{sig_badge}** | As of: `{dan_info.get('date', 'latest') if dan_info else 'latest'}`")

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

            # =====================================================================
            # 🛡️ Structured Decision & Policy Compliance Gates (Step 6.4 & 7.2)
            # =====================================================================
            st.markdown("#### 🛡️ Structured Decision & Policy Compliance Gates")
            pos_action = stock_data.get("position_action") or "NO_ACTION"
            passed_g = stock_data.get("passed_gates") or []
            failed_g = stock_data.get("failed_gates") or []
            r_codes = stock_data.get("reason_codes") or []
            sizing_data = stock_data.get("position_sizing") or {}
            if not sizing_data and stock_data.get("raw_json"):
                try:
                    sizing_data = json.loads(stock_data["raw_json"]).get("position_sizing") or {}
                except Exception:
                    pass

            sd_c1, sd_c2 = st.columns([1.5, 3])
            with sd_c1:
                action_badge = "badge-open-long" if pos_action == "OPEN_LONG" else "badge-no-action"
                st.markdown(f"**Policy Action:** <span class='{action_badge}'>{pos_action}</span>", unsafe_allow_html=True)
                vetoed_sd = is_signal_vetoed(stock_data)
                gate_str_sd = format_compliance_gate_action(stock_data)
                gate_badge_sd = "badge-sell" if vetoed_sd else "badge-buy"
                st.markdown(f"**Gate Status:** <span class='{gate_badge_sd}'>{gate_str_sd}</span>", unsafe_allow_html=True)
                if sizing_data.get("binding_constraint"):
                    st.caption(f"Binding Constraint: `{sizing_data['binding_constraint']}`")
            with sd_c2:
                if passed_g:
                    p_tags = " ".join([f"<span class='badge-gate-pass'>✓ {g}</span>" for g in passed_g])
                    st.markdown(f"**Passed Gates:** {p_tags}", unsafe_allow_html=True)
                if failed_g:
                    f_tags = " ".join([f"<span class='badge-gate-fail'>✗ {g}</span>" for g in failed_g])
                    st.markdown(f"**Failed Gates:** {f_tags}", unsafe_allow_html=True)
                if r_codes:
                    rc_str = ", ".join([f"`{rc}`" for rc in r_codes])
                    st.markdown(f"**Reason Codes:** {rc_str}")

            # =====================================================================
            # 📐 Institutional Position Sizing & Risk Controls (Step 7.1 - 7.3)
            # =====================================================================
            st.markdown("#### 📐 Institutional Position Sizing & Risk Controls")
            sz_c1, sz_c2, sz_c3, sz_c4, sz_c5 = st.columns(5)
            shares_al = sizing_data.get("shares_allowed", 0)
            sz_c1.metric("Shares Allowed", f"{shares_al} shs")
            sz_c2.metric("Risk Budget", f"${sizing_data.get('risk_budget_usd', 1000):,.0f}", help="1% Portfolio Risk Budget")
            sz_c3.metric("Dollar Risk", f"${sizing_data.get('dollar_risk', 0):,.0f}", help="Max loss to stop-out")
            gap_pct = sizing_data.get("gap_reserve_pct", 0.20)
            sz_c4.metric("Gap Reserve", f"{gap_pct * 100:.0f}%", help="Overnight ATR gap buffer")
            max_alloc = sizing_data.get("max_capital_allocation", 0)
            sz_c5.metric("Capital Capped", f"${max_alloc:,.0f}" if max_alloc else "N/A", help="Single-stock & sector max limits")

            rejections = sizing_data.get("rejection_reasons") or []
            if rejections:
                st.warning("⚠️ **Active Position Sizing Restrictions:** " + "; ".join(rejections))

            st.markdown("---")

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("🐂 Bull Case Arguments")
                st.info(stock_data.get("bull_case") or "No bullish arguments provided.")

                st.subheader("🏛️ SEC EDGAR Institutional & Filings")
                st.write(stock_data.get("institutional_data") or "No SEC filing summary available.")

                st.subheader("🌐 Macro Economy (FRED) & CFTC COT")
                st.write(stock_data.get("macro_data") or stock_data.get("marco_data") or "No macro summary available.")

                st.subheader("📰 Today's News Catalysts")
                st.write(stock_data.get("news") or "No news catalyst summary reported.")

            with col_right:
                st.subheader("🐻 Downside Risks & Bear Case")
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
    st.subheader("🤖 Point-in-Time 20-Session ML Alpha Pipeline (XGBoost)")
    st.caption("Standalone point-in-time machine learning pipeline trained to predict 20-session forward excess return vs the S&P 500 benchmark. Operates purely on market data and point-in-time fundamentals without qualitative LLM reasoning or trade risk gates.")

    all_ml = get_all_ml_scores()
    if not all_ml:
        st.info("No active ML scores found in `stock_ml/data/xgb_current_scores_1m.csv`.")
    else:
        top_10 = [x for x in all_ml if x.get("in_top_10_portfolio") or x["rank"] <= 10][:10]
        bottom_10 = all_ml[-10:]

        # Pipeline KPI Overview
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ML Universe Size", f"{len(all_ml)} Stocks")
        k2.metric("Top Model Alpha Pick", f"{all_ml[0]['ticker']} (#{all_ml[0]['rank']})", delta=f"{all_ml[0]['predicted_20d_excess_return']*100:+.1f}% 20d")
        k3.metric("Bottom Model Pick", f"{all_ml[-1]['ticker']} (#{all_ml[-1]['rank']})", delta=f"{all_ml[-1]['predicted_20d_excess_return']*100:+.1f}% 20d")
        k4.metric("Forecast Horizon", f"{all_ml[0].get('age_days', 0)}d old | 20 Sessions")

        st.markdown("---")

        # Top 10 Long Conviction Portfolio
        st.markdown("#### 🏆 S&P 500 Top-10 High-Conviction ML Long Basket")
        st.caption("The 10 highest-ranked S&P 500 stocks selected by the Point-in-Time XGBoost alpha model.")
        top_rows = []
        for s in top_10:
            top_rows.append({
                "Rank": f"#{s['rank']}",
                "Ticker": s["ticker"],
                "Sector": s["sector"],
                "ML Score": f"{s['score_0_100']:.1f}/100",
                "Predicted 20d Return": f"{s['predicted_20d_excess_return']*100:+.2f}%",
                "Pure ML Stance": s["badge"],
                "Fundamental Rank": f"#{s['rank_fundamentals']}" if s.get("rank_fundamentals") else "—",
                "Technical Rank": f"#{s['rank_technical']}" if s.get("rank_technical") else "—",
            })
        st.dataframe(pd.DataFrame(top_rows), width="stretch", hide_index=True)

        st.markdown("---")

        # Bottom 10 Short Basket
        st.markdown("#### 🔻 Bottom-10 ML Underperform Basket (Short Candidates)")
        st.caption("The 10 lowest-ranked stocks predicted to underperform the S&P 500 benchmark over the next 20 sessions.")
        bot_rows = []
        for s in bottom_10:
            bot_rows.append({
                "Rank": f"#{s['rank']}",
                "Ticker": s["ticker"],
                "Sector": s["sector"],
                "ML Score": f"{s['score_0_100']:.1f}/100",
                "Predicted 20d Return": f"{s['predicted_20d_excess_return']*100:+.2f}%",
                "Pure ML Stance": s["badge"],
                "Fundamental Rank": f"#{s['rank_fundamentals']}" if s.get("rank_fundamentals") else "—",
                "Technical Rank": f"#{s['rank_technical']}" if s.get("rank_technical") else "—",
            })
        st.dataframe(pd.DataFrame(bot_rows), width="stretch", hide_index=True)

        st.markdown("---")

        # Full Interactive ML Universe Screener
        st.markdown("#### 🔎 S&P 500 ML Universe Screener")
        ml_s_col1, ml_s_col2, ml_s_col3 = st.columns([1.5, 1.5, 1.5])
        with ml_s_col1:
            all_sectors = sorted(list({x["sector"] for x in all_ml if x.get("sector")}))
            sec_filter = st.selectbox("Filter by Sector:", options=["All Sectors"] + all_sectors, key="ml_sec_flt")
        with ml_s_col2:
            tier_filter = st.selectbox(
                "Filter by Quintile:",
                options=["All Quintiles", "Strong Outperform (80-100)", "Outperform (60-80)", "Neutral (40-60)", "Underperform (20-40)", "Strong Underperform (0-20)"],
                key="ml_tier_flt"
            )
        with ml_s_col3:
            search_ml_sym = st.text_input("Search Ticker:", "", key="ml_screener_search").strip().upper()

        filtered_ml = all_ml
        if sec_filter != "All Sectors":
            filtered_ml = [x for x in filtered_ml if x.get("sector") == sec_filter]
        if tier_filter == "Strong Outperform (80-100)":
            filtered_ml = [x for x in filtered_ml if x["score_0_100"] >= 80.0]
        elif tier_filter == "Outperform (60-80)":
            filtered_ml = [x for x in filtered_ml if 60.0 <= x["score_0_100"] < 80.0]
        elif tier_filter == "Neutral (40-60)":
            filtered_ml = [x for x in filtered_ml if 40.0 <= x["score_0_100"] < 60.0]
        elif tier_filter == "Underperform (20-40)":
            filtered_ml = [x for x in filtered_ml if 20.0 <= x["score_0_100"] < 40.0]
        elif tier_filter == "Strong Underperform (0-20)":
            filtered_ml = [x for x in filtered_ml if x["score_0_100"] < 20.0]
        if search_ml_sym:
            filtered_ml = [x for x in filtered_ml if search_ml_sym in x["ticker"]]

        screener_rows = []
        for s in filtered_ml:
            screener_rows.append({
                "Rank": f"#{s['rank']}",
                "Ticker": s["ticker"],
                "Sector": s["sector"],
                "ML Score": f"{s['score_0_100']:.1f}/100",
                "Predicted 20d Excess Return": f"{s['predicted_20d_excess_return']*100:+.2f}%",
                "Pure ML Stance": s["badge"],
                "Top 10 Basket": "⭐ YES" if s.get("in_top_10_portfolio") else "—",
                "Fundamental Rank": f"#{s['rank_fundamentals']}" if s.get("rank_fundamentals") else "—",
                "Technical Rank": f"#{s['rank_technical']}" if s.get("rank_technical") else "—",
            })
        st.dataframe(pd.DataFrame(screener_rows), width="stretch", hide_index=True)

with tab4:
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
            "AI Analysis Count (Recursive: Flint):",
            options=[3, 5, 10],
            index=1,
            help="Number of top active stocks to sequentially analyze with main.py using Recursive: Flint (free) ('flint')"
        )
    with top_ctrl3:
        if st.button("🔄 Refresh Active Data", key="refresh_active_stocks", use_container_width=True):
            get_cached_active_stocks.clear()
            st.rerun()

        run_analyse = st.button(
            "🚀 Refresh active data & Analyse",
            key="refresh_and_analyse_stocks",
            help="Refreshes active stocks data and executes master.py using Recursive: Flint (free) (flint)",
            use_container_width=True,
            type="primary"
        )

    if run_analyse:
        get_cached_active_stocks.clear()
        master_script = os.path.join(BASE_DIR, "master.py")
        cmd = [
            sys.executable,
            master_script,
            "flint",
            "--limit", str(analyse_count),
            "--dashboard-limit", str(limit_val),
            "--market", selected_market,
            "--prefer-exchange", selected_exchange,
            "--no-streamlit",
        ]
        status_market_label = f"Europe (gettex / XETRA: {selected_exchange})" if is_eu else "US Markets"
        with st.status(f"🚀 Running Gloomberb Master Orchestration ({status_market_label} | Model: flint, Limit: {analyse_count})...", expanded=True) as status_box:
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
                    st.success(f"Successfully analyzed top {analyse_count} active stocks using Ling 3.1 Flash ('ling')! Trade signals saved to database.")
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
            ml_ctx = get_ml_context(s["symbol"])
            if ml_ctx and ml_ctx.get("score_0_100") is not None:
                ml_disp = f"{ml_ctx['score_0_100']:.1f} (#{ml_ctx.get('rank', '—')})"
            else:
                ml_disp = "—"
            table_rows.append({
                "Ticker": s["symbol"],
                "Company": s["name"],
                "Price": f"{curr_sym}{s['price']:.2f}",
                "Change": f"{sign}{chg:.2f}% ({sign}{curr_sym}{abs(s['change']):.2f})",
                "🤖 ML Score": ml_disp,
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
            ml_ctx = get_ml_context(s["symbol"])
            with st.expander(f"{badge} **{s['symbol']}** ({s['name']}) — {curr_sym}{s['price']:.2f} ({chg_sign}{s['change_pct']:.2f}%) | Vol: {format_volume(s['volume'])} ({s['rvol']:.1f}x 20d avg)"):
                m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                m_c1.metric("Day Low / High", f"{curr_sym}{s['day_low']:.2f} - {curr_sym}{s['day_high']:.2f}")
                m_c2.metric("Market Cap", s["market_cap_str"])
                m_c3.metric("RVOL (20d avg)", f"{s['rvol']:.1f}x")
                m_c4.metric("RSI14 / ATR", f"{s.get('rsi14', 'N/A')} / {curr_sym}{s.get('atr', 'N/A')}")
                if ml_ctx and ml_ctx.get("score_0_100") is not None:
                    pred_ret = ml_ctx.get("predicted_20d_excess_return")
                    delta_str = f"{pred_ret*100:+.1f}% 20d" if pred_ret is not None else None
                    m_c5.metric("🤖 Pure ML Score", f"{ml_ctx['score_0_100']:.1f}/100", delta=delta_str)
                else:
                    m_c5.metric("🤖 Pure ML Score", "N/A")

                st.markdown(f"**⚡ Catalyst Tag:** `{s['catalyst_type']}`")
                st.info(f"**Reason for High Volume:** {s['reason_summary']}")

                news_list = s.get("news", [])
                if news_list:
                    st.markdown("**Recent News Headlines:**")
                    for n in news_list:
                        pub = f"[{n['publisher']}] " if n.get('publisher') else ""
                        url = n.get('url', '#')
                        st.markdown(f"- {pub}[{n['title']}]({url})")



with tab5:
    st.subheader("Historical Signal Analysis Log")
    history_records = get_signal_history(limit=200)

    if not history_records:
        st.info("No historical records logged yet.")
    else:
        df_hist_raw = pd.DataFrame(history_records)
        df_hist_raw["model_rec_col"] = df_hist_raw.apply(format_model_recommendation, axis=1)
        df_hist_raw["gate_action_col"] = df_hist_raw.apply(format_compliance_gate_action, axis=1)
        df_hist_raw["final_action_col"] = df_hist_raw.apply(format_executed_action, axis=1)

        cols = [
            "id", "timestamp", "symbol", "model_rec_col", "gate_action_col", "final_action_col", "confidence", "model_used", "reason"
        ]
        cols = [c for c in cols if c in df_hist_raw.columns]
        df_hist = df_hist_raw[cols].rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "symbol": "Ticker",
            "model_rec_col": "🤖 Model Recommendation",
            "gate_action_col": "🛡️ Safety Gate (Veto Action)",
            "final_action_col": "Final Executed Action",
            "confidence": "Confidence",
            "model_used": "Model Used",
            "reason": "💡 Model Decision Reasoning"
        })
        st.dataframe(df_hist, width="stretch", hide_index=True)

with tab6:
    st.subheader("🎯 Model Ground-Truth Outcome Tracking & Evaluation")
    st.caption("Tracks new 20-session forecasts over forward 20 trading bars. Older signals retain their original recorded horizon.")

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
            SELECT o.id, s.timestamp as signal_date, s.symbol,
                   COALESCE(s.model_decision, s.decision) as model_decision,
                   s.decision,
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
        df_outcomes = df_outcomes.rename(columns={
            "model_decision": "🤖 Model Rec",
            "decision": "Final Action"
        })
        st.dataframe(df_outcomes, width="stretch", hide_index=True)
    else:
        st.info("No trade outcomes evaluated yet. New 20-session signals are evaluated after 20 forward trading bars; older signals use their recorded horizon.")

with tab7:
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

with tab8:
    st.subheader("⚡ Institutional Shadow Trading & Order Execution Desk")
    st.caption("Point-in-Time Shadow Trading Validation, 8 Formal Measurement Scorecards, and Operational Kill Switch (Steps 7.3 & 8).")

    # 1. Operational Kill Switch Controls
    ks_status = GLOBAL_KILL_SWITCH.get_status()
    st.markdown("### 🛑 Operational Kill Switch")
    ks_col1, ks_col2 = st.columns([3, 1])
    with ks_col1:
        if ks_status["is_tripped"]:
            st.error(
                f"🚨 **KILL SWITCH TRIPPED** | Execution is FROZEN.\n\n"
                f"- **Reason:** `{ks_status['trip_reason']}`\n"
                f"- **Tripped By:** `{ks_status['tripped_by']}`\n"
                f"- **Timestamp:** `{ks_status['trip_timestamp']}`"
            )
        else:
            st.success("✅ **KILL SWITCH OPERATIONAL & ACTIVE** | Zero-latency pre-flight gating active. Normal automated execution enabled.")
    with ks_col2:
        if ks_status["is_tripped"]:
            if st.button("🔄 Reset / Restore Trading", type="primary", use_container_width=True, key="reset_ks_tab8"):
                GLOBAL_KILL_SWITCH.reset(operator="WebConsole")
                st.success("Kill switch reset! Trading execution restored.")
                st.rerun()
        else:
            if st.button("🚨 Emergency Trip", type="secondary", use_container_width=True, key="trip_ks_tab8"):
                GLOBAL_KILL_SWITCH.trip(
                    reason=KillSwitchReason.MANUAL_OPERATOR_TRIP,
                    details={"operator": "WebConsole"},
                    operator="WebConsole"
                )
                st.warning("Kill switch tripped! All entries frozen.")
                st.rerun()

    st.markdown("---")

    # 2. Institutional Promotion Gate Scorecard
    st.markdown("### 🏆 Institutional Promotion Gate Scorecard (Predefined Criteria)")
    shadow_raw = get_shadow_decisions(limit=100)

    if not shadow_raw:
        st.info("No shadow trading records logged yet. Run analyses in terminal or dashboard to stream simulated execution decisions.")
    else:
        # Convert to ShadowDecisionRecord objects for analyzer
        shadow_objs = []
        for r in shadow_raw:
            rec = ShadowDecisionRecord(
                session_id=r.get("session_id", ""),
                run_id=r.get("run_id", ""),
                ticker=r.get("ticker", ""),
                timestamp=r.get("timestamp", ""),
                decision=r.get("decision", "HOLD"),
                position_action=r.get("position_action", "NO_ACTION"),
                composite_score=r.get("composite_score"),
                signal_strength=r.get("signal_strength"),
                ml_score=r.get("ml_score"),
                ml_predicted_return=r.get("ml_predicted_return"),
                entry_price=r.get("entry_price"),
                suggested_stop_loss=r.get("suggested_stop_loss"),
                suggested_target_price=r.get("suggested_target_price"),
                passed_gates=r.get("passed_gates") or [],
                failed_gates=r.get("failed_gates") or [],
                reason_codes=r.get("reason_codes") or [],
                proposed_shares=r.get("proposed_shares", 0),
                binding_constraint=r.get("binding_constraint"),
                is_order_proposed=bool(r.get("is_order_proposed")),
                simulated_order_id=r.get("simulated_order_id"),
                simulated_fill_price=r.get("simulated_fill_price"),
                simulated_slippage_bps=r.get("simulated_slippage_bps"),
                simulated_commission=r.get("simulated_commission"),
                simulated_status=r.get("simulated_status"),
                realized_forward_return_pct=r.get("realized_forward_return_pct"),
                realized_net_return_pct=r.get("realized_net_return_pct"),
                is_fresh=bool(r.get("is_fresh", True)),
                decision_latency_seconds=r.get("decision_latency_seconds", 0.001),
                quote_age_seconds=r.get("quote_age_seconds", 0.0),
                input_snapshot=r.get("input_snapshot_json") or {},
                component_outputs=r.get("component_outputs_json") or {},
            )
            shadow_objs.append(rec)

        analyzer = ShadowMetricsAnalyzer()
        report = analyzer.analyze_session(records=shadow_objs, session_id="WEB_AGGREGATE")
        evaluator = ShadowPromotionEvaluator()
        promo = evaluator.evaluate(report)

        p_status = promo.status.value if hasattr(promo.status, "value") else str(promo.status)
        p_badge = "badge-open-long" if p_status == "PROMOTED_TO_PRODUCTION" else ("badge-ml-neutral" if "OBSERVATION" in p_status else "badge-sell")

        obs_count = report.total_observations
        min_obs = evaluator.policy.shadow_trading.min_observations_for_promotion if hasattr(evaluator.policy, "shadow_trading") else 20
        passed_count = len(promo.passed_criteria)
        total_req_count = len(promo.passed_criteria) + len(promo.failed_criteria)

        p_c1, p_c2, p_c3, p_c4 = st.columns([1.5, 1, 1, 1])
        p_c1.markdown(f"**Gate Status:** <span class='{p_badge}'>{p_status}</span>", unsafe_allow_html=True)
        p_c2.metric("Observations", f"{obs_count} / {min_obs}")
        p_c3.metric("Passed Requirements", f"{passed_count} / {total_req_count}" if total_req_count > 0 else "0 / 8")
        p_c4.metric("Live Eligible", "YES" if promo.is_promoted else "NO")

        st.caption(f"**Summary:** {promo.remediation_guidance or p_status}")

        if promo.failed_criteria:
            st.warning("⚠️ **Required Remediation Actions:** " + " • ".join(promo.failed_criteria))

        # 8 Formal Measurements Table
        st.markdown("#### 📊 The Eight Formal Shadow Measurements")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("1. Signal Reproducibility", f"{report.reproducibility_rate * 100:.0f}%", help="Deterministic decision match rate")
        m_col2.metric("2. Mean Latency", f"{report.mean_latency_seconds * 1000:.1f}ms", help="Decision execution latency")
        m_col3.metric("3. Directional Accuracy", f"{report.directional_accuracy * 100:.0f}%", help="Forecast direction consistency")
        m_col4.metric("4. Cost-Adjusted Sharpe", f"{report.net_sharpe_ratio:.2f}", help="Net Sharpe after realistic slippage")

        m_col5, m_col6, m_col7, m_col8 = st.columns(4)
        m_col5.metric("5. Gate Rejection Rate", f"{report.rejection_rate_pct:.0f}%", help="Trades filtered by compliance gates")
        m_col6.metric("6. Model Drift (PSI)", f"{report.overall_psi:.3f}", help="Population Stability Index")
        m_col7.metric("7. Temporal Integrity", f"{(1.0 - report.temporal_leakage_rate) * 100:.0f}%", help="Timestamp-verified Point-in-Time data citations")
        m_col8.metric("8. Ledger Discrepancies", f"{report.reconciliation_error_rate * 100:.0f}%", help="State drift between portfolio & ledger")

        # Shadow Trading Decisions Audit Log
        st.markdown("#### 📜 Shadow Trading Execution Log")
        log_rows = []
        for r in shadow_raw:
            log_rows.append({
                "Timestamp": r.get("timestamp"),
                "Ticker": r.get("ticker"),
                "Decision": r.get("decision"),
                "Action": r.get("position_action"),
                "Composite": f"{r.get('composite_score', 0):.1f}" if r.get("composite_score") is not None else "N/A",
                "ML Score": f"{r.get('ml_score', 0):.1f}" if r.get("ml_score") is not None else "N/A",
                "Shares": r.get("proposed_shares", 0),
                "Sim Fill Price": f"${r.get('simulated_fill_price', 0):.2f}" if r.get("simulated_fill_price") else "N/A",
                "Slippage": f"{r.get('simulated_slippage_bps', 0):.1f} bps" if r.get("simulated_slippage_bps") is not None else "0 bps",
                "Status": r.get("simulated_status", "SIMULATED"),
                "Passed Gates": ", ".join(r.get("passed_gates") or []),
                "Failed Gates": ", ".join(r.get("failed_gates") or []),
                "Reason Codes": ", ".join(r.get("reason_codes") or []),
            })
        df_shadow = pd.DataFrame(log_rows)
        st.dataframe(df_shadow, width="stretch", hide_index=True)


with tab9:
    st.subheader("🔬 Baseline Decision Audit & Component Breakdown")
    st.caption("Point-in-Time baseline of recent signals, model proposals, quant scores, candidate setups, compliance gates, and root-cause explanations for HOLD decisions.")

    baseline_path = os.path.join(BASE_DIR, "data", "baseline_signals_audit.json")
    baseline_records = []
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_records = json.load(f)
        except Exception as e:
            st.error(f"Error loading baseline file: {e}")

    if not baseline_records:
        st.info("No baseline audit records found in `data/baseline_signals_audit.json`.")
    else:
        # High-level KPIs
        total_b = len(baseline_records)
        no_setup_count = sum(1 for r in baseline_records if r.get("hold_classification") == "NO_SETUP_QUALIFIED")
        gate_failed_count = sum(1 for r in baseline_records if r.get("hold_classification") == "GATE_FAILED")
        actions_count = sum(1 for r in baseline_records if str(r.get("hold_classification", "")).startswith("ACTION_"))
        model_hold_count = sum(1 for r in baseline_records if r.get("hold_classification") == "MODEL_PROPOSED_HOLD")

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        kpi_c1.metric("Total Baseline Signals", total_b)
        kpi_c2.metric("🛑 No Setup Qualified", f"{no_setup_count} ({no_setup_count/total_b*100:.0f}%)" if total_b else "0")
        kpi_c3.metric("⚠️ Gate Failed", f"{gate_failed_count} ({gate_failed_count/total_b*100:.0f}%)" if total_b else "0")
        kpi_c4.metric("🟢 Certified Actions (BUY/SELL)", f"{actions_count} ({actions_count/total_b*100:.0f}%)" if total_b else "0")

        # Filters
        f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1])
        with f_col1:
            cat_choice = st.selectbox(
                "Filter by Classification:",
                ["All Signals", "🛑 No Setup Qualified", "⚠️ Gate Failed", "🟢 Certified Actions (BUY/SELL)", "💡 Model Discretionary HOLD"],
                key="baseline_cat_filter"
            )
        with f_col2:
            sym_search = st.text_input("Search Ticker:", "", key="baseline_sym_search").strip().upper()
        with f_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            with open(baseline_path, "r", encoding="utf-8") as f:
                raw_json_str = f.read()
            st.download_button(
                "📥 Download JSON",
                data=raw_json_str,
                file_name="baseline_signals_audit.json",
                mime="application/json",
                use_container_width=True
            )

        # Filter records
        filtered_records = baseline_records
        if cat_choice == "🛑 No Setup Qualified":
            filtered_records = [r for r in filtered_records if r.get("hold_classification") == "NO_SETUP_QUALIFIED"]
        elif cat_choice == "⚠️ Gate Failed":
            filtered_records = [r for r in filtered_records if r.get("hold_classification") == "GATE_FAILED"]
        elif cat_choice == "🟢 Certified Actions (BUY/SELL)":
            filtered_records = [r for r in filtered_records if str(r.get("hold_classification", "")).startswith("ACTION_")]
        elif cat_choice == "💡 Model Discretionary HOLD":
            filtered_records = [r for r in filtered_records if r.get("hold_classification") == "MODEL_PROPOSED_HOLD"]

        if sym_search:
            filtered_records = [r for r in filtered_records if sym_search in r.get("symbol", "").upper()]

        # Format rows for table
        table_rows = []
        for r in filtered_records:
            passed_g = r.get("passed_gates") or []
            if isinstance(passed_g, str):
                try:
                    passed_g = json.loads(passed_g)
                except Exception:
                    passed_g = [passed_g]
            failed_g = r.get("failed_gates") or []
            if isinstance(failed_g, str):
                try:
                    failed_g = json.loads(failed_g)
                except Exception:
                    failed_g = [failed_g]
            reasons = r.get("reason_codes") or []
            if isinstance(reasons, str):
                try:
                    reasons = json.loads(reasons)
                except Exception:
                    reasons = [reasons]

            gates_str = ""
            if failed_g:
                gates_str = "Failed: " + ", ".join(failed_g)
            elif reasons and reasons != ["NO_TRADE_DEFAULT"]:
                gates_str = ", ".join(reasons)
            elif passed_g:
                gates_str = "Passed: " + ", ".join(passed_g)
            else:
                gates_str = "—"

            q_score = r.get("quant_score")
            q_disp = f"{q_score:.1f}" if q_score is not None else "—"

            dc = r.get("data_completeness")
            dc_disp = f"{int(dc * 100)}%" if dc is not None else "—"

            h_class = r.get("hold_classification", "")
            if h_class == "NO_SETUP_QUALIFIED":
                h_disp = "🛑 No Setup Qualified"
            elif h_class == "GATE_FAILED":
                h_disp = "⚠️ Gate Failed"
            elif h_class == "ACTION_BUY":
                h_disp = "🟢 Certified BUY"
            elif h_class == "ACTION_SELL":
                h_disp = "🔴 Certified SELL"
            elif h_class == "MODEL_PROPOSED_HOLD":
                h_disp = "💡 Model HOLD"
            else:
                h_disp = h_class

            table_rows.append({
                "ID": r.get("id"),
                "Timestamp": r.get("timestamp"),
                "Symbol": r.get("symbol"),
                "Model Proposal": r.get("model_decision"),
                "Quant Score": q_disp,
                "Data Compl.": dc_disp,
                "Candidate Setup": r.get("candidate_setup"),
                "Active Gates / Reasons": gates_str,
                "Final Action": r.get("final_action"),
                "Hold Classification": h_disp,
                "Root Cause Explanation": r.get("root_cause_explanation"),
            })

        df_b = pd.DataFrame(table_rows)
        st.dataframe(df_b, width="stretch", hide_index=True)

        # Drill-down inspection
        st.markdown("#### 🔍 Baseline Signal Inspection")
        sig_ids = [f"#{r['id']} {r['symbol']} ({r['timestamp']})" for r in filtered_records]
        if sig_ids:
            selected_sig_str = st.selectbox("Select signal to view full structured contract:", sig_ids, key="sel_baseline_sig")
            sel_id = int(selected_sig_str.split(" ")[0].replace("#", ""))
            sel_rec = next((r for r in filtered_records if r["id"] == sel_id), None)
            if sel_rec:
                with st.expander("📄 View Full JSON Contract", expanded=True):
                    st.json(sel_rec)


