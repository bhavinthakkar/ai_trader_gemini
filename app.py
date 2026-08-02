import streamlit as st
import pandas as pd
import json
import subprocess
import sys
from db import init_db, get_latest_signals, get_signal_history, get_summary_stats

# Page Configuration
st.set_page_config(
    page_title="AI Trader - 4-Agent Swing Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Modern Theme with Glassmorphism Accent)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #141721 100%);
        border: 1px solid #2a2e3d;
        border-radius: 12px;
        padding: 200px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
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
st.sidebar.title("⚡ AI Trader Control Center")
st.sidebar.markdown("---")

st.sidebar.subheader("Run New Pipeline Analysis")
model_option = st.sidebar.radio(
    "Select Model:",
    ["Gemini 3.6 Flash", "gemma4:12b (Ollama Local)"],
    index=0
)

run_button = st.sidebar.button("🚀 Run 4-Agent Analysis", use_container_width=True)

if run_button:
    model_arg = "local" if "Ollama" in model_option else "gemini"
    st.sidebar.info(f"Launching pipeline with {model_option}...")
    with st.spinner("Analyzing market technicals, news catalysts, risk boundaries & bank ratings..."):
        try:
            # Run main.py as subprocess with selected model argument
            cmd = [sys.executable, "main.py", model_arg]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                st.sidebar.success("Pipeline Analysis Completed Successfully!")
                st.rerun()
            else:
                st.sidebar.error(f"Error running pipeline: {result.stderr}")
        except Exception as e:
            st.sidebar.error(f"Execution error: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Dashboard Data", use_container_width=True):
    st.rerun()

# Main App Layout
st.title("📈 Autonomous Stock Swing Trading Dashboard")
st.caption("Powered by 4-Agent Multi-Agent Architecture (Market, News, Risk & Analyst Agents) & SQLite")

# Summary KPI Cards
stats = get_summary_stats()
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Tracked Stocks", value=stats["total_tracked"])
with col2:
    st.metric(label="🟢 Buy Signals", value=stats["buy_count"])
with col3:
    st.metric(label="🔴 Sell Signals", value=stats["sell_count"])
with col4:
    st.metric(label="⚪ Hold Signals", value=stats["hold_count"])
with col5:
    st.metric(label="Last Analysis Run", value=stats["last_run"])

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📊 Latest Signals", "🔍 Stock Deep-Dive", "📜 Signal History Log"])

latest_signals = get_latest_signals()

with tab1:
    st.subheader("Latest Swing Trade Recommendations")

    if not latest_signals:
        st.info("No signal data found in database yet. Run a pipeline analysis from the sidebar to populate signals!")
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
            df_table = df_display[[
                "symbol", "decision", "confidence", "pe_and_peg",
                "risk_assessment", "investment_bank_coverage", "model_used", "timestamp"
            ]].rename(columns={
                "symbol": "Ticker",
                "decision": "Decision",
                "confidence": "Confidence",
                "pe_and_peg": "Fwd Valuation",
                "risk_assessment": "Risk Profile",
                "investment_bank_coverage": "Bank Coverage",
                "model_used": "Model",
                "timestamp": "Timestamp"
            })

            # Format dataframe display with badges
            def format_decision(val):
                if val == "BUY":
                    return "🟢 BUY"
                elif val == "SELL":
                    return "🔴 SELL"
                return "⚪ HOLD"

            df_table["Decision"] = df_table["Decision"].apply(format_decision)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
        else:
            st.warning("No signals match the selected filters.")

with tab2:
    st.subheader("Sub-Agent Intelligence & Synthesis Breakdown")

    if not latest_signals:
        st.info("No data available. Run an analysis to view stock details.")
    else:
        symbol_list = [s["symbol"] for s in latest_signals]
        selected_stock = st.selectbox("Select Ticker Symbol to Inspect:", options=symbol_list)

        stock_data = next((s for s in latest_signals if s["symbol"] == selected_stock), None)

        if stock_data:
            c1, c2, c3 = st.columns([1, 1, 2])
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
                st.markdown(f"**Model Used:** `{stock_data.get('model_used', 'N/A')}`")
                st.markdown(f"**Timestamp:** `{stock_data.get('timestamp', 'N/A')}`")

            st.markdown("---")

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("💡 Swing Setup Rationale")
                st.info(stock_data.get("reason") or "No rationale provided.")

                st.subheader("📰 Today's News Catalysts")
                st.write(stock_data.get("news") or "No news catalyst summary reported.")

            with col_right:
                st.subheader("🏦 Wall Street Bank Coverage")
                st.write(stock_data.get("investment_bank_coverage") or "No bank rating changes recorded.")

                st.subheader("⚠️ Risk Profile & Stop-Loss")
                st.warning(stock_data.get("risk_assessment") or "Low risk profile.")

                st.subheader("📊 Forward Valuation")
                st.write(f"Forward P/E Ratio: `{stock_data.get('pe_and_peg', 'N/A')}`")

            # Display Raw JSON details if available
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
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
