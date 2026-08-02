import streamlit as st
import pandas as pd
import json
from db import init_db, get_latest_signals, get_signal_history, get_summary_stats

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
💻 **Terminal Trigger Mode**
To run a new pipeline analysis, execute in your terminal:
- `python main.py gemini` (Gemini 3.1 Pro)
- `python main.py gemma` (Local gemma4:12b)
- `python main.py qwen` (Local qwen3:30b)
""")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Report Data", width="stretch"):
    st.rerun()

# Main App Layout
st.title("📈 Autonomous Stock Swing Trading Report")
st.caption("Reporting dashboard reading analysis results from SQLite (`trader.db`)")

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
    st.subheader("Sub-Agent Intelligence & Synthesis Breakdown")

    if not latest_signals:
        st.info("No data available. Run `python main.py` in terminal to generate stock details.")
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
