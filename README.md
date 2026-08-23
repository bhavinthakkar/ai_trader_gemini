# ⚡ Gloomberb Multi-Agent AI Trader & Dual-Horizon RAG Analysis Engine

An autonomous, multi-agent financial reasoning system designed for deterministic stock swing trading analysis. Powered by **NVIDIA Nemotron-3 Super 120B**, **Deterministic 5-Pillar Quantitative Scoring**, **Multi-Query Dual-Horizon FastEmbed RAG**, the **official Gloomberb CLI**, and a **16-Channel Institutional Data Pipeline**.

---

## 🏗️ System Architecture & Workflow

The platform operates on a **Deterministic quantitative-first architecture** paired with **Multi-Query Dual-Horizon Hybrid RAG**. Numerical market metrics are computed strictly outside the LLM to prevent hallucinations, while qualitative text feeds are retrieved using a multi-factor multiplicative ranking engine.

```
                                    ┌────────────────────────┐
                                    │   FastEmbed (384-dim)  │
                                    │  BAAI/bge-small-en-v1.5│
                                    └───────────┬────────────┘
                                                │
         ┌──────────────────────────────────────┴──────────────────────────────────────┐
         │                                                                             │
 ┌───────┴───────────────────────┐                                     ┌───────────────┴──────────────────────────┐
 │   Gloomberb Terminal Engine   │                                     │    Institutional Data Engine             │
 ├───────────────────────────────┤                                     ├──────────────────────────────────────────┤
 │ 1. News & Catalysts           │                                     │ 11. Investor-Relations (IR) Website      │
 │ 2. SEC EDGAR Filings (CLI)    │                                     │ 12. SEC EDGAR Direct Submissions (API)   │
 │ 3. Financial Statements       │                                     │ 13. Earnings Call Transcripts & Guidance │
 │ 4. Options Flow & Volatility  │                                     │ 14. Official Corporate Press Releases    │
 │ 5. Insider/Institutional %    │                                     │ 15. Reputable News (Reuters/Bloomberg)   │
 │ 6. Peer Relative Multiples    │                                     │ 16. Macro US Treasury Yield Curve (FRED) │
 │ 7. Ticker Architecture        │                                     └──────────────────────────────────────────┘
 │ 8. Macro Fear & Greed Index   │
 │ 9. Economic Events Calendar   │
 │ 10. Sector ETF Benchmarks     │
 └───────────────┬───────────────┘
                 │
                 ├──────────────────────────────────────────────┐
                 │                                              │
                 ▼                                              ▼
┌──────────────────────────────────┐          ┌──────────────────────────────────┐
│  Deterministic 5-Pillar Engine   │          │ Multiplicative Hybrid RAG        │
│  (Trend, Sector, Alpha,          │          │ (Sim x Reliability x Importance  │
│   Valuation History, Peer Val)   │          │  x Recency Weight)               │
└────────────────┬─────────────────┘          └────────────────┬─────────────────┘
                 │                                              │
                 │     ┌──────────────────────────────────┐     │
                 └────►│  Dual-Horizon Partitioning       │◄────┘
                       │  - CURRENT CONTEXT (24h/7d)      │
                       │  - HISTORICAL CONTEXT (10-K/Multi)│
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────┐
                       │ NVIDIA Nemotron-3 Super 120B     │
                       │ (Data Producer)                  │
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────┐
                       │ Python JSON Schema Validator     │
                       │ (normalize_master_trader_json)    │
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────┐
                       │ Python Telegram Formatter        │
                       │ (format_telegram_digest)         │
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────┐
                       │ Telegram API (Display Only)      │
                       └──────────────────────────────────┘
```

---

## 🧮 1. Deterministic 5-Pillar Quantitative Scoring Engine

To prevent the LLM from turning isolated positive facts into an unearned BUY decision, quantitative scores (0–100) are computed mathematically outside the model using standard financial logic:

$$\text{Composite Score} = 0.25(\text{Trend}) + 0.20(\text{Sector}) + 0.20(\text{Alpha}) + 0.15(\text{Valuation History}) + 0.20(\text{Peer Valuation})$$

* **Trend Score (25%)**: Evaluates 5-day return velocity, RSI14, and EMA20/EMA50 alignment.
* **Sector Relative Score (20%)**: Evaluates stock 5-day velocity vs. SPDR Sector ETF benchmark (`XLK`, `XLC`, `XLY`, etc.).
* **Market Alpha Score (20%)**: Evaluates stock 5-day return relative to S&P 500 (`SPY`).
* **Valuation History Score (15%)**: Evaluates current Forward P/E vs. company 3-year historical average P/E.
* **Peer Valuation Score (20%)**: Benchmarks Forward P/E, P/S, EV/EBITDA, and Price/FCF against direct industry peers (e.g. NVDA vs AMD/AVGO).

---

## ⚡ 2. Multiplicative Multi-Factor RAG Ranking Formula

Text passage retrieval uses a pure multiplicative scoring formula:

$$\text{Final RAG Score} = \text{Semantic Similarity} \times \text{Source Reliability} \times \text{Event Importance} \times \text{Recency Weight}$$

### **Source Reliability Hierarchy**
* **`1.00`**: Official SEC EDGAR Filings (Form 10-K, 10-Q, 8-K, Form 4) → **MAXIMUM AUTHORITY**
* **`0.95`**: Official Company Investor Relations (IR) & Corporate Press Releases → **VERY HIGH AUTHORITY**
* **`0.90`**: Tier-1 Wires (Reuters, Bloomberg) & Earnings Call Transcripts → **HIGH AUTHORITY**
* **`0.75`**: Wall Street Analyst Equity Research → **MODERATE AUTHORITY**
* **`0.70`**: Secondary Financial Media (MarketWatch, CNBC, Yahoo) → **SECONDARY AUTHORITY**
* **`0.30`**: Social Media & Retail Buzz → **LOW CONVICTION CHATTER**

### **Event Importance Hierarchy**
* **`1.00`**: CEO/Executive Resignations, Guidance Revisions, Form 10-K/10-Q/8-K, Earnings Transcripts
* **`0.90`**: Official Company IR Announcements
* **`0.85`**: SEC Form 4 Insider Trading Transactions
* **`0.75`**: Routine Product Press Releases & Analyst Upgrades/Downgrades
* **`0.70`**: Secondary Financial Media News
* **`0.30`**: Social Media Sentiment

---

## ⏳ 3. Dual-Horizon Temporal RAG Partitioning

To avoid mixing short-term 7-day catalysts with multi-year historical filings, RAG vector retrieval is partitioned into two temporal horizons:

### **HORIZON A: CURRENT CONTEXT (Last 24h / 7d / Latest Earnings)**
- *Sub-Question 1*: What recent material events changed for `{symbol}` in the last 7 days?
- *Sub-Question 2*: What are the primary short-term bullish and bearish catalysts?
- *Sub-Question 3*: What updated management guidance was issued in the latest earnings release?

### **HORIZON B: LONGER-TERM HISTORICAL CONTEXT (Prior Filings & Multi-Year Patterns)**
- *Sub-Question 4*: What is the longer-term historical business trend and execution pattern for `{symbol}`?
- *Sub-Question 5*: What historical valuation concerns or structural headwinds have persisted over time?
- *Sub-Question 6*: How do historical macro interest rate cycles compare to current conditions?

---

## 🛡️ 4. Decoupled Pipeline Contract Architecture

To ensure strict system reliability and guarantee that raw LLM text is never formatted directly into notification alerts:

```text
  RAG Engine
      │
      ▼
  Nemotron 120B (Data Producer)
      │
      ▼
  JSON Schema Validator & Normalizer (Python Contract)
      │
      ▼
  Telegram Formatter (Python Contract)
      │
      ▼
  Telegram API (Display Only)
```

1. **Nemotron (Data Producer)**: Synthesizes structured technicals and retrieved dual-horizon RAG evidence into raw JSON.
2. **JSON Schema Validator (`normalize_master_trader_json`)**: Intercepts the LLM response, validates field types, clamps confidence scores $[0.0, 1.0]$, normalizes BUY/SELL/HOLD decisions, and injects safe defaults.
3. **Telegram Formatter (`format_telegram_digest`)**: Accepts strictly validated Python dictionaries and constructs formatted Markdown.
4. **Telegram API (Display Only)**: Dispatches sanitized, pre-formatted messages to subscriber channels.

---

## 🔄 Execution Workflow

1. **Structured Data Extraction (`MarketAgent`)**: Sourcing price action, RSI14, EMA20/50, ATR, and 5-day relative alpha vs SPY.
2. **Gloomberb CLI Ingestion (`GloomberbService`)**: Fetching 10 subcommands (news, SEC filings, financials, options flow, insider % profile, peers, sector ETFs, fear/greed, econ calendar, yield curve).
3. **Institutional Data Sourcing (`InstitutionalDataService`)**: Ingesting IR RSS releases, direct SEC EDGAR API filings, earnings call transcripts, press releases, Tier-1 financial media, and FRED yield curves.
4. **Deterministic Quantitative Scoring (`QuantitativeScoringService`)**: Calculating the explicit 5-pillar composite quantitative score (0-100).
5. **Dual-Horizon Multiplicative RAG Indexing (`RAGService`)**: Indexing qualitative document chunks, embedding with FastEmbed `BAAI/bge-small-en-v1.5`, and retrieving top passages across `CURRENT CONTEXT` vs `HISTORICAL CONTEXT`.
6. **Reasoning Synthesis (`LLMService`)**: Prompting NVIDIA Nemotron-3 Super 120B to synthesize structured metrics and dual-horizon textual evidence.
7. **Storage & Telegram Distribution**: Storing structured JSON records in SQLite (`trading_analysis.db`) and sending digests to Telegram.

---

## 📋 Required JSON Output Schema

```json
{
  "stock": "NVDA",
  "buy_score": 0.20,
  "hold_score": 0.65,
  "sell_score": 0.15,
  "decision": "HOLD",
  "confidence": 0.70,
  "bull_case": "NVDA shows strong valuation alignment with history (85.0 score) and peers (74.76 score), supported by exceptional revenue growth (85.2% YoY)...",
  "bear_case": "Despite robust fundamentals, NVDA exhibits weak technical and relative performance: trend score (55.36) and sector relative score (26.25)...",
  "key_risk": "Primary risk is persistent failure to outperform sector benchmarks and generate market alpha...",
  "missing_information": "Lacks specific forward guidance quantitatives from latest earnings call..."
}
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bhavinthakkar/ai_trader_gemini.git
   cd ai_trader_gemini
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration (`.env`)**:
   Create a `.env` file in the root directory:
   ```env
   NVIDIA_API_KEY=nvapi-...
   GEMINI_API_KEY=AIzaSy...
   TELEGRAM_CHANNEL_API_TOKEN=bot...
   FINNHUB_API_KEY=...
   NEWS_API_KEY=...
   FMP_API_KEY=...
   ```

5. **Gloomberb CLI Installation**:
   Ensure official `gloomberb` binary is installed at `~/.local/bin/gloomberb`.

---

## 🚀 Usage Guide

### **Run Pipeline with Nemotron-3 Super 120B (NVIDIA Cloud)**
```bash
./venv/bin/python main.py nemotron AAPL
```
```bash
./venv/bin/python main.py nemotron NVDA,META,TSLA
```

### **Run Pipeline with Gemini 3.1 Pro**
```bash
./venv/bin/python main.py gemini AAPL
```

### **Run Pipeline with Local 2-Stage Ollama Chain**
```bash
./venv/bin/python main.py twostage NVDA
```

---

## 🛠️ Technology Stack

* **LLM Reasoning**: NVIDIA Nemotron-3 Super 120B (`nvidia/nemotron-3-super-120b`), Gemini 3.1 Pro, Ollama Qwen 2.5 14B / Qwen 3 30B.
* **Vector Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5`, 384-dimensional dense vectors).
* **CLI Terminal Feed**: Official `gloom-sh/gloomberb` CLI.
* **Macro Data**: FRED API (US Treasury Yield Curve) & CNN Fear & Greed Index.
* **Database & UI**: SQLite3 (`trading_analysis.db`), Streamlit dashboard.

