# ⚡ Gloomberb Multi-Agent AI Trader & Dual-Horizon RAG Analysis Engine

An autonomous, multi-agent financial reasoning system designed for deterministic stock swing trading analysis. Powered by **Moonshot AI Kimi-K3**, **NVIDIA Nemotron-3 Ultra 550B & Super 120B**, **Deterministic 5-Pillar Quantitative Scoring**, **Multi-Query Dual-Horizon FastEmbed RAG**, the **official Gloomberb CLI**, and a **16-Channel Institutional Data Pipeline**.

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
                       │ SQLite & Reporting Layer         │
                       │ (trader.db & Web / Mobile Apps)  │
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
  SQLite Database & Live Dashboards (Streamlit & Mobile)
```

1. **Nemotron (Data Producer)**: Synthesizes structured technicals and retrieved dual-horizon RAG evidence into raw JSON.
2. **JSON Schema Validator (`normalize_master_trader_json`)**: Intercepts the LLM response, validates field types, clamps confidence scores $[0.0, 1.0]$, normalizes BUY/SELL/HOLD decisions, and injects safe defaults.
3. **SQLite & API Layer**: Saves structured analysis to `trader.db` and serves real-time REST data via FastAPI.
4. **Web & Mobile Dashboards**: Visualizes signals on Streamlit (`app.py`) and native Android (`mobile_app`).

---

## 🔄 Execution Workflow

1. **Structured Data Extraction (`MarketAgent`)**: Sourcing price action, RSI14, EMA20/50, ATR, and 5-day relative alpha vs SPY.
2. **Gloomberb CLI Ingestion (`GloomberbService`)**: Fetching 22+ official CLI channels (real-time quotes, historical price series, annual/quarterly financial statements, valuation multiples & fundamentals, options chains & flow dynamics, Form 4 insider transactions, institutional 13F holdings, Wall Street analyst consensus & price targets, earnings calendar & revision momentum, quarterly earnings surprise history, SEC EDGAR filings, breaking news catalysts, CNN Fear & Greed, Treasury yield curve, FRED macro series, economic calendar, benchmark US indices, market breadth & active movers, SPDR sector ETFs, 1Y price correlation engine, multi-symbol comparison, watchlists and portfolios).
3. **Institutional Data Sourcing (`InstitutionalDataService`)**: Ingesting IR RSS releases, direct SEC EDGAR API filings, earnings call transcripts, press releases, Tier-1 financial media, and FRED yield curves.
4. **Deterministic Quantitative Scoring (`QuantitativeScoringService`)**: Calculating the explicit 5-pillar composite quantitative score (0-100).
5. **Dual-Horizon Multiplicative RAG Indexing (`RAGService`)**: Indexing qualitative document chunks, embedding with FastEmbed `BAAI/bge-small-en-v1.5`, and retrieving top passages across `CURRENT CONTEXT` vs `HISTORICAL CONTEXT`.
6. **Reasoning Synthesis (`LLMService`)**: Prompting NVIDIA Nemotron-3 Super 120B to synthesize structured metrics and dual-horizon textual evidence.
7. **Storage & Dashboard Distribution**: Storing structured JSON records in SQLite (`trader.db`) and serving web & mobile dashboards.

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
OPENROUTER_API_KEY=sk-or-v1-...
NVIDIA_READ_TIMEOUT=300
NVIDIA_CONNECT_TIMEOUT=30
NVIDIA_TOTAL_TIMEOUT=600
NVIDIA_PROGRESS_INTERVAL=15
OPENROUTER_READ_TIMEOUT=300
OPENROUTER_CONNECT_TIMEOUT=30
OPENROUTER_TOTAL_TIMEOUT=600
OPENROUTER_PROGRESS_INTERVAL=15
   FINNHUB_API_KEY=...
   NEWS_API_KEY=...
   FMP_API_KEY=...
   LLAMACPP_BASE_URL=http://127.0.0.1:11434/v1
   LLAMACPP_MODEL=qwen2.5
   LLAMACPP_API_KEY=
   LLAMACPP_CONTEXT_TOKENS=8192
   LLAMACPP_MAX_TOKENS=2048
   LLAMACPP_PROMPT_SAFETY_TOKENS=256
   LLAMACPP_TEMPERATURE=0.2
   LLAMACPP_CONNECT_TIMEOUT=5
   LLAMACPP_READ_TIMEOUT=900
   LLAMACPP_WRITE_TIMEOUT=30
   LLAMACPP_POOL_TIMEOUT=10
   LLAMACPP_SERVER_BIN=~/Desktop/git/Bonsai-demo/bin/vulkan/llama-server
   LLAMACPP_GGUF_PATH=~/models/qwen2.5-14b-instruct-q4_k_m.gguf
   LLAMACPP_SERVER_LOG=~/models/llamacpp_server.log
   LLAMACPP_SERVER_NGL=99
   LLAMACPP_SERVER_PARALLEL=1
   LLAMACPP_SERVER_ALIAS=qwen2.5
   LLAMACPP_STARTUP_TIMEOUT=300
   LLAMACPP_SHUTDOWN_TIMEOUT=30
   ```

   `LLAMACPP_API_KEY` may be omitted when the local server does not require authentication. The application sends requests to the OpenAI-compatible endpoint `<LLAMACPP_BASE_URL>/chat/completions`.

   The last seven variables control **auto-start**. They are only read when a `llama.cpp` alias is selected, and all of them are optional: `LLAMACPP_SERVER_BIN` and `LLAMACPP_GGUF_PATH` fall back to auto-discovery, and the remaining values fall back to the defaults shown above. `LLAMACPP_SERVER_NGL` must stay explicit — `-ngl -1` (auto-detect) hangs on the Vulkan build and can crash the host, so the value is rejected if it is negative. Extra slots (`LLAMACPP_SERVER_PARALLEL` > 1) exhaust UMA memory and severely slow Vulkan. `LLAMACPP_SERVER_ALIAS` must match `LLAMACPP_MODEL` so the client and the server agree on the model name.

5. **Gloomberb CLI Installation**:
   Ensure official `gloomberb` binary is installed at `~/.local/bin/gloomberb`.

6. **Mobile App Dependencies (Expo / React Native)**:
   Ensure Node.js (v18+) is installed, then install the mobile app packages:
   ```bash
   cd mobile_app
   npm install
   cd ..
   ```

---

## 🚀 Usage Guide

### **Run Pipeline with Moonshot AI Kimi-K3 (NVIDIA Cloud)**
```bash
./venv/bin/python main.py kimi AAPL
```
```bash
./venv/bin/python main.py kimi NVDA,META,TSLA
```
```bash
# Optional tuning: adjust temperature and reasoning effort ('low', 'medium', 'high', 'max')
./venv/bin/python main.py kimi NVDA --temperature 1.0 --reasoning-effort max
```

### **Run Pipeline with Nemotron-3 Ultra 550B / Super 120B (NVIDIA Cloud)**
```bash
./venv/bin/python main.py nemotron AAPL
```
```bash
./venv/bin/python main.py nemotron NVDA,META,TSLA
```

### **Run Pipeline with Nemotron-3 Ultra 550B + Super 120B Fallback (`free`)**
```bash
./venv/bin/python main.py free AAPL
```
```bash
./venv/bin/python main.py free NVDA,TSLA
```

`free` is a "best free reasoning model" preset rather than a zero-cost tier: it needs `NVIDIA_API_KEY` and draws on NVIDIA NIM quota, not OpenRouter credits. It calls Nemotron-3 Ultra 550B first and, if that call stalls, transparently retries against Nemotron-3 Super 120B (`nvidia/nemotron-3-super-120b-a12b`) instead of waiting out a second Ultra attempt. Rate limits (429) and 502/503/504 gateway errors advance to the same fallback. The active model is printed in the run log, and the model actually used is recorded on the analysis record.

NVIDIA and OpenRouter responses are consumed as a stream, so the request timeout only guards the gap *between* tokens rather than total generation time. A large reasoning model routinely spends minutes generating after a sub-second time-to-first-token, and buffering the whole response made the old non-streaming timeout kill healthy requests.

NVIDIA terminates a stream with `finish_reason: "stop"` and does not reliably send a trailing `[DONE]` sentinel, so the reader stops on either signal. Relying on `[DONE]` alone left the client blocked on an open connection indefinitely. Progress is logged every `NVIDIA_PROGRESS_INTERVAL` seconds, including during the long silent reasoning phase before the first output token, so a slow model is visibly alive rather than looking frozen.

Generation is bounded on three axes so a call can never run away:

| Variable | Default | Guards |
|---|---|---|
| `NVIDIA_CONNECT_TIMEOUT` | 30s | TCP/TLS establishment |
| `NVIDIA_READ_TIMEOUT` | 300s | silence *between* tokens (a true stall) |
| `NVIDIA_TOTAL_TIMEOUT` | 600s | total wall clock across all candidate models |
| `OPENROUTER_READ_TIMEOUT` | 300s | silence between tokens (OpenRouter) |
| `OPENROUTER_TOTAL_TIMEOUT` | 600s | total wall clock across OpenRouter candidates |

When the total budget is hit the current model is abandoned and the fallback is tried; if none remain, the call raises. Lower `REASONING_BUDGET` (default 16000) if you want shorter generations.

> Note: `kimi` currently points at `moonshotai/kimi-k3`, which the NVIDIA gateway is returning `504` for (after ~300s, at every reasoning effort). The retired `moonshotai/kimi-k2-instruct` and `kimi-k2-thinking` return `410 Gone`, and `moonshotai/kimi-k2.6` returns `404`. Use `free` or `super` until Kimi is served again.

To reach OpenRouter's free-model router instead, pass `openrouter` (or `openrouter/free`).

### **Run Pipeline with Space Bunny Alpha (`bunny`, free, 1M context)**
```bash
./venv/bin/python main.py bunny ORCL
```
```bash
./venv/bin/python main.py bunny ORCL,NVDA
```

`bunny` calls `stealth/space-bunny-alpha` on OpenRouter and needs `OPENROUTER_API_KEY`. Pricing is **$0 per token** with a 1M-token context window, so it costs nothing but consumes no NVIDIA quota either.

Its reasoning is **mandatory** and defaults to `max` effort, which is slow — roughly **214s** on a ~6.8k-token prompt versus **39s** at `low` effort. The client therefore pins `reasoning_effort=low` for this model. Override it when you want deeper reasoning and can accept the latency:
```bash
./venv/bin/python main.py bunny ORCL --reasoning-effort high
```

On an identical prompt, for reference: `bunny` 39.2s, Nemotron-3 Super 120B 101.4s, Nemotron-3 Ultra 550B 191.7s.

The model is anonymous, about a week old, unmoderated, and not version-pinned, so outputs are not reproducible or auditable. It is exposed as a separate opt-in choice and does not alter the `free` preset. Because it is free with no stated guarantee of continued availability, it has no fallback model configured — a failure raises rather than silently switching.

### **Run Pipeline with OpenRouter Free Models Router (`openrouter/free`)**
```bash
./venv/bin/python main.py openrouter AAPL
```
```bash
./venv/bin/python main.py openrouter NVDA,TSLA
```

### **Run Pipeline with Gemini 3.1 Pro**
```bash
./venv/bin/python main.py gemini AAPL
```

### **Run Pipeline with Local Qwen 2.5 14B (llama.cpp + Vulkan)**
Selecting a `llama.cpp` alias starts the local server for you. The pipeline launches `llama-server`, blocks until the OpenAI-compatible endpoint answers, runs the analysis, and then stops the server so it does not sit in VRAM between runs. If a server is already listening on the configured port, it is reused and left running, because the application does not own it.

The only prerequisite is a `llama-server` binary and a GGUF model. Both are auto-discovered (`LLAMACPP_SERVER_BIN`, then `LLAMACPP_GGUF_PATH`, then `$PATH`); override them with the `LLAMACPP_SERVER_BIN` and `LLAMACPP_GGUF_PATH` variables documented in the environment section above.

For a local llama.cpp checkout on Debian/Ubuntu, install the build prerequisites (including `glslc` and the Vulkan/SPIR-V development packages required by your distribution), then configure and build with Vulkan enabled:
```bash
sudo apt install cmake libvulkan-dev glslc
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j
```

Use any of the local aliases:
```bash
./venv/bin/python main.py qwen NVDA
./venv/bin/python main.py llamacpp NVDA
./venv/bin/python main.py qwen-llamacpp NVDA
```

These aliases send JSON chat-completion requests to `${LLAMACPP_BASE_URL:-http://127.0.0.1:11434/v1}` and automatically use a compact prompt profile sized for the local 8K context. The client serializes requests because the server has one slot and enforces a 2048-token output reserve plus a safety margin.

The auto-started server is launched with the equivalent of:
```bash
./llama.cpp/build/bin/llama-server \
  -m /path/to/qwen2.5-14b-q4_k_m.gguf \
  --alias qwen2.5 \
  --host 127.0.0.1 --port 11434 \
  -c 8192 -np 1 -ngl 99 --flash-attn on
```

Startup is gated on `LLAMACPP_STARTUP_TIMEOUT` (default 300s, which covers a cold Vulkan model load). If the binary exits during startup, the error includes its exit code and the tail of `LLAMACPP_SERVER_LOG`. To review the current portfolio with the same local model:
```bash
./venv/bin/python main.py portfolio qwen
```

### **Run Portfolio Risk Review (Investment-Committee Memo)**
```bash
./venv/bin/python main.py portfolio gemini
```
Runs a portfolio-level risk review using the holdings from the Gloomberb CLI portfolio list. It injects the live Fed/rate macro outlook (FRED yield curve, real yields, fed funds rate, yield velocity, CFTC COT — the same `interest_rate_outlook` block the single-stock workflow uses) into the portfolio payload, then analyzes it with the `PORTFOLIO_SYSTEM_INSTRUCTION` (concentration/exposure mapping, stress tests, Fed policy assessment, diversification gaps, resilience options). Results are saved to the `portfolio_reviews` table and rendered in the Streamlit dashboard's "📁 Portfolio Risk Review" tab.

### **📱 Run AI Trader Mobile Application (React Native & Expo)**

The project includes a cross-platform mobile client (`mobile_app/`) built with React Native and Expo (SDK 57). The mobile application communicates with the FastAPI REST API (`api.py` / `api_server.py`) on port `8000` to deliver real-time swing trade alerts, quantitative scores, stock deep-dives, screener tools, and trade history directly to your Android or iOS device.

#### **1. Prerequisites**
* **Expo Go**: Install the official **Expo Go** app on your phone from the [Google Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent) (Android) or [Apple App Store](https://apps.apple.com/app/expo-go/id982107779) (iOS).
* **Node.js**: Ensure Node.js (v18+) is installed on your host machine.
* **Dependencies**: Run `cd mobile_app && npm install && cd ..` if not already installed.
* **Network**: For LAN mode, ensure your phone and computer are connected to the same Wi-Fi network.

#### **2. Quick Start: Automated Mobile Launcher (Recommended)**
The simplest way to start the mobile ecosystem is using `mobile_launcher.py`:

```bash
./venv/bin/python mobile_launcher.py
```

This single command:
1. Automatically verifies if the FastAPI backend (`http://localhost:8000/api/health`) is running; if inactive, starts `api_server.py` in the background as a detached daemon.
2. Configures the environment and boots the Expo Metro bundler in LAN mode.
3. Renders a QR code and connection URL in your terminal.

**To open the app on your phone:**
1. Open **Expo Go** on Android (or the default **Camera** app on iOS).
2. Scan the QR code displayed in your terminal.
3. The JavaScript bundle will compile and launch the AI Trader app immediately.

#### **3. Tunnel Mode (Bypassing Router Isolation & Firewalls)**
If your phone is on a separate subnet, cellular data, or your Wi-Fi router enforces client isolation (common on guest or corporate networks), use tunnel mode:

```bash
./venv/bin/python mobile_launcher.py --tunnel
```

* You can also press `s` inside the running Metro terminal to toggle between LAN and Tunnel mode.
* To clear Metro bundler cache if encountering bundle state issues:
  ```bash
  ./venv/bin/python mobile_launcher.py --clear
  # or combine with tunnel mode:
  ./venv/bin/python mobile_launcher.py --tunnel --clear
  ```

#### **4. Standalone Backend & Manual Metro Workflow**
If you prefer running the backend and bundler in separate terminal windows:

* **Terminal 1 — FastAPI Mobile Backend (`api_server.py`)**:
  ```bash
  # Start backend daemon on port 8000
  ./venv/bin/python api_server.py start

  # Check server status, local/LAN IPs, and OpenAPI documentation
  ./venv/bin/python api_server.py status

  # Stop or restart the backend server
  ./venv/bin/python api_server.py stop
  ./venv/bin/python api_server.py restart
  ```
  Interactive Swagger documentation is accessible at `http://localhost:8000/docs`.

* **Terminal 2 — Expo Metro Bundler**:
  ```bash
  cd mobile_app
  npx expo start --lan
  # or with tunnel:
  npx expo start --tunnel
  ```

#### **5. Configuring the Backend URL in the Mobile App**
The mobile application defaults to connecting to port `8000` on your host's local area network IP (`http://<LAN_IP>:8000`). If your IP changes or you are using an ngrok / custom tunnel:
1. Tap the **⚙️ Settings** tab at the bottom of the mobile app.
2. Enter your workstation's IP or tunnel URL (e.g., `http://192.168.1.150:8000`).
3. Tap **Update API URL**. The app immediately verifies backend health and updates the status indicator (🟢 **CONNECTED**).

#### **6. Mobile Application Features**
* **⚡ Signals Tab**: View active swing trading signals, filter by decision (BUY / HOLD / SELL / ALL), search by symbol or catalyst, adjust recency window (1D, 7D, 30D, 90D, All), and monitor aggregated KPI cards (Signal counts, Bullish/Bearish distribution, Average Confidence).
* **🔬 Deep Dive Tab**: Select any stock or search any ticker to review 5-pillar quantitative scores (Trend, Sector, Alpha, Valuation History, Peer Valuation), full bull/bear cases, key risks, price geometries, and trigger an on-demand multi-agent synthesis.
* **📊 Movers Tab**: Live volume screener identifying top gainers, losers, and volume leaders across US markets and European (gettex / XETRA) exchanges.
* **📜 History Tab**: Complete historical archive of past signals and outcome tracking metrics.
* **⚙️ Settings Tab**: Real-time backend connection health monitor, editable API base URL, and manual trigger for trade outcome evaluations.

#### **7. Troubleshooting**
* **"Failed to download remote update" / Connection Timeout**: Your router may block peer-to-peer traffic between devices. Restart with `./venv/bin/python mobile_launcher.py --tunnel`.
* **Backend Status Red / "Cannot connect to server"**:
  - Run `./venv/bin/python api_server.py status` to check if FastAPI is running.
  - Verify that your PC firewall allows inbound connections on port `8000`.
  - Check the IP entered under the mobile app's **Settings** tab.
* **Metro Bundler Cache Issues**: Run `./venv/bin/python mobile_launcher.py --clear` or `npx expo start -c` in `mobile_app/`.

---

### **💻 Run Streamlit Web Dashboard**

In addition to the mobile app, you can launch the interactive browser dashboard:

```bash
# Start Streamlit dashboard on port 8501
./venv/bin/python streamlit_server.py start

# Check status or stop the server
./venv/bin/python streamlit_server.py status
./venv/bin/python streamlit_server.py stop
```

Access the dashboard locally at `http://localhost:8501` or via local network at `http://<LAN_IP>:8501`.

---

### **🍓 Raspberry Pi & Linux Autostart (systemd)**

To keep the FastAPI REST backend and Streamlit dashboard running 24/7 across reboots and automatically restart them on crash:

```bash
# Install and enable all services on boot
sudo bash systemd/install_services.sh

# Monitor service status and logs
sudo systemctl status ai-trader-api
sudo systemctl status ai-trader-web
journalctl -u ai-trader-api -f
journalctl -u ai-trader-web -f

# Uninstall services
sudo bash systemd/uninstall_services.sh
```

---

## 🛠️ Technology Stack

* **LLM Reasoning**: Moonshot AI Kimi-K3 (`moonshotai/kimi-k3` via NVIDIA NIM), NVIDIA Nemotron-3 Ultra 550B (`nvidia/nemotron-3-ultra-550b-a55b`) & Super 120B (`nvidia/nemotron-3-super-120b`), the `free` preset (Ultra 550B first, Super 120B fallback on stall), Space Bunny Alpha (`bunny` via OpenRouter, $0/token, 1M context), OpenRouter Free Models Router (`openrouter/free`, 200k context window), Gemini 3.1 Pro, and local Qwen 2.5 14B through Vulkan-enabled llama.cpp (`qwen` / `llamacpp` / `qwen-llamacpp`, 8K compact profile).
* **Vector Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5`, 384-dimensional dense vectors).
* **CLI Terminal Feed**: Official `gloom-sh/gloomberb` CLI.
* **Macro Data**: FRED API (US Treasury Yield Curve) & CNN Fear & Greed Index.
* **Database & UI**: SQLite3 (`trader.db`), Streamlit web dashboard (`app.py`), React Native & Expo mobile application (`mobile_app`), and FastAPI REST backend (`api.py`).

