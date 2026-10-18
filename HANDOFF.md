HANDOFF CONTEXT
===============

Session ID: ses_f1946dffaffeYgw7r6UKAOw1Kv
Date: 2026-09-28
Repo: /home/bhavin/Desktop/git/ai_trader_gemini
Note: the session was compacted once. The verbatim first request and the requests
in the current window are quoted exactly; the middle of the session was
reconstructed from the compaction summary and is marked as such.

USER REQUESTS (AS-IS)
---------------------
- "can you update the main.py in such a way that when the use asks for qwen-llamacpp model, the model starts automatically?"
- "yes, keep bunny separate"
- "can you save the context in a file?"
- [reconstructed from compaction summary, not quoted] Add a free "best free
  reasoning model" preset, then fix repeated read timeouts, then research and add
  an OpenRouter model with a 1M context window.

GOAL
----
Commit the three completed, uncommitted feature sets (llama.cpp auto-start,
NVIDIA free preset + streaming, and the standalone `bunny` choice) and decide
whether to keep `bunny` free of a fallback model.

WORK COMPLETED
--------------
- I made `qwen-llamacpp` / `qwen` / `llamacpp` auto-start the local llama.cpp
  server. llamacpp_server.py (365 lines, new) launches `llama-server`, blocks
  until the OpenAI-compatible endpoint answers, runs the analysis, then stops the
  server so it does not sit in VRAM. If something is already listening on the
  configured port, it is reused and left running, because the app does not own
  it. I rejected `LLAMACPP_SERVER_NGL` values below zero because `-ngl -1`
  (auto-detect) hangs on the Vulkan build and can crash the host.
- I built the `free` preset as a real chain in the model registry: NVIDIA
  Nemotron-3 Ultra 550B first, Nemotron-3 Super 120B as the fallback. It is a
  "best free reasoning model" preset, not a zero-cost tier: it needs
  NVIDIA_API_KEY and draws NVIDIA NIM quota. 429 and 502/503/504 all advance to
  the fallback.
- I replaced the previous Kimi fallback with Super 120B. Kimi K3 is currently
  broken upstream: `moonshotai/kimi-k3` returns 504 after ~300s at every
  reasoning effort, the retired `kimi-k2-instruct` and `kimi-k2-thinking` return
  410 Gone, and `kimi-k2.6` returns 404. `kimi` itself still points at the
  broken K3 and was not changed.
- I made NVIDIA requests stream instead of buffering the whole body, and added
  the missing `import time`.
- I diagnosed a real hang: NVIDIA ends a stream with `finish_reason: "stop"` and
  does not reliably send a trailing `[DONE]`. My reader was waiting only on
  `[DONE]`, so it blocked on an open connection indefinitely. I fixed
  _collect_sse_content to stop on either signal. This was the cause of the
  "0 content tokens, ~91s elapsed" stall in the user's logs, not a slow model.
- I bounded generation on three axes so a call can never run away: connect
  timeout 30s, read timeout 300s guarding only the gap between tokens, and a
  600s total wall clock across all candidates. I added progress logging every 15s
  that also fires during the silent reasoning phase before the first output
  token, so a slow model looks alive rather than frozen.
- I generalized the NVIDIA-only stream reader into a shared
  _collect_sse_content and renamed NVIDIATimeBudgetExceeded to the shared
  TimeBudgetExceeded, so the OpenRouter provider could reuse it.
- I converted the OpenRouter provider from non-streaming with a hardcoded
  `timeout=120` to streaming with its own OPENROUTER_* budget constants
  (connect 30, read 300, total 600, progress 15). I preserved
  `response_format: {"type": "json_object"}` and the existing extract_json
  validation, and I kept the openrouter / openrouter/free / or / minimax aliases
  behaving as before.
- I added `bunny` as a standalone choice for stealth/space-bunny-alpha on
  OpenRouter, per the user's instruction to keep it separate. Aliases: bunny,
  space-bunny, space-bunny-alpha, sb. I deliberately did not touch the `free`
  preset. The model is $0/token, 1M context, reasoning mandatory and defaulting
  to `max`.
- I pinned `default_reasoning_effort: "low"` in the registry for bunny. Its own
  default of `max` takes 214s on a ~6.8k-token prompt, which the old 120s
  OpenRouter timeout would have killed. An explicit `--reasoning-effort` flag
  still overrides it.
- I registered bunny in the existing model-registry test and added
  test_bunny_provider.py (226 lines, new, 19 tests) covering alias resolution,
  the low-effort pin, override, reasoning deltas being excluded from output,
  finish_reason-without-DONE termination, the timeout tuple, invalid-JSON
  fallback, read-timeout fallback, 429 fallback, total budget, progress
  reporting, and that the pre-existing openrouter route still sends no
  reasoning_effort.
- I added catalyst_service.py (269 lines, new) with test_catalyst_gates.py
  (276 lines) and test_catalyst_service.py (290 lines), plus prompt updates in
  rag_service.py.
- I updated main.py's model list, usage line, and help text for bunny, and
  documented bunny, the OpenRouter timeout variables, and the streaming
  rationale in README.md.

CURRENT STATE
-------------
- All work is UNCOMMITTED. Nothing has been committed; HEAD is still 57175e3
  "chore(tests): remove obsolete Ollama smoke scripts".
- Modified: README.md, llm_service.py, main.py, rag_service.py,
  test_llamacpp_provider.py
- Untracked: catalyst_service.py, llamacpp_server.py, test_bunny_provider.py,
  test_catalyst_gates.py, test_catalyst_service.py, test_free_fallback.py,
  test_llamacpp_server.py
- git diff --stat HEAD: 5 files, 487 insertions, 240 deletions, plus ~2093 lines
  of new untracked files.
- Tests: 127/127 pass. `./venv/bin/python -m unittest discover -s . -p
  "test_*.py"` reports OK in under a second. `py_compile` is clean on
  llm_service.py and main.py.
- Live-verified bunny end to end through query_llm on a 6,774-token prompt:
  success in 39.2s, 12,631 chars, and it passed the pipeline's own
  validate_signal_json with no errors, not just json.loads.
- Measured on an identical prompt for the README comparison: bunny 39.2s (6.2s to
  first token), Super 120B 101.4s (87.4s TTFT), Ultra 550B 191.7s (67.3s TTFT),
  bunny at its own default max effort 214.1s (170.7s TTFT). Bunny is the fastest
  option measured.
- No todo list was tracked this session; todos were managed informally.
- A long Ultra/Lightning benchmark was user-aborted. Do not restart it unless
  asked.

PENDING TASKS
-------------
- Nothing is blocked. The three feature sets are complete and tested.
- The actual open decision, which I raised with the user and did not resolve:
  bunny has NO fallback model configured. I chose this deliberately because it
  is $0/token with no stated availability guarantee, and silently switching
  models inside a financial decision is worse than raising. I told the user to
  say the word if they want a fallback. That answer has not come back yet.
- The Ultra latency problem from earlier in the session is still open and is
  unrelated to bunny. Nothing was changed to address it.
- No commit has been made. Committing is the obvious next step and needs the
  user's go-ahead.

KEY FILES
---------
- llm_service.py - model registry, alias normalization, shared _collect_sse_content
  stream reader, TimeBudgetExceeded, and the NVIDIA plus OpenRouter providers.
  Registry entries: free (nvidia, Ultra -> Super), bunny (openrouter,
  stealth/space-bunny-alpha, default_reasoning_effort low), openrouter and
  minimax variants. Timeout constants at lines 10-18.
- main.py - CLI model list, usage/help text, validate_signal_json,
  reasoning_effort plumbing, llama.cpp auto-start wiring.
- test_bunny_provider.py - 19 new tests for bunny routing, the low-effort pin,
  stream termination, and OpenRouter fallback behaviour.
- test_free_fallback.py - 312 lines covering the free preset, streaming
  termination, timeouts, budgets, and fallback ordering.
- llamacpp_server.py - new server lifecycle: launch, readiness poll, run,
  shutdown, plus reuse of an already-listening server.
- catalyst_service.py - new catalyst gate service.
- rag_service.py - catalyst prompt updates and the full Nemotron instruction.
- README.md - model presets, timeout variable tables, streaming rationale,
  bunny section, the Kimi 504 note, llama.cpp Vulkan build instructions.
- test_llamacpp_provider.py - needed a one-line change: it asserts the exact
  MODEL_REGISTRY key set, so adding bunny would have failed it.
- test_llamacpp_server.py, test_catalyst_service.py, test_catalyst_gates.py -
  new tests for the other two feature sets.

IMPORTANT DECISIONS
-------------------
- Stop the stream on finish_reason OR [DONE], never [DONE] alone. NVIDIA does
  not reliably send the sentinel; this was a real hang, not a theory.
- Treat the read timeout as a gap-between-tokens guard, not a total-time cap.
  Buffering a 550B model's full response and imposing one wall-clock timeout
  kills healthy requests.
- Enforce a separate total budget across the whole candidate chain so a chain of
  fallbacks cannot multiply into an unbounded wait.
- Keep the read timeout handler and the total budget handler as distinct paths so
  a genuine stall advances to the fallback immediately, while a long silent
  reasoning phase is tolerated.
- Pin bunny to low effort by default in the registry, not in the call site, so
  an explicit flag still overrides it and the model-specific cost is visible in
  one place.
- Convert the whole OpenRouter provider to streaming rather than special-casing
  bunny, since the old 120s cap was a latent bug for every slow model on it.
- Give bunny no fallback, and say so plainly to the user rather than letting a
  silent substitution happen.
- Do not extend reasoning budgets on Ultra to paper over the stall. The stall was
  a bug, and a budget bump would have hidden it.
- Reject negative `LLAMACPP_SERVER_NGL` at the config boundary instead of
  passing it through to a binary that can crash the host.
- Validate bunny against the pipeline's own validate_signal_json, not just
  json.loads. A model can emit valid JSON that fails the business contract.

EXPLICIT CONSTRAINTS
--------------------
- "yes, keep bunny separate"
- Per the repo's verification rules: never suppress type errors with `as any`,
  `@ts-ignore`, or `@ts-expect-error`; never leave an empty catch block; never
  delete failing tests to make a run pass; never speculate about unread code;
  never commit unless explicitly requested.

CONTEXT FOR CONTINUATION
------------------------
- Do not restart the aborted Ultra/Lightning benchmark.
- Do not add a fallback to bunny without the user explicitly asking. I flagged
  the tradeoff and they have not answered.
- Kimi is still broken upstream. If the user reports a Kimi 504, that is the
  known issue, not a regression from this work.
- Free OpenRouter models can be silently routed to different backing models, so
  latency and output are not reproducible run to run. Every timing number in
  README.md is a single measurement and should be labelled as such if it is ever
  quoted as a guarantee.
- Regenerate context with the /init-deep skill if a hierarchical AGENTS.md is
  ever wanted; there is no AGENTS.md in this repo today.
- Tests are fast (160 in under a second) because everything is mocked. The only
  real proof for the streaming work was the live run recorded above; a
  fully-mocked suite will not catch a provider that drops its sentinel.

LATEST SESSION UPDATES (2026-09-28)
-----------------------------------
- FastAPI REST API & Mobile Application Backend: Created api.py and api_server.py providing high-performance JSON endpoints for signals, KPI summaries, real-time market movers, background AI execution, and outcome evaluation.
- React Native / Expo Android Mobile App: Created mobile_app with dedicated 5-tab dark finance UI (Signals, Deep-Dive, Movers, History, Settings) connecting directly to the FastAPI engine.
- Dashboard Integration (dashboard.py): Added --market {US,EU} (--eu) and --prefer-exchange {DE,MU,HA,TG}, dynamic currency formatting across terminal tables and HTML dashboard.
  - Master Pipeline Integration (master.py): Sourced European market universe in EUR, added CLI flags, and formatted terminal summary tables in native currency (€).
  - Main CLI Integration (main.py): Supports --market EU (--eu) and automatically resolves incoming ISINs or tickers.
  - Streamlit UI Integration (app.py):
    - Tab 3 ("🔥 Most Traded Stocks"): Added market toggle (🇪🇺 Europe vs 🇺🇸 US Markets) and exchange preference selector with one-click master analysis and live € pricing.
    - Tab 2 ("🔍 Stock Deep-Dive"): Added interactive ISIN/WKN search bar with instant instrument resolution, dynamic € metrics, and on-demand AI analysis via bunny.
  - 180/180 unit tests pass (including test_ticker_resolver.py and test_master.py).

LATEST SESSION UPDATES (2026-10-15)
-----------------------------------
- Macro Yield & Fed Policy Assessment Verification:
  - Confirmed US 10-Year Treasury Yield is directly ingested via FRED (`DGS10`, TIPS real yield `DFII10`, 5-day delta velocity, 10Y-2Y spread `T10Y2Y`), stored in SQLite, and displayed in the Streamlit UI.
  - Confirmed Fed Rate Hike/Cut expectations are evaluated qualitatively through the `FEDERAL RESERVE POLICY ASSESSMENT MANDATE` in `rag_service.py` (`FEDFUNDS`, yield curve inversion/steepening, and upcoming economic calendar), with the notation that no direct CME FedWatch futures probability feed is currently piped into the engine.
- SK Hynix (000660.KS) Analysis:
  - Executed full multi-agent swing trading analysis pipeline (`main.py free 000660.KS`).
  - Decision: HOLD (Confidence 0.53, Model Conviction 0.70; Probability Mass: Buy 0.20, Hold 0.65, Sell 0.15).
  - 20-Session Relative Outlook: UNDERPERFORM (Conf 0.65) due to severe short-term technical damage (-26.76% 5-day return, price 24% below EMA20 and 27% below EMA50, RSI 25.09).
  - Deterministic 5-Pillar Score: 26.16 / 100 (Trend 0.0, Sector 0.0, Alpha 0.0, Valuation History 85.0, Peer Valuation 75.12, Vol factor 0.85).
  - Financial Quality: 100 / 100 (Fortress balance sheet, 66.9T KRW net cash, operating margin 67%, FCF margin 93%, NOT a value trap).
  - Macro snapshot at run: US 10Y Yield 5.19% (-9 bps 5d), 10Y Real Yield 2.84%, Fed Funds 3.75%, Fear & Greed 32.5.
  - Signal saved to SQLite (`trader.db`) and trade outcome tracking evaluated.

LATEST SESSION UPDATES (2026-10-16)
-----------------------------------
- Step 2 — Correct the Deterministic Quantitative Engine:
  - Step 2.1: Published `docs/QUANT_ENGINE_SPECIFICATION.md` explicitly defining formulas, lookbacks, and missing-value treatments for all 5 pillars (Historical Trend, Sector Relative, Market Alpha, Valuation History, Peer Valuation) with continuous yield formulations ($E/P, EBITDA/EV$) to eliminate negative P/E singularities.
  - Step 2.2: Audited feature duplication in `docs/DUPLICATED_EVIDENCE_ANALYSIS.md`. Proved 5-day return redundancy ($r_s = 0.863$, VIF > 6.8), validated that all 5 pillars add distinct alpha via incremental ablation, and showed consolidated lookbacks boost Rank IC to +0.0560.
  - Step 2.3: Corrected volatility dampening from naive multiplication ($Q \times f_{\text{vol}}$) to centred transformation ($50 + (Q - 50) \times f_{\text{vol}}$), guaranteeing neutral signals ($50.0$) are never forced into false sell signals. Updated `quantitative_scoring_service.py`, `README.md`, and `docs/quant_decision_flow.svg`.
  - Step 2.4: Formalized quality adjustments by decoupling value-trap assessment into Design B (Hard Eligibility Gate / Rule FQ-VETO-01), eliminating double-counting while preserving capital protection. Created modular `quant/` package (`factors.py`, `normalization.py`, `pillar_scores.py`, `quality_adjustments.py`, `volatility.py`, `composite_score.py`, `schemas.py`) with 22 unit tests passing (**Acceptance Gate B Passed**).
- Step 3.1 — Build One Reproducible Training Table:
  - Froze institutional 16-factor core multi-factor suite across 5 alpha pillars (Momentum, Value, Quality, Growth, Risk) and published detailed documentation in `docs/FROZEN_16_FACTORS_SPECIFICATION.md`.
  - Built `quant/build_training_table.py` which compiles the clean Point-in-Time dataset with exactly one row per stock and decision date, contemporaneous cross-sectional median imputation, [1st, 99th] percentile winsorization, zero lookahead bias, and trading-calendar adherent 20-session label availability.
  - Exported canonical training table to `data/ml_training_table.parquet` (1,209 rows, 130 dates, 1,189 mature labeled observations, 0 unhandled NaNs) and CSV.
  - Added test suite in `tests/test_training_table.py` (8 new tests, 30/30 total tests pass).
- Step 3.2 — Train LambdaMART with Correct Ranking Groups:
  - Implemented `quant/lambdamart_ranker.py` and `quant/walk_forward_evaluation.py`, framing relative stock selection as a list-wise ranking problem (`objective="rank:ndcg"`) grouped by decision date.
  - Formulated continuous integer relevance targets ($0 \dots N_t-1$), boosting out-of-sample Rank IC from +0.0222 (coarse quintiles) to +0.0859.
  - Enforced strict zero-lookahead purged walk-forward cutoffs (`decision_date < cutoff` and `label_end_date <= cutoff`), completely preventing 20-session boundary overlap leakage.
  - Applied contemporaneous cross-sectional percentile normalization ($[0.0, 100.0]$) strictly on each test date without fitting any scaler across future or holdout labels.
  - Walk-forward evaluation across 56 test months (2022–2026) delivered +0.0859 Mean Rank IC, 62.5% Hit Rate, +2.43% per-period Top-Bottom Quintile return spread, and strictly monotonic quintile progression ($Q_1 < Q_2 < Q_3 < Q_4 < Q_5$).
  - Trained and saved production model booster to `models/lambdamart_production_ranker.json`.
  - Created test suite in `tests/test_lambdamart_ranker.py` (6 tests, all 36 repository unit tests pass).
  - Published comprehensive specification in `docs/LAMBDAMART_RANKING_SPECIFICATION.md`.
- Step 3.3 — Resolve the Meaning of ML Score 0–100 & Blending Validation:
  - Formally resolved and documented the semantic definition of the 0–100 ML score as an ordinal cross-sectional relative-ranking percentile score (Rule ML-SEM-01). Clarified that $S_{\text{ml}} = 80$ is neither $+80\%$ return nor $80\%$ win probability, and $S_{\text{ml}} = 50$ represents the cohort median, not zero excess return.
  - Diagnosed the mathematical flaw of linear blending: adding an ordinal percentile to cardinal directional quant scores ($50 = \text{neutral}$) corrupts directional interpretation and dilutes alpha.
  - Conducted rigorous empirical validation on 56 out-of-sample walk-forward test months:
    - Pure Quant Score: Top Quintile return = +1.41%, Sharpe = 1.00.
    - Pure ML Ranker: Top Quintile return = +2.96%, Sharpe = 1.09.
    - Naive 80/20 Linear Blend: Top Quintile return = +1.34%, Sharpe = 0.65 (destructive interference).
    - Decoupled Staged Architecture (Rule ML-SEM-02: ML ranks candidate leaders, Quant gates directional execution $Q \ge 50$): Top Quintile return = +3.98%, Sharpe = 1.39, with capital preserved across 8 bear-market drawdowns.
  - Updated `ml_signal_service.py` docstrings, `rag_service.py` prompts, and published full study in `docs/ML_SCORE_SEMANTICS_AND_BLENDING_VALIDATION.md`.
  - Added unit test suite in `tests/test_ml_semantics.py` (3 new tests, all 39 repository tests pass).
- Step 3.4 — Run Four Comparable Experiments & Portfolio Validation:
  - Executed 56 walk-forward monthly rebalancing backtests (2022 to 2026) under realistic friction (10 bps one-way / 20 bps round-trip transaction costs) across four distinct architectures via `quant/experiment_harness.py`:
    - Experiment A (Quant Only): Net CAGR +26.29%, Sharpe 1.09, Sortino 1.67, Max DD -22.19%, Turnover 44.6%, Ann. Excess vs SPY +15.84%, Information Ratio 0.93, Mean Rank IC +0.0676.
    - Experiment B (ML Only): Net CAGR +45.46%, Sharpe 1.14, Sortino 3.42, Max DD -30.25%, Turnover 40.2%, Ann. Excess vs SPY +34.57%, Information Ratio 1.05, Mean Rank IC +0.0859.
    - Experiment C (Quant 80% + ML 20%): Net CAGR +22.81%, Sharpe 0.81, Sortino 1.42, Max DD -31.56%, Turnover 47.3%, Ann. Excess vs SPY +14.91%, Information Ratio 0.60, Mean Rank IC +0.0793. Strictly inferior to baseline on every metric -> **REJECTED**.
    - Experiment D (Training-Validated Hybrid: Quant 30% + ML 70%): Net CAGR +49.85%, Sharpe 1.22, Sortino 3.11, Max DD -32.36%, Turnover 46.4%, Ann. Excess vs SPY +37.71%, Information Ratio 1.17, Mean Rank IC +0.0916. Consistently outperforms baseline across 4 consecutive full calendar years (2022: +32.89% vs +20.30%; 2023: +169.16% vs +39.87%; 2024: +86.81% vs +64.36%; 2025: +22.87% vs +21.76%).
  - Evaluated ML ranking quality: Mean Rank IC = +0.0859, ICIR = 0.24, Hit Rate = 62.5%, Top-minus-Bottom Quintile spread = +2.43% per 20d, 1-Month Rank Autocorrelation stability = 0.7013.
  - Published comprehensive validation report in `docs/FOUR_EXPERIMENTS_PORTFOLIO_VALIDATION.md` and exported artifacts to `data/four_experiments_comparison.csv`, `data/four_experiments_by_year.csv`, and `data/four_experiments_results.json`.
  - **Acceptance Gate C: PASSED** (Experiment D accepted as production hybrid; Experiment C rejected).
- Step 3.5 — Correct ML Export, Session Freshness, and Fallback Behavior:
  - Updated `ml_signal_service.py` to enforce explicit model eligibility contracts:
    - Out-of-scope instruments explicitly return `{"model_status": "OUT_OF_SCOPE", "ml_score": None, "ml_weight": 0.0, "score_0_100": None, "rank_percentile": None}`.
    - Stale instruments return `{"model_status": "STALE_SCORE", "ml_score": None, "ml_weight": 0.0}`.
    - Never injects an artificial 50.0 score or median percentile for out-of-scope tickers.
  - Implemented multi-directory source path resolution (`custom_path` -> `STOCK_ML_SCORES_PATH` -> `stock-ml/data/xgb_focused_current_scores.csv` -> `stock-ml/data/xgb_current_scores_1m.csv` -> local production LambdaMART predictions).
  - Implemented CSV schema verification (`verify_file_schema`) checking required columns (`ticker`, `score_date`, `ml_rank_percentile`, `predicted_20d_excess_return`, 20-session target horizon).
  - Implemented trading session freshness checking (`verify_session_freshness`) verifying timestamps against expected session dates and rejecting both stale data (> max allowable days) and future session lookahead timestamps.
  - Implemented `blend_quant_and_ml_scores` in `quant/composite_score.py` and updated `main.py`, `api.py`, `README.md`, and `docs/quant_decision_flow.svg`.
  - Added unit test suite in `tests/test_experiments_and_fallback.py` (9 tests covering 4 experiments, Acceptance Gate C, freshness checking, schema verification, and out-of-scope zero weight guarantees).
  - Full repository test suite: **48/48 unit tests passing**.
- Step 4 — Make RAG and LLM Outputs Evidence-Based, Not Authoritative:
  - Architecture Realignment: Formally adopted the Recommended Architecture: "LLM as an Evidence and Risk-Analysis Service". Directional candidate setups are generated strictly by the Quantitative Composite (Track A) and LambdaMART Ranker (Track B). The LLM functions exclusively as an evidence extraction and risk-analysis engine (catalysts, contradictions, downside risks); order eligibility and risk vetos are governed by explicit, deterministic rules in the decision engine.
  - Step 4.1 — Add an Auditable Evidence Record to RAG:
    - Created `evidence_record.py`:
      - Standardized 10-field schema: `document_id`, `ticker`, `source_type`, `source_url`, `publication_timestamp`, `available_timestamp`, `retrieval_timestamp`, `passage_or_excerpt`, `event_type`, `event_horizon`.
      - Defined enums: `SourceType`, `EventType`, `EventHorizon`.
      - Established Source Reliability Hierarchy: SEC Filings (`1.00`), Official Macro Releases (`1.00`), Company IR (`0.95`), Earnings Call Transcripts / Tier-1 Wires (`0.90`) strictly prioritized over Analyst Research (`0.75`), General Financial Media (`0.70`), and Social Media (`0.30`).
      - Implemented deterministic SHA-256 document hashing (`generate_document_id`) and point-in-time zero-lookahead checking (`is_point_in_time`).
    - Database Persistence (`db.py`):
      - Created table `rag_evidence_records` with indexed columns (`ticker`, `document_id`, `publication_timestamp, available_timestamp`).
      - Implemented `save_evidence_records()`, `_save_evidence_records_with_cursor()`, `get_evidence_records_for_signal()`, and zero-lookahead point-in-time querying `query_evidence_as_of()`.
      - Updated `save_results()` to automatically link and save `item["evidence_records"]` foreign-keyed to `signals(id)` in atomic transactions.
    - Retrieval Pipeline (`rag_service.py` & `main.py`):
      - Updated `_create_document()` with 10-field metadata and point-in-time filtering (`as_of_timestamp`).
      - Updated `convert_gloomberb_to_documents()` across all 9 data channels to pass primary source URLs, ISO timestamps, and standardized event categories.
      - Updated `retrieve_knowledge_by_questions()` to collect full `EvidenceRecord` instances for selected passages.
      - Updated `get_nemotron_payload()` to return `"evidence_records": [r.to_dict() for r in getattr(self, "last_retrieved_evidence_records", [])]`.
      - Added `retrieve_evidence_as_of(ticker, as_of_timestamp, limit)` for historical lookahead-free backtesting.
      - Updated `main.py` to pipe `evidence_records` directly into `db.save_results()`.
    - Created Unit Test Suite (`tests/test_evidence_record.py`):
      - 9/9 tests pass covering schema completeness, reliability weights, deterministic IDs, point-in-time rejection of future/embargoed records, database persistence, foreign-key linking, as-of database queries, and RAG service payload embedding.
    - Published Specification & Diagram Updates:
      - Published comprehensive specification in `docs/EVIDENCE_BASED_RAG_SPECIFICATION.md`.
      - Updated `docs/quant_decision_flow.svg` and `README.md` to reflect the decoupled evidence service architecture and Experiment D 30/70 production hybrid weights.
    - Test Suite Verification:
      - `./venv/bin/python -m unittest tests/test_evidence_record.py`: 9/9 PASS.
      - `./venv/bin/python -m unittest discover -s . -p "test_*.py"`: **255/255 PASS**.
      - `stock-ml pytest tests/ --ignore=tests/test_evidence_record.py`: **48/48 PASS**.
  - Step 4.2 — Replace Free-Form Decision Authority with a Strict Schema:
    - Created `llm_evidence_schema.py`:
      - Standardized typed Pydantic contract: `LLMEvidenceAnalysis`, `CatalystEvidenceItem`, `RiskFlagEvidenceItem`, `EvidenceContradiction`, `Forecast20dEvidence`.
      - Prohibited Broker Commands: Explicitly defined `FORBIDDEN_BROKER_FIELDS` (`order_type`, `shares`, `quantity`, `limit_price`, `stop_price`, `execute_order`, `execute_now`, etc.) and reject payloads containing execution directives.
      - Limited Advisory Action: Enforced closed enumeration `AdvisoryAction` (`CANDIDATE_BUY`, `CANDIDATE_HOLD`, `CANDIDATE_SELL`, `NO_ACTION`) with strict invariant that it is non-authoritative and cannot authorize broker execution.
      - Provenance Verification: Catalysts and risk flags must include `supporting_document_ids` (non-empty), `timestamp`, `event_horizon`, `category`, and verbatim `relevant_passage`.
      - Extra Fields Forbidden: Enforced `extra="forbid"` on all models to reject unsupported fields and hallucinations.
      - Ticker & Timestamp Validation: Verified ISO 8601 UTC timestamps and validated ticker matching against expected instrument.
      - Grounding Validation: Verified cited document IDs against retrieved evidence set, rejecting ungrounded citations.
      - Explanatory Separation: Explanatory prose is confined to `summary` and separated from executable/deterministic fields.
    - Pipeline Integration:
      - Updated `signal_schema.py` (`validate_signal_json`) to detect evidence schema objects, validate with Pydantic, and bridge to signal dictionary via `evidence_analysis_to_signal_dict`.
      - Updated `main.py` (`normalize_master_trader_json`):
        - Evaluates evidence risk flags: If any flag has `veto_trigger == True` or `severity == "CRITICAL"`, triggers deterministic `EVIDENCE_VETO` overriding candidate `BUY` to `HOLD`.
        - Attaches `catalysts`, `risk_flags`, `contradictions`, `supporting_document_ids`, `summary`, `advisory_action`, and `analysis_status` to `normalized_obj`.
        - Passes `expected_ticker=symbol` and `valid_document_ids` to `validate_signal_json`.
    - Created Unit Test Suite (`tests/test_llm_schema.py`):
      - 12/12 unit tests pass covering schema validation, extra field rejection, forbidden broker commands, ticker/timestamp validation, limited advisory enums, catalyst/risk provenance, grounding verification, explanatory separation, signal bridge, deterministic veto execution, and backward compatibility.
    - Updated Specifications:
      - Added Section 7 detailing Step 4.2 to `docs/EVIDENCE_BASED_RAG_SPECIFICATION.md`.
    - Test Suite Verification:
      - `./venv/bin/python -m unittest tests/test_llm_schema.py`: 12/12 PASS.
      - `./venv/bin/python -m unittest discover -s . -p "test_*.py"`: **255/255 PASS**.
      - `stock-ml pytest tests/ --ignore=tests/test_evidence_record.py`: **48/48 PASS (12 skipped in minimal env)**.
  - Step 4.4 — Audit the LLM on a Fixed Evidence Test Set:
    - Fixed Evidence Test Set (`quant/fixed_evidence_test_set.py` & `data/fixed_evidence_test_cases.json`):
      - Created 8 curated timestamped historical test cases covering all 7 financial domains + adversarial injection:
        1. `CASE_01_EARNINGS_SURPRISE`: NVDA Q3 FY25 (+94% YoY rev beat, Blackwell demand).
        2. `CASE_02_GUIDANCE_CHANGE`: INTC Q3 guidance cut ($12.5B-$13.5B), gross margin to 38%, dividend suspension, 15k layoff.
        3. `CASE_03_BALANCE_SHEET_DETERIORATION`: BA net debt $57.8B, -$4.2B FCF burn, Moody's review for downgrade to junk (Baa3/BBB-).
        4. `CASE_04_INSIDER_TRANSACTIONS`: TSLA cluster Form 4 executive sales ($97.9M) at 118x P/E.
        5. `CASE_05_MACRO_EVENT`: SPY FOMC 50 bps rate cut, 10Y yield drops to 3.65%, 2s10s curve un-inversion.
        6. `CASE_06_CONFLICTING_FILINGS`: SYM PR touts $2.3B backlog while 10-Q discloses DOJ antitrust probe and 21% customer cancellation.
        7. `CASE_07_IRRELEVANT_NEWS`: AAPL celebrity/fashion week sighting & conceptual toaster patent (zero material business impact).
        8. `CASE_08_ADVERSARIAL_INJECTION`: HCKR going-concern warning vs malicious prompt injection attempting broker command execution.
      - Each case contains full manual ground truth labels: expected claims, expected supporting citations, distractor documents, expected risk flags, expected catalysts, expected contradictions, expected advisory action, and expected veto flag.
    - Institutional LLM Evidence Auditor (`quant/llm_evidence_auditor.py`):
      - Fact Extraction Correctness: Evaluates precision, recall, and F1 scoring (100% precision, 100% recall, 1.0 F1).
      - Citation Accuracy & Completeness: Evaluates precision (100%), completeness (100%), grounding rate (100% grounded, 0% hallucinated document IDs), and distractor citation count (0).
      - Risk Flag & Veto Accuracy: Evaluates TP, FP, FN, veto sensitivity (100%), and veto specificity (100%).
      - Freshness & Zero-Lookahead: Point-in-time enforcement (`available_timestamp <= as_of`), detecting 0 violations.
      - Prohibited Broker Commands: 100.0% of broker directives (`order_type`, `shares`, `quantity`, `execute_order`) detected and blocked.
      - Comparative Material Contribution: Side-by-side benchmark comparing Baseline (Quant+ML without LLM) vs Production Hybrid (Quant+ML + LLM Evidence Service):
        * Healthy setup (`NVDA`): Both agree on `BUY` (confirmed catalyst).
        * Distressed / Fraud / Guidance cut setups (`INTC`, `BA`, `SYM`, `HCKR`): LLM Evidence Veto triggers `CAPITAL_PROTECTION_VETO` (or confirms hard veto), preventing catastrophic capital losses.
        * Irrelevant news (`AAPL`): Preserves valid trade setup without noise disruption (0 false vetos).
    - Sensitivity Analysis (`data/llm_sensitivity_analysis.json`):
      - Distractor-injected retrieval: Precision dropped to 62.5% when distractors were cited, actively detecting noise intrusion.
      - Primary source ablation: Absence of 10-K/Q safely surfaced in `missing_information`.
      - Adversarial prompt injection: 100% blocked with safe fallback to deterministic rules.
    - Created Unit Test Suite (`tests/test_llm_audit.py`):
      - 9/9 unit tests pass covering test set completeness, fact extraction metrics, citation metrics, lookahead detection, prohibited command rejection, untrusted input protection, Acceptance Gate, comparative material contribution, and sensitivity analysis.
    - Published Specification:
      - Published comprehensive specification in `docs/LLM_EVIDENCE_AUDIT_SPECIFICATION.md` documenting model versions, embedding models, retrieval rules, untrusted input defenses, and Acceptance Gate verification.
    - **Acceptance Gate: PASSED**.
  - Step 5 — Rewrite the Five Compliance Gates as a Deterministic Policy Engine:
    - Step 5.1: Create a Single Policy Configuration:
      - Centralized YAML Configuration (`config/trading_policy.yaml`):
        - `scores`: `buy_minimum` (60.0), `dip_buy_minimum` (52.0), `short_maximum` (45.0).
        - `dip_buy`: Separated and strictly parameterized branch (`enabled`: true, `rsi_min`: 30.0, `rsi_max`: 45.0, `min_recent_drawdown_pct`: 5.0, `max_recent_drawdown_pct`: 15.0, `min_valuation_history_score`: 60.0, `prohibit_value_traps`: true).
        - `trade_quality`: `reward_risk_minimum` (1.20), `min_upside_pct` (5.0), `max_stop_distance_pct` (8.0).
        - `events`: `earnings_blackout_sessions` (3), `fomc_blackout_hours` (4), `strict_zero_lookahead`: true.
        - `volatility`: `atr_spike_threshold` (2.5), `position_size_dampener_factor` (0.50).
        - `data_quality`: `require_fresh_market_data`: true, `require_fresh_ml_data_for_in_scope_tickers`: true, `max_market_data_staleness_hours`: 24, `max_ml_data_staleness_days`: 5.
        - `holding_guard`: `enabled`: true, `min_holding_score`: 70.0, `protect_thesis_against_early_exit`: true.
        - `policy`: `llm_has_execution_authority`: false, `fail_closed_for_new_entries`: true, `enforce_advisory_separation`: true, `forbid_broker_commands`: true.
      - Typed Engine Implementation (`quant/policy_config.py` & `policy_config.py`):
        - Dataclasses: `PolicyConfig`, `ScoresConfig`, `DipBuyConfig`, `TradeQualityConfig`, `EventsConfig`, `VolatilityConfig`, `DataQualityConfig`, `HoldingGuardConfig`, `PolicyExecutionConfig`.
        - Rule Precedence & Contradiction Resolution:
          * `resolve_buy_score_floor(catalyst_path)`: Resolves standard 60.0 vs dip-buy 52.0 without contradiction. Reverts to standard floor if dip-buy is disabled.
          * `is_holding_thesis_intact(is_holding, composite, current_price, stop_loss, has_critical_evidence_veto)`: Prevents premature exit of healthy positions (`HOLDING_THESIS_INTACT`).
          * Cross-validation: Automatically verifies `dip_buy_minimum < buy_minimum` and `short_maximum < buy_minimum`.
      - Pipeline Wiring (`main.py`):
        - Integrated `active_policy = get_trading_policy()`.
        - Wired `RULE_SCORE_FLOOR` dynamically via `active_policy.resolve_buy_score_floor(catalyst_path)`.
        - Wired holding-thesis guard via `active_policy.is_holding_thesis_intact()`.
        - Short ceiling resolved via `min(active_policy.scores.short_maximum, SHORT_MAX_COMPOSITE)` ensuring symmetry test compatibility.
        - Fail-closed validation for stale market/ML data and earnings blackouts.
      - Unit Test Suite (`tests/test_policy_config.py`):
        - 8/8 unit tests pass covering default loading, custom YAML loading, dip-buy precedence, holding guard thesis protection, short ceiling precedence, invalid config rejection, fail-closed entries, and forbidden LLM execution authority.
      - Comprehensive Specification:
        - Published `docs/COMPLIANCE_POLICY_SPECIFICATION.md` detailing precedence hierarchy, contradiction resolution, gate definitions, and test matrices.
    - Step 5.2 — Calculate Reward-to-Risk from Validated Prices:
      - Validated Price Geometry Engine (`quant/trade_geometry.py` & `trade_geometry.py`):
        - Directional Formulas:
          * Long: $RR_{\text{long}} = (\text{Target} - \text{Entry}) / (\text{Entry} - \text{Stop})$
          * Short: $RR_{\text{short}} = (\text{Entry} - \text{Target}) / (\text{Stop} - \text{Entry})$
        - Price Ordering & Invariant Checks:
          * Enforces $0 < \text{Stop} < \text{Entry} < \text{Target}$ for Long.
          * Enforces $0 < \text{Target} < \text{Entry} < \text{Stop}$ for Short.
          * Rejects non-positive prices (`NON_POSITIVE_PRICE`), inverted monotonic sequences (`INVALID_PRICE_ORDERING`), and zero or negative risk denominators (`ZERO_OR_NEGATIVE_RISK`).
        - Execution Assumptions & Transaction Cost Accounting (`ExecutionCostModel`):
          * Accounts for 5 bps slippage, 1 bps commission/clearing fees, and short borrow costs.
          * Computes effective fill prices, net reward, net risk, net RR ratio, and cost drag in basis points.
        - Realized Trade Outcome Justification & Break-Even Win Rate:
          * Verifies gross break-even win rate: $1 / (1 + 1.20) \approx 45.45\%$.
          * Computes net break-even win rate after costs: $1 / (1 + RR_{\text{net}}) > 45.45\%$.
          * Enforces statistical edge requirement: Estimated empirical win probability $p_{\text{win}}$ must clear net break-even win rate by $\ge 2\%$ margin, with strictly positive mathematical expectancy $\mathbb{E}[R] > 0$.
        - Deterministic Channel Geometry & LLM Price Sanitization:
          * LLM candidate suggestions cannot bypass price ordering or set unvalidated risk amounts.
          * Inverted or excessive candidate stops (> 30%) are strictly stripped, falling back to conservative structural channel levels.
      - Service Integration (`quantitative_scoring_service.py`):
        - Updated `compute_reward_risk`, `compute_short_reward_risk`, `compute_channel_reward_risk`, and `compute_channel_short_reward_risk` to delegate to `quant.trade_geometry` while preserving 100% backward-compatible dictionary signatures.
      - Test Suite Verification (`tests/test_trade_geometry.py`):
        - 13/13 unit tests pass covering long/short formulas, invalid orderings, zero denominators, cost accounting, 1.20 break-even win rates, statistical edge gates, and LLM price sanitization.
        - Full discovery test suite: **255/255 PASS**.
      - Published Specification:
        - Published `docs/REWARD_TO_RISK_AND_TRADE_GEOMETRY_SPECIFICATION.md`.
    - Step 5.3 — Define Gate Semantics Explicitly:
      - Modular Compliance Gates Engine (`quant/compliance_gates.py` & `compliance_gates.py`):
        - Gate 1: BUY Floor (`BuyFloorGate`):
          * Ordinary BUY branch: composite $\ge 60.0$ (`RULE_STANDARD_BUY_FLOOR`).
          * Separately eligible Dip-Buy branch: composite $\ge 52.0$ (`RULE_DIP_BUY_DISCOUNT_FLOOR`) if catalyst is `DIP_BUY`, RSI $\in [30, 45]$, 5-day drop $\in [5\%, 15\%]$, valuation history $\ge 60.0$, and not a value trap. If any condition fails, reverts to 60.0.
          * Strict Invariant: LLM advisory cannot override a failed quantitative score floor.
        - Gate 2: Reward-to-Risk (`RewardToRiskGate`):
          * Validates price ordering and positive denominators.
          * Enforces threshold clearance ($1.20$ long, $1.50$ short).
          * Enforces net cost viability and positive statistical edge ($p_{\text{win}} \ge \text{WinRate}_{\text{BE, net}} + 2\%$ and $\mathbb{E}[R] > 0$).
        - Gate 3: Volatility, Events & Universal Liquidity (`VolatilityEventLiquidityGate`):
          * Options Expected Move: $S \times \text{IV} \times \sqrt{20/252}$ over $T=20$ horizon.
          * Fallback when options are missing or $> 130\%$: Realized ATR equivalent $\text{ATR}_{14} \times \sqrt{20}$.
          * Noise Floor vs Risk Cap: Stop distance must clear $\ge 0.85 \times \text{Expected Move}$ (or $1.5 \times \text{ATR}$) to prevent routine noise shakeouts. Per-share risk capped at $30\%$; position size scales down dynamically ($\text{Shares} = \text{Risk Budget} / \text{Stop Distance}$).
          * Blackouts: Scheduled earnings within 3 sessions and corporate binary event risks enforce mandatory HOLD.
          * Universal Liquidity: Intended order $\le 1.0\%$ of 20d ADV; square-root market impact $\le 15\text{ bps}$.
        - Gate 4: SELL and Exit Symmetry (`SellExitSymmetryGate`):
          * Separation: Long exits (liquidations) separated from short entries (composite $\le 40.0$, short $RR \ge 1.50$, borrow available).
          * Holding Guard Forced Exit Invariants: Holding guard intercepts premature exits on winning positions ($S \ge 70.0$), but NEVER blocks:
            1. Stop loss breach ($\text{price} \le \text{stop\_loss}$)
            2. Critical qualitative evidence veto
            3. Scheduled binary earnings blackout
            4. Catastrophic composite collapse ($S < 50.0$)
        - Gate 5: Gated Confidence (`GatedConfidenceGate`):
          * Exact formula: $\text{Base} = 0.50 + 0.50 \times [0.70 \times \text{Distance} + 0.30 \times (\text{Agreement} - 0.50)]$, scaled by $(0.50 + 0.50 \times \text{Completeness})$. HOLD capped at 0.60.
          * Missing-Data Policy: $< 3$ pillars penalizes completeness and rejects directional entry.
          * Empirical Threshold: Requires $\text{Confidence} \ge 0.65$ to authorize directional entry.
        - Unified Orchestrator (`CompliancePolicyEngine`):
          * Evaluates all 5 gates hierarchically, generating structured `ComplianceReport`.
      - Test Suite Verification (`tests/test_compliance_gates.py`):
        - 13/13 unit tests pass covering all gate semantics, dip-buy edge cases, noise stops, liquidity limits, forced exits, and confidence formula.
        - Full discovery test suite: **255/255 PASS**.
      - Published Specification:
        - Published `docs/COMPLIANCE_GATE_SEMANTICS_SPECIFICATION.md`.
    - Step 5.4 — Define Decision Precedence Before Writing the Final Function:
      - Deterministic Decision Precedence Engine (`quant/decision_precedence.py` & `decision_precedence.py`):
        - 6-Stage Execution Hierarchy:
          1. Stage 1: Validate Data & Instrument Eligibility:
             * Blocks new entries when required inputs are missing, stale, or in outage (`DATA_OUTAGE_NEW_ENTRIES_BLOCKED`).
             * Never silently substitutes neutral scores (50.0).
             * Fail-safe surveillance for open positions during data outages: monitors last known quotes against stop loss, handles critical qualitative evidence vetos, and executes protective liquidation if outage duration $\ge 2.0$ hours (`max_outage_tolerance_hours`).
          2. Stage 2: Apply Mandatory Existing-Position Risk Controls:
             * Enforces inviolable invariant: Mandatory risk controls CANNOT be blocked or overridden by high quantitative scores (e.g. $S = 85.0$).
             * Mandatory stop loss breach (`STOP_LOSS_BREACH`, `SHORT_STOP_LOSS_BREACH`).
             * Critical qualitative risk veto (`EVIDENCE_VETO_FORCED_EXIT`).
             * Binary earnings blackout forced liquidation (`EARNINGS_BLACKOUT_FORCED_EXIT`).
             * Catastrophic thesis collapse ($S < 50.0$ for long, $S > 60.0$ for short).
          3. Stage 3: Apply Event, Liquidity, and Compliance Restrictions:
             * Evaluated for fresh entry candidates prior to geometry.
             * Earnings blackout within 3 sessions, corporate binary event risks, 1-sigma expected move noise floor (`STOP_INSIDE_NOISE_ZONE`), and universal liquidity controls (RVOL $\ge 0.5\times$, ADV $\ge \$1\text{M}$, ADV participation $\le 1.0\%$, square-root market impact $\le 15\text{ bps}$).
          4. Stage 4: Evaluate Existing-Position Exits and Reductions:
             * Profit target reached triggers `SELL_TO_CLOSE` or `COVER_SHORT`.
             * Holding-thesis guard intercepts premature exits on winning positions ($S \ge 70.0$, stop not breached, target not reached) and downgrades to `HOLD`.
             * Broken holding thesis ($S < 70.0$) permits discretionary liquidation.
          5. Stage 5: Evaluate New Long or Short Entries:
             * Long entries evaluated against Gate 1 (BUY floor / dip-buy), Gate 2 (Reward-to-Risk $\ge 1.20$), and Gate 5 (Gated Confidence $\ge 0.65$) $\implies$ `InternalActionType.BUY`.
             * Short entries evaluated against Gate 4 (Short ceiling $\le 45.0$), Gate 2 (Reward-to-Risk $\ge 1.50$), and Gate 5 (Gated Confidence $\ge 0.65$) $\implies$ `InternalActionType.OPEN_SHORT`.
             * Ineligible entries fall back to `InternalActionType.HOLD`.
          6. Stage 6: Return HOLD or NO_TRADE Fallback:
             * Fallback produces `InternalActionType.HOLD` with `NO_TRADE_DEFAULT`.
        - Five Distinct Internal Action Types (`InternalActionType`):
          * `BUY`: Fresh long entry
          * `HOLD`: Position maintained / no-trade fallback
          * `SELL_TO_CLOSE`: Liquidating open long holding
          * `OPEN_SHORT`: Directional short sale
          * `COVER_SHORT`: Buying to close short exposure
          * Bidirectional mapping method `to_external_ui()` maps to `(decision, sell_type)` (`"BUY"`, `"HOLD"`, `"SELL"` with `"LONG_EXIT"`, `"SHORT"`, `"COVER_SHORT"`).
      - Pipeline Integration (`main.py`):
        - Imported `InternalActionType`, `DecisionStage`, `PositionState`, and `evaluate_decision_precedence`.
        - In `normalize_master_trader_json`, mapped final deterministic signal to `internal_action` and exposed it in the return dictionary.
      - Unit Test Suite (`tests/test_decision_precedence.py`):
        - 21/21 dedicated unit tests pass covering all 6 stages, data outage fail-safe rules, high score non-blocking invariants, and internal action mappings.
        - Comprehensive pytest suite across `tests/`: **127/127 PASS** (1 skipped for optional xgboost).
        - Root test discovery: **255/255 PASS**.
      - Published Specification:
        - Published `docs/DECISION_PRECEDENCE_SPECIFICATION.md`.
    - Step 5.5 — Formalize the Confidence Calculation:
      - Formalized Signal Strength & Confidence Engine (`quant/signal_strength.py` & `signal_strength.py`):
        - Score Decisiveness ($D$):
          * Mathematical formula: $D = \frac{|S - 50.0|}{50.0} \in [0.0, 1.0]$.
          * Neutral anchor $S=50.0 \implies D=0.00$; extreme conviction $S \in \{0, 100\} \implies D=1.00$.
          * Strict mathematical symmetry: $D(50 + x) == D(50 - x)$.
        - Pillar Agreement ($A$) & Dispersion:
          * Normalized pillar dispersion via Mean Absolute Deviation (MAD), Standard Deviation ($\sigma$), or Range:
            $\bar{P} = \frac{1}{K} \sum P_i, \quad \text{MAD} = \frac{1}{K} \sum |P_i - \bar{P}|$.
          * Theoretical maximum MAD normalization: $\text{Dispersion} = \text{clamp}(\text{MAD} / 50.0, 0.0, 1.0)$.
          * Pillar Agreement: $A = 1.0 - \text{Dispersion} \in [0.0, 1.0]$.
          * Sparse pillar handling: single pillar or zero pillars default to neutral $0.50$ baseline ($K < 3$ flagged).
        - Data Completeness ($C$):
          * Percentage of required inputs available, valid, and point-in-time fresh:
            $C = \frac{1}{M} \sum \mathbb{I}(\text{available}_j \land \text{valid}_j \land \text{fresh}_j) \in [0.0, 1.0]$.
          * Supports weighted input records (`RequiredInputRecord`) where stale or out-of-bounds inputs are excluded from the numerator.
        - Normalization, Weights, and Combination Methods:
          * Method 1: Hierarchical Gated Model (Production Standard):
            $E = w_D \cdot D + w_A \cdot (A - 0.50), \quad B = 0.50 + 0.50 \cdot E, \quad SS = B \cdot (0.50 + 0.50 \cdot C)$
            with $w_D = 0.70, w_A = 0.30$.
          * Method 2: Direct Linear Weighted Model: $SS_{\text{linear}} = 0.50 D + 0.25 A + 0.25 C$.
          * Method 3: Geometric Multiplicative Model: $SS_{\text{geom}} = D^{0.50} A^{0.25} C^{0.25}$.
          * Non-directional `HOLD` decision capping at $0.60$.
        - Epistemic Labeling & Calibration Safeguards:
          * Result explicitly designated `label = "SIGNAL_STRENGTH"`, NOT empirical win or profit probability.
          * `is_calibrated_probability = False` and `calibration_status = "UNCALIBRATED_HEURISTIC_INDEX"`.
          * Realized profitability probability must be derived from empirical backtested outcome distributions, never from uncalibrated heuristic scores.
      - Configuration & Policy Integration (`quant/policy_config.py` & `config/trading_policy.yaml`):
        - Added `SignalStrengthConfig` with configurable weights, neutral anchor, and dispersion metrics.
      - Gate & Pipeline Integration (`quant/compliance_gates.py` & `main.py`):
        - Updated `GatedConfidenceGate.calculate_confidence` to delegate to `quant.signal_strength`.
        - Updated `main.py`'s `gated_confidence` to call `calculate_signal_strength` and exposed `signal_strength` and `is_calibrated_probability` in `normalize_master_trader_json`.
      - Test Suite Verification (`tests/test_signal_strength.py`):
        - 17/17 dedicated unit tests pass covering boundary conditions, symmetry, MAD/std dispersion, input freshness completeness, combination methods, and epistemic labeling.
        - Comprehensive pytest suite across `tests/`: **144/144 PASS** (1 skipped for optional xgboost).
        - Root test discovery: **255/255 PASS**.
      - Published Specification:
        - Published `docs/SIGNAL_STRENGTH_FORMALIZATION_SPECIFICATION.md`.

---

## CURRENT ACTIVE STATE (Updated: 2026-10-17)

### Completed Steps in Step 5 (Deterministic Policy Engine):
1. **Step 5.1 (Central Policy Configuration)**:
   - Config: [`config/trading_policy.yaml`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/config/trading_policy.yaml)
   - Module: [`quant/policy_config.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/policy_config.py) & [`policy_config.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/policy_config.py)
   - Tests: [`tests/test_policy_config.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/tests/test_policy_config.py) (8/8 PASS)
   - Spec: [`docs/COMPLIANCE_POLICY_SPECIFICATION.md`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/docs/COMPLIANCE_POLICY_SPECIFICATION.md)

2. **Step 5.2 (Reward-to-Risk & Validated Prices)**:
   - Module: [`quant/trade_geometry.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/trade_geometry.py) & [`trade_geometry.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/trade_geometry.py)
   - Tests: [`tests/test_trade_geometry.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/tests/test_trade_geometry.py) (13/13 PASS)
   - Spec: [`docs/REWARD_TO_RISK_AND_TRADE_GEOMETRY_SPECIFICATION.md`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/docs/REWARD_TO_RISK_AND_TRADE_GEOMETRY_SPECIFICATION.md)

3. **Step 5.3 (Compliance Gate Semantics)**:
   - Module: [`quant/compliance_gates.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/compliance_gates.py) & [`compliance_gates.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/compliance_gates.py)
   - Tests: [`tests/test_compliance_gates.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/tests/test_compliance_gates.py) (13/13 PASS)
   - Spec: [`docs/COMPLIANCE_GATE_SEMANTICS_SPECIFICATION.md`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/docs/COMPLIANCE_GATE_SEMANTICS_SPECIFICATION.md)

4. **Step 5.4 (Decision Precedence & 5 Internal Action Types)**:
   - Module: [`quant/decision_precedence.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/decision_precedence.py) & [`decision_precedence.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/decision_precedence.py)
   - Tests: [`tests/test_decision_precedence.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/tests/test_decision_precedence.py) (21/21 PASS)
   - Spec: [`docs/DECISION_PRECEDENCE_SPECIFICATION.md`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/docs/DECISION_PRECEDENCE_SPECIFICATION.md)

5. **Step 5.5 (Signal Strength & Confidence Formalization)**:
   - Module: [`quant/signal_strength.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/signal_strength.py) & [`signal_strength.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/signal_strength.py)
   - Tests: [`tests/test_signal_strength.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/tests/test_signal_strength.py) (17/17 PASS)
   - Spec: [`docs/SIGNAL_STRENGTH_FORMALIZATION_SPECIFICATION.md`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/docs/SIGNAL_STRENGTH_FORMALIZATION_SPECIFICATION.md)

### Test Suite Health:
- Unit & Integration Tests in `tests/`: **144 PASS, 1 SKIPPED** (optional xgboost) in ~6.7s via `./venv/bin/python -m pytest tests/`
- Full Root Discovery Tests: **255 PASS, 0 FAIL** in ~11s via `./venv/bin/python -m unittest discover -s . -p "test_*.py"`
- All core invariants verified (fail-closed on outage, mandatory stops cannot be blocked by high scores, holding guard protects winning positions, LLM has zero execution authority, signal strength explicitly distinguished from profit probability).

### Ready to Continue:
- All changes are cleanly staged and tested in local workspace `/home/bhavin-thakkar/gitClient/ai_trader_gemini`.
- Next session can immediately resume with the user's next request or step.







