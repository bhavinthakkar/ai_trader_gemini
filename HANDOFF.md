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


