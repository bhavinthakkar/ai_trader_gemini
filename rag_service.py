import os
import json
import datetime
import numpy as np
from typing import List, Dict

# FastEmbed Dense Vector Embeddings
from fastembed import TextEmbedding

# LangChain Document Splitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Quantitative 5-Pillar Scoring Engine
from quantitative_scoring_service import QuantitativeScoringService


def compute_recency_score(published_at_str: str, time_horizon: str = "CURRENT") -> float:
    """
    Calculates Horizon-Aware Recency Score with Recency Floors:
    - CURRENT Horizon: Strong decay (1.00 <=1d, 0.90 <=7d, 0.75 <=30d, 0.60 <=90d, floor 0.40).
    - HISTORICAL Horizon: Weak decay (1.00 <=30d, 0.95 <=180d, 0.90 <=365d, floor 0.85).
    """
    if not published_at_str or published_at_str == "N/A":
        return 0.95 if time_horizon == "HISTORICAL" else 0.85
    try:
        pub_dt = datetime.datetime.strptime(published_at_str[:10], "%Y-%m-%d")
        now = datetime.datetime.now()
        days_diff = (now - pub_dt).days

        if time_horizon == "HISTORICAL":
            if days_diff <= 30:
                return 1.00
            elif days_diff <= 180:
                return 0.95
            elif days_diff <= 365:
                return 0.90
            else:
                return 0.85  # Recency Floor 0.85 for historical filings/profile
        else:
            if days_diff <= 1:
                return 1.00
            elif days_diff <= 7:
                return 0.90
            elif days_diff <= 30:
                return 0.75
            elif days_diff <= 90:
                return 0.60
            else:
                return 0.40  # Recency Floor 0.40 for news
    except Exception:
        return 0.95 if time_horizon == "HISTORICAL" else 0.85


def compute_hybrid_score(sim_score: float, reliability: float, importance: float, recency: float) -> float:
    """
    Computes Multiplicative Multi-Factor Hybrid RAG Score:
    Final Score = Semantic Similarity x Source Reliability x Event Importance x Recency Weight
    """
    sim_clamped = max(0.01, sim_score)
    return sim_clamped * reliability * importance * recency


class RAGService:
    """
    Metadata-Aware Hybrid Vector RAG Retrieval Engine for Nemotron-3 Super 120B.
    Combines FastEmbed Dense Semantic Search with Source Reliability, Document Importance, Recency Decay,
    and Deterministic Quantitative 5-Pillar Scoring.
    """

    NEMOTRON_SYSTEM_INSTRUCTION = """
You are a senior Master Trader & Quantitative Portfolio Manager operating with Nemotron 3 Super intelligence.

You are provided with a high-conviction market analysis payload assembled via Metadata-Aware Hybrid Vector RAG and Deterministic Quantitative 5-Pillar Scoring:
1. STRUCTURED MARKET & QUANTITATIVE DATA:
   - Price, RSI14, EMA20, EMA50, ATR volatility parameters.
   - Forward P/E, Trailing P/E, P/S, EV/EBITDA, P/FCF.
   - Deterministic 5-Pillar Scores: Trend (25%), Sector (20%), Market Alpha (20%), Valuation History (15%), Peer Valuation (20%), and Weighted Composite Score (0-100).
2. CATEGORIZED RETRIEVED EVIDENCE (RAG): Qualitative text passages retrieved across 6 explicit sub-questions (Material Events, Bullish Drivers, Bearish Risks, Guidance Changes, Valuation Concerns, Macro Risks).

CRITICAL QUANTITATIVE SCORE MANDATE:
- ANCHOR YOUR DECISION AND SCORES STRICTLY AROUND THE DETERMINISTIC COMPOSITE QUANTITATIVE SCORE:
  - Composite Score >= 70.0: Strong quantitative alignment for BUY (confirm with fundamental RAG evidence).
  - Composite Score 45.0 - 69.9: Neutral / Mixed alignment. Default to HOLD unless an extraordinary high-reliability SEC/Earnings catalyst exists.
  - Composite Score < 45.0: Weak / Overvalued alignment. Default to SELL or HOLD.
- DO NOT turn isolated positive facts into an unearned BUY decision if the composite quantitative score is neutral or weak.

EXPLICIT SOURCE RELIABILITY HIERARCHY MANDATE:
- Every retrieved passage contains a [Metadata] header specifying its source and Reliability Score (1.00 to 0.30):
  - 1.00: Official SEC Filings (Form 10-K, 10-Q, 8-K, Form 4) -> MAXIMUM AUTHORITY
  - 0.95: Official Company IR & Corporate Press Releases -> VERY HIGH AUTHORITY
  - 0.90: Tier-1 Wires (Reuters, Bloomberg) & Earnings Transcripts -> HIGH AUTHORITY
  - 0.75: Wall Street Analyst Equity Research -> MODERATE AUTHORITY
  - 0.70: Secondary Financial Media (MarketWatch, CNBC, Yahoo) -> SECONDARY AUTHORITY
  - 0.30: Social Media & Retail Forums -> LOW CONVICTION CHATTER
- EVIDENCE AUTHORITY RULE: Higher reliability sources (SEC Filings @ 1.00, Company IR @ 0.95, Tier-1 Wires @ 0.90) strictly override lower reliability sources (General Media @ 0.70, Social Media @ 0.30). Never base a thesis on low-reliability chatter when contradicted by official SEC filings or IR releases.

Output MUST be a valid JSON object matching the exact requested schema.

Schema:
{
  "stock": "Ticker Symbol",
  "decision": "BUY|SELL|HOLD",
  "confidence": 0.70,
  "buy_score": 0.20,
  "hold_score": 0.65,
  "sell_score": 0.15,
  "horizon_days": 10,
  "quant_score": 64.8,
  "pillar_scores": {
    "trend": 55.36,
    "sector": 26.25,
    "alpha": 61.70,
    "valuation_history": 85.00,
    "peer_valuation": 74.76
  },
  "bull_case": [
    "Key bullish thesis point 1 synthesizing technical alpha, peer valuation discount, or earnings growth",
    "Key bullish thesis point 2 detailing short-term RAG catalysts"
  ],
  "bear_case": [
    "Key bearish thesis point 1 detailing relative benchmark underperformance or valuation premium",
    "Key bearish thesis point 2 detailing structural headwinds from historical filings"
  ],
  "key_risks": [
    "Primary downside risk factor 1",
    "Stop-loss or macro execution trigger risk"
  ],
  "missing_information": [
    "Missing forward guidance metrics from latest transcript",
    "Underspecified capex or inventory details"
  ],
  "data_completeness": 0.87
}
"""

    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        print("[Metadata-Hybrid RAG] Loading FastEmbed BAAI/bge-small-en-v1.5 384-dim dense embedding model...")
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def convert_gloomberb_to_documents(self, gloomberb_payload: Dict, technical_data: Dict = None, institutional_data: Dict = None) -> List[Document]:
        """
        Converts qualitative textual data feeds strictly into LangChain Documents.
        Numerical indicators (RSI, EMA, ATR, Price, P/E, Yields, Options IV) are EXCLUDED from vector embedding
        and passed directly as structured data to the LLM.
        Applies Explicit Source Reliability Hierarchy:
        - SEC Filings: 1.00
        - Company IR & Press Releases: 0.95
        - Reuters / Bloomberg / Transcripts: 0.90
        - Analyst Research: 0.75
        - General Financial Media: 0.70
        - Social Media: 0.30
        """
        docs = []
        symbol = gloomberb_payload.get("symbol", "N/A")
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        # 1. Gloomberb News Catalysts -> Horizon: CURRENT
        for item in gloomberb_payload.get("news", []):
            src = item.get('source', 'Gloomberb News')
            rel_score = 0.90 if any(w in src.lower() for w in ["reuters", "bloomberg", "wsj", "gloomberb"]) else 0.70
            text = f"[{src} Catalyst] {item.get('title', '')}. Details: {item.get('summary', '')}"
            docs.append(Document(page_content=text, metadata={
                "ticker": symbol,
                "source": src,
                "document_type": "AggregatedNews",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": rel_score,
                "importance": 0.65,
                "time_horizon": "CURRENT"
            }))

        # 2. SEC EDGAR Filings (10-K = HISTORICAL, 10-Q / 8-K = CURRENT)
        for filing in gloomberb_payload.get("filings", []):
            form_type = str(filing.get('form', '')).upper()
            horizon = "HISTORICAL" if "10-K" in form_type else "CURRENT"
            text = f"[SEC EDGAR Form {filing.get('form')} ({filing.get('date')})] {filing.get('summary')}"
            docs.append(Document(page_content=text, metadata={
                "ticker": symbol,
                "source": "SEC EDGAR",
                "document_type": f"Form-{filing.get('form', 'Filing')}",
                "published_at": filing.get('date', today_str),
                "effective_date": filing.get('date', today_str),
                "reliability": 1.00,
                "importance": 1.00,
                "time_horizon": horizon
            }))

        # 3. Form 4 Insider Transactions & 13F Institutional Holdings -> Horizon: HISTORICAL
        insider = gloomberb_payload.get("insider_institutional", {})
        if insider:
            inst_pct = insider.get("institutional_ownership_pct", "N/A")
            txs = insider.get("insider_transactions", [])
            tx_text = "; ".join(f"{t.get('insider')}: {t.get('transaction')} ({t.get('shares')} shs)" for t in txs) if txs else "Form 4 filings logged."
            insider_text = f"[Form 4 Insiders & 13F Holdings] Institutional Ownership: {inst_pct}. Recent Insider Activity: {tx_text}."
            docs.append(Document(page_content=insider_text, metadata={
                "ticker": symbol,
                "source": "SEC EDGAR (Form 4/13F)",
                "document_type": "Form-4",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 1.00,
                "importance": 0.85,
                "time_horizon": "HISTORICAL"
            }))

        # 4. Company Profile & Business Architecture Description -> Horizon: HISTORICAL
        profile = gloomberb_payload.get("profile", {})
        if profile and profile.get("description") != "N/A":
            profile_text = f"[Company Profile & Business Description] Sector: {profile.get('sector')}. Description: {profile.get('description')}"
            docs.append(Document(page_content=profile_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Profile",
                "document_type": "CompanyProfile",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.85,
                "importance": 0.75,
                "time_horizon": "HISTORICAL"
            }))

        # 5. Wall Street Analyst Research, Ratings & Price Targets -> Horizon: CURRENT
        analyst = gloomberb_payload.get("analyst_ratings", {})
        if analyst and (analyst.get("recent_major_bank_actions") or analyst.get("mean_target_price") != "N/A"):
            actions_text = "; ".join(analyst.get("recent_major_bank_actions", [])[:6])
            analyst_text = (
                f"[Wall Street Analyst Research & Consensus] Mean Target Price: ${analyst.get('mean_target_price')}, "
                f"Median Target: ${analyst.get('median_target_price')}, High: ${analyst.get('high_target_price')}, Low: ${analyst.get('low_target_price')}. "
                f"Recent Major Bank Ratings: {actions_text if actions_text else 'Consensus ratings active.'}"
            )
            docs.append(Document(page_content=analyst_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Analyst Research",
                "document_type": "AnalystRatings",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.85,
                "importance": 0.85,
                "time_horizon": "CURRENT"
            }))

        # 6. Upcoming Earnings Guidance, Estimates & Revision Momentum -> Horizon: CURRENT
        earnings = gloomberb_payload.get("earnings", {})
        if earnings and earnings.get("earnings_date") != "N/A":
            rev7 = earnings.get("eps_revisions_7d", {})
            rev30 = earnings.get("eps_revisions_30d", {})
            earnings_text = (
                f"[Earnings Calendar & Revision Momentum] Upcoming Earnings: {earnings.get('earnings_date')} ({earnings.get('timing')}). "
                f"EPS Consensus: ${earnings.get('eps_estimate')} (YoY Growth: {earnings.get('eps_growth_yoy')}), "
                f"Revenue Estimate: {earnings.get('revenue_estimate')}. "
                f"Analyst EPS Revisions: Last 7 Days (Up: {rev7.get('up', 0)}, Down: {rev7.get('down', 0)}), "
                f"Last 30 Days (Up: {rev30.get('up', 0)}, Down: {rev30.get('down', 0)})."
            )
            docs.append(Document(page_content=earnings_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Earnings Calendar",
                "document_type": "EarningsGuidance",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.95,
                "importance": 0.90,
                "time_horizon": "CURRENT"
            }))

        # 7. Historical Earnings Surprises & Corporate Events -> Horizon: HISTORICAL
        events = gloomberb_payload.get("events", {})
        if events and events.get("historical_earnings_surprises"):
            surprises = events.get("historical_earnings_surprises", [])
            s_text = "; ".join(f"[{s.get('date')}: Actual ${s.get('actual')} vs Est ${s.get('estimate')} ({s.get('surprise_pct')})]" for s in surprises[:4])
            events_text = f"[Historical Earnings Surprises & Execution History] Past Quarters: {s_text}."
            docs.append(Document(page_content=events_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Corporate Events",
                "document_type": "EarningsSurprises",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 1.00,
                "importance": 0.85,
                "time_horizon": "HISTORICAL"
            }))

        # 8. Major Benchmark Market Indices & Market Breadth -> Horizon: CURRENT
        indices = gloomberb_payload.get("market_indices", [])
        movers = gloomberb_payload.get("market_movers", [])
        if indices or movers:
            idx_str = ", ".join(f"{i.get('name', i.get('symbol'))} {i.get('change_pct')}" for i in indices)
            mover_str = ", ".join(f"{m.get('symbol')} {m.get('change_pct')}" for m in movers[:4]) if movers else "Normal breadth"
            market_text = f"[Major Benchmark Indices & Market Breadth] Index Performance: {idx_str}. Top Active Movers: {mover_str}."
            docs.append(Document(page_content=market_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Benchmark Indices",
                "document_type": "MarketBreadth",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.95,
                "importance": 0.80,
                "time_horizon": "CURRENT"
            }))

        # 9. Institutional Multi-Source Qualitative Text Documents
        if institutional_data:
            # 5a. Company Investor-Relations (IR) Website Releases -> Horizon: CURRENT
            for ir in institutional_data.get("investor_relations", []):
                text = f"[Company Investor Relations ({ir.get('source')})] {ir.get('title')}. Summary: {ir.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": ir.get('source', 'Investor Relations'),
                    "document_type": ir.get('document_type', 'IRRelease'),
                    "published_at": ir.get('published_at', today_str),
                    "effective_date": ir.get('published_at', today_str),
                    "reliability": ir.get('reliability', 0.95),
                    "importance": ir.get('importance', 0.90),
                    "time_horizon": "CURRENT"
                }))

            # 5b. SEC EDGAR Direct Filings -> Horizon: CURRENT (8-K / 10-Q) or HISTORICAL (10-K)
            for sec in institutional_data.get("sec_edgar_direct", []):
                form_type = str(sec.get('form', '')).upper()
                horizon = "HISTORICAL" if "10-K" in form_type else "CURRENT"
                text = f"[SEC EDGAR Direct Filing Form {sec.get('form')} ({sec.get('date')})] {sec.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": "SEC EDGAR Direct",
                    "document_type": sec.get('document_type', f"Form-{sec.get('form')}"),
                    "published_at": sec.get('published_at', today_str),
                    "effective_date": sec.get('published_at', today_str),
                    "reliability": sec.get('reliability', 1.00),
                    "importance": sec.get('importance', 1.00),
                    "time_horizon": horizon
                }))

            # 5c. Quarterly Earnings Call Transcript Highlights -> Horizon: CURRENT
            for call in institutional_data.get("earnings_calls", []):
                text = f"[Quarterly Earnings Call Transcript Takeaways] {call.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": "Earnings Call Feed",
                    "document_type": call.get('document_type', 'EarningsTranscript'),
                    "published_at": call.get('published_at', today_str),
                    "effective_date": call.get('published_at', today_str),
                    "reliability": call.get('reliability', 0.90),
                    "importance": call.get('importance', 1.00),
                    "time_horizon": "CURRENT"
                }))

            # 5d. Official Corporate Press Releases -> Horizon: CURRENT
            for pr in institutional_data.get("official_press_releases", []):
                text = f"[Official Corporate Press Release ({pr.get('source')})] {pr.get('title')}. Summary: {pr.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": pr.get('source', 'Press Release'),
                    "document_type": pr.get('document_type', 'PressRelease'),
                    "published_at": pr.get('published_at', today_str),
                    "effective_date": pr.get('published_at', today_str),
                    "reliability": pr.get('reliability', 0.95),
                    "importance": pr.get('importance', 0.75),
                    "time_horizon": "CURRENT"
                }))

            # 5e. Reputable Financial Media News Coverage & Institutional Commentary -> Horizon: CURRENT
            for news_item in institutional_data.get("reputable_news", []):
                text = f"[Reputable Financial News ({news_item.get('source')})] {news_item.get('title')}. Details: {news_item.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": news_item.get('source', 'Reputable News'),
                    "document_type": news_item.get('document_type', 'ReputableNews'),
                    "published_at": news_item.get('published_at', today_str),
                    "effective_date": news_item.get('published_at', today_str),
                    "reliability": news_item.get('reliability', 0.70),
                    "importance": news_item.get('importance', 0.70),
                    "time_horizon": "CURRENT"
                }))

        return docs

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates 384-dimensional dense vector embeddings using FastEmbed."""
        embeddings_generator = self.embed_model.embed(texts)
        embeddings_list = [vec for vec in embeddings_generator]
        return np.array(embeddings_list, dtype=np.float32)

    def get_analytical_questions(self, symbol: str) -> Dict[str, Dict]:
        """
        Returns targeted sub-queries partitioned by temporal horizon:
        CURRENT CONTEXT (Last 24h / 7d / Latest Earnings) vs LONGER-TERM HISTORICAL CONTEXT.
        """
        return {
            # CURRENT CONTEXT (Last 24h / 7d / Latest Earnings)
            "Recent Material Events (Last 24h / 7d)": {
                "horizon": "CURRENT",
                "query": f"What recent material events, SEC 8-K/10-Q filings, press releases, or news catalysts changed for {symbol} in the last 7 days?"
            },
            "Short-Term Bullish & Bearish Drivers": {
                "horizon": "CURRENT",
                "query": f"What are the primary short-term bullish growth drivers and bearish downside risks for {symbol}?"
            },
            "Updated Guidance & Earnings Takeaways": {
                "horizon": "CURRENT",
                "query": f"What updated management guidance, executive outlook, or capex plans were issued in the latest earnings release for {symbol}?"
            },
            # LONGER-TERM HISTORICAL CONTEXT
            "Longer-Term Historical Execution Pattern": {
                "horizon": "HISTORICAL",
                "query": f"What is the longer-term historical business trend, product architecture, and multi-year execution pattern for {symbol}?"
            },
            "Persistent Structural & Valuation Headwinds": {
                "horizon": "HISTORICAL",
                "query": f"What historical valuation concerns, balance sheet liabilities, or structural headwinds have persisted over time for {symbol}?"
            },
            "Historical Macro & Industry Cycles": {
                "horizon": "HISTORICAL",
                "query": f"How do historical macro interest rate cycles, yield curve trends, and industry competitive dynamics impact {symbol}?"
            }
        }

    def retrieve_knowledge_by_questions(self, gloomberb_payload: Dict, technical_data: Dict = None, institutional_data: Dict = None, max_per_question: int = 2) -> Dict[str, List[str]]:
        """
        Executes Multi-Query Dual-Horizon Hybrid RAG Retrieval:
        1. Ingests all document chunks with time_horizon metadata ('CURRENT' vs 'HISTORICAL').
        2. Generates document vector embeddings.
        3. Queries vector space across 6 analytical questions partitioned by temporal horizon.
        4. Calculates Multiplicative Hybrid Score (Similarity x Reliability x Importance x Recency) per question.
        5. Returns top unique passages per question category.
        """
        symbol = gloomberb_payload.get("symbol", "N/A")
        print(f"[Multi-Query Hybrid RAG] Ingesting Gloomberb & Institutional Data for {symbol}...")

        raw_docs = self.convert_gloomberb_to_documents(gloomberb_payload, technical_data, institutional_data)
        doc_chunks = self.text_splitter.split_documents(raw_docs)
        chunk_texts = [d.page_content for d in doc_chunks]

        if not chunk_texts:
            return {"General": ["No document chunks available."]}

        print(f"[Multi-Query Hybrid RAG] Generating FastEmbed dense vectors for {len(chunk_texts)} document chunks...")
        doc_embeddings = self.generate_embeddings(chunk_texts)
        norm_docs = np.linalg.norm(doc_embeddings, axis=1)
        norm_docs[norm_docs == 0] = 1e-9

        questions_dict = self.get_analytical_questions(symbol)
        categorized_results = {}
        global_used_indices = set()

        print(f"[Multi-Query Hybrid RAG] Executing Dual-Horizon Multi-Query Search across 6 Analytical Categories:")

        for cat_name, q_info in questions_dict.items():
            target_horizon = q_info["horizon"]
            q_text = q_info["query"]

            q_emb = self.generate_embeddings([q_text])[0]
            norm_q = np.linalg.norm(q_emb)
            if norm_q == 0:
                norm_q = 1e-9

            cosine_sims = np.dot(doc_embeddings, q_emb) / (norm_docs * norm_q)

            scored_chunks = []
            for idx, doc in enumerate(doc_chunks):
                meta = doc.metadata or {}
                doc_horizon = meta.get("time_horizon", "CURRENT")

                # Filter by temporal horizon preference
                if doc_horizon != target_horizon:
                    horizon_penalty = 0.50  # Soft penalty if cross-horizon match
                else:
                    horizon_penalty = 1.00

                sim = float(cosine_sims[idx])
                rel = float(meta.get("reliability", 0.75))
                imp = float(meta.get("importance", 0.75))
                pub = meta.get("published_at", "N/A")
                rec = compute_recency_score(pub, doc_horizon)
                hybrid_score = compute_hybrid_score(sim, rel, imp, rec) * horizon_penalty

                scored_chunks.append({
                    "idx": idx,
                    "chunk": doc,
                    "hybrid_score": hybrid_score,
                    "sim": sim,
                    "metadata": meta
                })

            # Rank by Hybrid Score descending
            scored_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)

            category_passages = []
            count = 0
            for item in scored_chunks:
                if count >= max_per_question:
                    break
                idx = item["idx"]
                if idx in global_used_indices:
                    continue  # Skip duplicate chunks across questions

                global_used_indices.add(idx)
                count += 1

                m = item["metadata"]
                passage_text = item["chunk"].page_content
                formatted_passage = f"[Metadata: Ticker={m.get('ticker')} | Horizon={m.get('time_horizon')} | Source={m.get('source')} | Type={m.get('document_type')} | Published={m.get('published_at')} | Reliability={m.get('reliability')} | Importance={m.get('importance')}]\nContent: {passage_text}"
                category_passages.append(formatted_passage)

            print(f"  • [{target_horizon}] {cat_name}: Retrieved {len(category_passages)} top hybrid-ranked passages.")
            categorized_results[cat_name] = category_passages

        return categorized_results

    def retrieve_knowledge(self, gloomberb_payload: Dict, technical_data: Dict = None, institutional_data: Dict = None, query: str = None) -> List[str]:
        """Legacy single-query interface fallback, wrapper around retrieve_knowledge_by_questions."""
        cat_map = self.retrieve_knowledge_by_questions(gloomberb_payload, technical_data, institutional_data)
        flat_list = []
        for passages in cat_map.values():
            flat_list.extend(passages)
        return flat_list

    def get_nemotron_payload(self, gloomberb_payload: Dict, technical_data: Dict, institutional_data: Dict = None) -> Dict:
        """
        Directly generates Nemotron-3 Super prompt payload.
        Passes Multi-Query Dual-Horizon RAG Knowledge partitioned into CURRENT CONTEXT vs LONGER-TERM HISTORICAL CONTEXT.
        """
        symbol = gloomberb_payload.get("symbol", "N/A")
        categorized_rag = self.retrieve_knowledge_by_questions(gloomberb_payload, technical_data, institutional_data, max_per_question=2)

        profile = gloomberb_payload.get("profile", {})
        peer_val = gloomberb_payload.get("peer_valuation", {})
        fin_ratios = gloomberb_payload.get("financials", {}).get("key_ratios", {})
        macro_econ = gloomberb_payload.get("macro_econ", {})
        sector_bench = gloomberb_payload.get("sector_benchmark", {})
        options_flow = gloomberb_payload.get("options", {})

        # Compute deterministic 5-pillar quantitative scores (0-100)
        quant_service = QuantitativeScoringService()
        quant_scores = quant_service.compute_5pillar_scores(technical_data, gloomberb_payload, sector_bench)

        market_benchmark_summary = {
            "symbol": symbol,
            "sector": profile.get("sector", "N/A"),
            "deterministic_5pillar_scores": quant_scores,
            "current_price": technical_data.get("current_price"),
            "change_5d_pct": technical_data.get("change_5d_pct"),
            "market_spy_5d_pct": technical_data.get("market_spy_5d_pct"),
            "relative_alpha_5d": technical_data.get("relative_alpha_5d"),
            "sector_etf_benchmark": sector_bench,
            "return_1y": profile.get("return_1y", "N/A"),
            "return_3y": profile.get("return_3y", "N/A"),
            "valuation_multiples": {
                "forward_pe": fin_ratios.get("forward_pe", "N/A"),
                "trailing_pe": fin_ratios.get("trailing_pe", "N/A"),
                "price_to_sales": peer_val.get("price_to_sales", "N/A"),
                "ev_to_ebitda": peer_val.get("ev_to_ebitda", "N/A"),
                "price_to_free_cash_flow": peer_val.get("price_to_free_cash_flow", "N/A")
            },
            "direct_peer_benchmarks": peer_val.get("direct_peer_benchmarks", []),
            "macro_market_sentiment": macro_econ,
            "benchmark_market_indices": gloomberb_payload.get("market_indices", []),
            "market_spy_correlation": gloomberb_payload.get("market_spy_correlation", {}),
            "realtime_quote": gloomberb_payload.get("quote", {}),
            "wall_street_analyst_coverage": {
                "mean_target_price": gloomberb_payload.get("analyst_ratings", {}).get("mean_target_price"),
                "median_target_price": gloomberb_payload.get("analyst_ratings", {}).get("median_target_price"),
                "high_target_price": gloomberb_payload.get("analyst_ratings", {}).get("high_target_price"),
                "low_target_price": gloomberb_payload.get("analyst_ratings", {}).get("low_target_price"),
                "recommendation_rating": gloomberb_payload.get("analyst_ratings", {}).get("recommendation_rating"),
                "recommendations_breakdown": gloomberb_payload.get("analyst_ratings", {}).get("recommendations_breakdown", {}),
                "recent_major_bank_actions": gloomberb_payload.get("analyst_ratings", {}).get("recent_major_bank_actions", [])[:5]
            },
            "upcoming_earnings_and_revisions": gloomberb_payload.get("earnings", {}),
            "historical_earnings_surprises": gloomberb_payload.get("events", {}).get("historical_earnings_surprises", []),
            "top_institutional_holders": gloomberb_payload.get("insider_institutional", {}).get("top_institutional_holders", []),
            "granular_options_flow": {
                "put_call_ratio": options_flow.get("put_call_ratio"),
                "call_volume": options_flow.get("call_volume"),
                "put_volume": options_flow.get("put_volume"),
                "call_open_interest": options_flow.get("call_open_interest"),
                "put_open_interest": options_flow.get("put_open_interest"),
                "implied_volatility": options_flow.get("implied_volatility"),
                "unusual_activity": options_flow.get("unusual_activity")
            },
            "growth_and_margins": {
                "revenue_growth": fin_ratios.get("revenue_growth", "N/A"),
                "earnings_growth": fin_ratios.get("earnings_growth", "N/A"),
                "profit_margins": fin_ratios.get("profit_margins", "N/A")
            },
            "technical_indicators": {
                "rsi14": technical_data.get("rsi14"),
                "ema20": technical_data.get("ema20"),
                "ema50": technical_data.get("ema50"),
                "atr": technical_data.get("atr")
            }
        }

        user_prompt = f"""
================ 1. STRUCTURED MARKET & QUANTITATIVE DATA ================
Target Ticker: {symbol}

{json.dumps(market_benchmark_summary, indent=2)}

================ 2. CURRENT CONTEXT (Last 24h / 7d / Latest Earnings) ================
"""
        current_cats = ["Recent Material Events (Last 24h / 7d)", "Short-Term Bullish & Bearish Drivers", "Updated Guidance & Earnings Takeaways"]
        for cat_name in current_cats:
            passages = categorized_rag.get(cat_name, [])
            user_prompt += f"\n>>> SUB-QUESTION: {cat_name} <<<\n"
            if not passages:
                user_prompt += "No specific current passages matched this category.\n"
            for p_idx, passage in enumerate(passages, 1):
                user_prompt += f"  [{p_idx}] {passage}\n\n"

        user_prompt += """
================ 3. LONGER-TERM HISTORICAL CONTEXT (Prior Filings & Multi-Year Patterns) ================
"""
        historical_cats = ["Longer-Term Historical Execution Pattern", "Persistent Structural & Valuation Headwinds", "Historical Macro & Industry Cycles"]
        for cat_name in historical_cats:
            passages = categorized_rag.get(cat_name, [])
            user_prompt += f"\n>>> SUB-QUESTION: {cat_name} <<<\n"
            if not passages:
                user_prompt += "No specific historical passages matched this category.\n"
            for p_idx, passage in enumerate(passages, 1):
                user_prompt += f"  [{p_idx}] {passage}\n\n"

        user_prompt += """
================ END PAYLOAD ================

Synthesize the Structured Market Data, Deterministic 5-Pillar Scores, Current Context (Last 24h/7d), and Longer-Term Historical Context. Contrast recent short-term changes against multi-year historical execution. Anchor your decision around the Composite Quantitative Score. Return ONLY a valid JSON object matching the required schema.
"""

        return {
            "system_instruction": self.NEMOTRON_SYSTEM_INSTRUCTION.strip(),
            "user_prompt": user_prompt.strip(),
            "technical_summary": market_benchmark_summary
        }
