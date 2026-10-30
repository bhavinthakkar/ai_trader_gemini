from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

DB_PATH = "trader.db"


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """
    Initializes the SQLite database schema if tables do not exist and applies migrations.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence REAL,
                reason TEXT,
                institutional_data TEXT,
                macro_data TEXT,
                news TEXT,
                investment_bank_coverage TEXT,
                risk_assessment TEXT,
                pe_and_peg TEXT,
                model_used TEXT,
                raw_json TEXT
            );
        """)

        # Migration: Dynamic column addition for comprehensive model metrics
        new_columns = {
            "institutional_data": "TEXT",
            "macro_data": "TEXT",
            "buy_score": "REAL",
            "hold_score": "REAL",
            "sell_score": "REAL",
            "horizon_days": "INTEGER DEFAULT 10",
            "quant_score": "REAL",
            "trend_score": "REAL",
            "sector_score": "REAL",
            "alpha_score": "REAL",
            "val_history_score": "REAL",
            "peer_val_score": "REAL",
            "data_completeness": "REAL",
            "entry_price": "REAL",
            "stop_loss_price": "REAL",
            "target_price": "REAL",
            "rsi14": "REAL",
            "rvol_20d": "REAL",
            "us_10y_yield": "REAL",
            "yield_spread_10y2y": "REAL",
            "fear_greed_score": "REAL",
            "days_to_earnings": "INTEGER",
            "bull_case": "TEXT",
            "bear_case": "TEXT",
            "key_risks": "TEXT",
            "missing_information": "TEXT",
            "reward_risk_ratio": "REAL",
            "breakeven_win_rate": "REAL",
            "analyst_target_rr": "REAL",
            "structural_stop_price": "REAL",
            "structural_target_price": "REAL",
            "vol_factor": "REAL",
            "atr_pct": "REAL",
            "primary_driver": "TEXT",
            "falsification_bull": "TEXT",
            "falsification_bear": "TEXT",
            "model_confidence": "REAL",
            "model_quant_score": "REAL",
            "model_pillar_scores": "TEXT",
            "no_trade_reason": "TEXT",
            "decision_origin": "TEXT",
            "model_data_completeness": "REAL",
            "deterministic_data_completeness": "REAL",
            "implied_volatility": "REAL",
            "expected_move": "REAL",
            "expected_move_pct": "REAL",
            "stop_distance": "REAL",
            "stop_distance_pct": "REAL",
            "expected_move_stop_ratio": "REAL",
            "volatility_regime": "TEXT",
            "days_to_next_event": "INTEGER",
            "next_event_type": "TEXT",
            "next_event_date": "TEXT",
            "sell_type": "TEXT",
            "short_reward_risk_ratio": "REAL",
            "structural_short_stop_price": "REAL",
            "structural_short_target_price": "REAL",
            "short_stop_distance": "REAL",
            "short_stop_distance_pct": "REAL",
            "short_expected_move_stop_ratio": "REAL",
            "pillar_status": "TEXT",
            "pillar_input_tracking": "TEXT",
            "financial_quality_score": "REAL",
            "is_value_trap": "INTEGER",
            "value_trap_reasons": "TEXT",
            "company_name": "TEXT",
            "ml_score": "REAL",
            "ml_predicted_return": "REAL",
            "ml_rank": "INTEGER",
            "ml_universe_size": "INTEGER",
            "ml_signal_label": "TEXT",
            "ml_disposition": "TEXT",
            "run_id": "TEXT",
            "model_status": "TEXT",
            "passed_gates": "TEXT",
            "failed_gates": "TEXT",
            "data_quality_status": "TEXT",
            "position_action": "TEXT",
            "reason_codes": "TEXT",
            "structured_decision": "TEXT",
            "danelfin_score": "REAL",
            "danelfin_signal": "TEXT",
            "danelfin_details": "TEXT",
            "model_decision": "TEXT",
            "candidate_action": "TEXT",
            "candidate_type": "TEXT",
            "setup_evidence": "TEXT",
            "gate_results": "TEXT",
            "rejection_reasons": "TEXT",
            "final_decision": "TEXT",
            "decision_reasoning": "TEXT"
        }

        cursor.execute("PRAGMA table_info(signals);")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type};")

        # Create Portfolio Risk Review Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                portfolio_json TEXT,
                review_json TEXT,
                raw_json TEXT
            );
        """)

        # Create Ground-Truth Outcome Evaluation Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                horizon_days INTEGER,
                entry_price REAL,
                exit_price REAL,
                realized_return_pct REAL,
                max_runup_pct REAL,
                max_drawdown_pct REAL,
                hit_target BOOLEAN,
                hit_stop BOOLEAN,
                is_profitable BOOLEAN,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            );
        """)

        # Create Auditable RAG Evidence Record Table (Step 4.1)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_evidence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                document_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT,
                publication_timestamp TEXT NOT NULL,
                available_timestamp TEXT NOT NULL,
                retrieval_timestamp TEXT NOT NULL,
                passage_or_excerpt TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_horizon TEXT NOT NULL,
                reliability REAL,
                importance REAL,
                hybrid_score REAL,
                sub_query TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_ticker ON rag_evidence_records(ticker);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_doc ON rag_evidence_records(document_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_pub_avail ON rag_evidence_records(publication_timestamp, available_timestamp);")

        # Step 7.3: Order Execution Engine Schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_orders (
                order_id TEXT PRIMARY KEY,
                client_order_id TEXT UNIQUE NOT NULL,
                idempotency_key TEXT UNIQUE NOT NULL,
                signal_id INTEGER,
                run_id TEXT,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_shares INTEGER NOT NULL,
                filled_shares INTEGER DEFAULT 0,
                remaining_shares INTEGER NOT NULL,
                signal_price REAL,
                limit_price REAL,
                stop_loss_price REAL,
                target_price REAL,
                avg_fill_price REAL DEFAULT 0.0,
                total_commission REAL DEFAULT 0.0,
                broker_order_id TEXT,
                rejection_reason TEXT,
                cancellation_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                raw_json TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_ticker ON execution_orders(ticker);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_status ON execution_orders(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_idemp ON execution_orders(idempotency_key);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                shares INTEGER NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                execution_id TEXT,
                slippage_bps REAL DEFAULT 0.0,
                raw_json TEXT,
                FOREIGN KEY (order_id) REFERENCES execution_orders (order_id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_fills_order ON execution_fills(order_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_fills_ticker ON execution_fills(ticker);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_events (
                event_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (order_id) REFERENCES execution_orders (order_id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_events_order ON execution_events(order_id);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operational_kill_switch_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                is_tripped INTEGER NOT NULL,
                reason TEXT,
                details TEXT,
                actor TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Step 8: Shadow Trading Schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                quote_timestamp DATETIME,
                decision_latency_seconds REAL,
                quote_age_seconds REAL,
                is_fresh BOOLEAN,
                decision TEXT NOT NULL,
                position_action TEXT NOT NULL,
                composite_score REAL,
                signal_strength REAL,
                ml_score REAL,
                ml_predicted_return REAL,
                entry_price REAL,
                suggested_stop_loss REAL,
                suggested_target_price REAL,
                passed_gates TEXT,
                failed_gates TEXT,
                reason_codes TEXT,
                proposed_shares INTEGER DEFAULT 0,
                binding_constraint TEXT,
                is_order_proposed BOOLEAN DEFAULT 0,
                pre_flight_valid BOOLEAN,
                rejection_reasons TEXT,
                simulated_order_id TEXT,
                simulated_fill_price REAL,
                simulated_slippage_bps REAL,
                simulated_commission REAL,
                simulated_status TEXT,
                input_snapshot_json TEXT,
                component_outputs_json TEXT,
                realized_forward_return_pct REAL,
                realized_net_return_pct REAL,
                outcome_evaluated_at DATETIME,
                hit_target BOOLEAN,
                hit_stop BOOLEAN,
                is_profitable BOOLEAN,
                raw_json TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_ticker ON shadow_decisions(ticker);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_session ON shadow_decisions(session_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_time ON shadow_decisions(timestamp);")

        conn.commit()


def _ensure_str(val, default="") -> str:
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return "; ".join(str(x) for x in val)
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def _to_float(val, default=None):
    if val is None or str(val).upper() == "N/A":
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).replace("%", "").replace("+", "").replace("$", "").replace("x", "").strip()
        return float(s)
    except Exception:
        return default


def save_results(results: list, model_used: str = "Gemini 3.6 Flash", db_path=DB_PATH):
    """
    Saves a list of analysis result dicts into the signals table.
    Populates both new quantitative columns and backward-compatible summary fields.
    """
    if not results:
        return

    init_db(db_path)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for item in results:
            stock = _ensure_str(item.get("stock", item.get("symbol", "N/A")))
            decision = str(item.get("decision", "HOLD")).upper()
            confidence = _to_float(item.get("confidence"), 0.0)
            model_confidence = _to_float(item.get("model_confidence"))

            buy_score = _to_float(item.get("buy_score"))
            hold_score = _to_float(item.get("hold_score"))
            sell_score = _to_float(item.get("sell_score"))
            horizon_days = int(item.get("horizon_days", 10) or 10)

            quant_score = _to_float(item.get("quant_score"))
            pillars = item.get("pillar_scores") or {}
            trend_score = _to_float(pillars.get("trend"))
            sector_score = _to_float(pillars.get("sector"))
            alpha_score = _to_float(pillars.get("alpha"))
            val_history_score = _to_float(pillars.get("valuation_history"))
            peer_val_score = _to_float(pillars.get("peer_valuation"))
            data_completeness = _to_float(item.get("data_completeness"))

            entry_price = _to_float(item.get("entry_price") or item.get("current_price"))
            stop_loss_price = _to_float(item.get("stop_loss_price") or item.get("suggested_stop_loss"))
            target_price = _to_float(item.get("target_price") or item.get("suggested_target_price"))

            rsi14 = _to_float(item.get("rsi14"))
            rvol_20d = _to_float(item.get("rvol_20d"))
            us_10y_yield = _to_float(item.get("us_10y_yield"))
            yield_spread_10y2y = _to_float(item.get("yield_spread_10y2y") or item.get("yield_curve_spread_10y2y"))
            fear_greed_score = _to_float(item.get("fear_greed_score"))
            days_to_earnings = item.get("days_to_earnings")
            if days_to_earnings is not None:
                try:
                    days_to_earnings = int(days_to_earnings)
                except Exception:
                    days_to_earnings = None

            bull_case = _ensure_str(item.get("bull_case", []))
            bear_case = _ensure_str(item.get("bear_case", []))
            key_risks = _ensure_str(item.get("key_risks", []))
            missing_info = _ensure_str(item.get("missing_information", []))
            reward_risk_ratio = _to_float(item.get("reward_risk_ratio"))
            breakeven_win_rate = _to_float(item.get("breakeven_win_rate"))
            analyst_target_rr = _to_float(item.get("analyst_target_rr"))
            structural_stop_price = _to_float(item.get("structural_stop_price"))
            structural_target_price = _to_float(item.get("structural_target_price"))
            vol_factor = _to_float(item.get("vol_factor"), 1.0)
            atr_pct = _to_float(item.get("atr_pct"), 0.0)
            primary_driver = _ensure_str(item.get("primary_driver"), "")
            falsification_bull = _ensure_str(item.get("falsification_bull"), "")
            falsification_bear = _ensure_str(item.get("falsification_bear"), "")
            model_quant_score = _to_float(item.get("model_quant_score"))
            model_pillar_scores = json.dumps(item.get("model_pillar_scores") or {})
            no_trade_reason = item.get("no_trade_reason")
            if no_trade_reason:
                no_trade_reason = str(no_trade_reason).upper()
            decision_origin = item.get("decision_origin")
            if decision_origin:
                decision_origin = str(decision_origin).upper()
            model_data_completeness = _to_float(item.get("model_data_completeness"))
            deterministic_data_completeness = _to_float(item.get("deterministic_data_completeness"))

            implied_volatility = _to_float(item.get("implied_volatility"))
            expected_move = _to_float(item.get("expected_move"))
            expected_move_pct = _to_float(item.get("expected_move_pct"))
            stop_distance = _to_float(item.get("stop_distance"))
            stop_distance_pct = _to_float(item.get("stop_distance_pct"))
            expected_move_stop_ratio = _to_float(item.get("expected_move_stop_ratio"))
            volatility_regime = _ensure_str(item.get("volatility_regime"), "")
            days_to_next_event = item.get("days_to_next_event")
            if days_to_next_event is not None:
                try:
                    days_to_next_event = int(days_to_next_event)
                except Exception:
                    days_to_next_event = None
            next_event_type = _ensure_str(item.get("next_event_type"), "")
            next_event_date = _ensure_str(item.get("next_event_date"), "")
            sell_type = _ensure_str(item.get("sell_type"), "")
            short_reward_risk_ratio = _to_float(item.get("short_reward_risk_ratio"))
            structural_short_stop_price = _to_float(item.get("structural_short_stop_price"))
            structural_short_target_price = _to_float(item.get("structural_short_target_price"))
            short_stop_distance = _to_float(item.get("short_stop_distance"))
            short_stop_distance_pct = _to_float(item.get("short_stop_distance_pct"))
            short_expected_move_stop_ratio = _to_float(item.get("short_expected_move_stop_ratio"))
            pillar_status = json.dumps(item.get("pillar_status") or {})
            pillar_input_tracking = json.dumps(item.get("pillar_input_tracking") or {})
            financial_quality_score = _to_float(item.get("financial_quality_score"))
            is_value_trap = 1 if item.get("is_value_trap") else 0
            value_trap_reasons = _ensure_str(item.get("value_trap_reasons", []))

            # Standalone ML pipeline fields
            ml_ctx = item.get("ml_signal") or {}
            ml_score = _to_float(item.get("ml_score_0_100") or ml_ctx.get("score_0_100"))
            ml_predicted_return = _to_float(ml_ctx.get("predicted_20d_excess_return"))
            ml_rank = int(ml_ctx.get("rank")) if ml_ctx.get("rank") is not None else None
            ml_universe_size = int(ml_ctx.get("universe_size")) if ml_ctx.get("universe_size") is not None else None
            ml_signal_label = _ensure_str(ml_ctx.get("signal") or (item.get("ml_signal_label") if item.get("ml_signal_label") else ""), "")
            if not ml_signal_label and ml_score is not None:
                try:
                    from ml_signal_service import classify_ml_signal
                    ml_signal_label = classify_ml_signal(ml_score)["signal"]
                except Exception:
                    pass
            ml_disposition = _ensure_str(item.get("ml_signal_assessment", {}).get("disposition") if isinstance(item.get("ml_signal_assessment"), dict) else "", "")

            # Structured decision fields (Step 6.4)
            run_id = _ensure_str(item.get("run_id"), "")
            model_status = _ensure_str(item.get("model_status"), "")
            passed_gates = json.dumps(item.get("passed_gates") or []) if isinstance(item.get("passed_gates"), (list, tuple)) else _ensure_str(item.get("passed_gates"), "[]")
            failed_gates = json.dumps(item.get("failed_gates") or []) if isinstance(item.get("failed_gates"), (list, tuple)) else _ensure_str(item.get("failed_gates"), "[]")
            data_quality_status = _ensure_str(item.get("data_quality_status"), "VALID")
            position_action = _ensure_str(item.get("position_action"), "NO_ACTION")
            reason_codes = json.dumps(item.get("reason_codes") or []) if isinstance(item.get("reason_codes"), (list, tuple)) else _ensure_str(item.get("reason_codes"), "[]")
            struct_dec = item.get("structured_decision")
            if isinstance(struct_dec, dict):
                structured_decision = json.dumps(struct_dec)
            else:
                structured_decision = _ensure_str(struct_dec, "")

            # Danelfin AI Intelligence metrics
            danelfin_data = item.get("danelfin") or {}
            danelfin_score = _to_float(item.get("danelfin_score") or danelfin_data.get("aiscore"), None)
            danelfin_signal = _ensure_str(item.get("danelfin_signal") or danelfin_data.get("signal", ""), "")
            danelfin_details = json.dumps(danelfin_data) if danelfin_data else ""

            # Explicit Decision Contract (Step 2)
            model_decision = _ensure_str(item.get("model_decision") or decision, "HOLD")
            candidate_action = _ensure_str(item.get("candidate_action"), "")
            candidate_type = _ensure_str(item.get("candidate_type"), "")
            setup_evidence = _ensure_str(item.get("setup_evidence") or {})
            gate_results = _ensure_str(item.get("gate_results") or {})
            rejection_reasons = _ensure_str(item.get("rejection_reasons") or [])
            final_decision = _ensure_str(item.get("final_decision") or decision, "HOLD")

            # Backward-compatible text summaries & decision reasoning
            reason = _ensure_str(item.get("decision_reasoning") or item.get("reason") or bull_case)
            decision_reasoning = _ensure_str(item.get("decision_reasoning") or reason)
            inst_data = _ensure_str(item.get("institutional_data", ""))
            macro_data = _ensure_str(item.get("macro_data", item.get("marco_data", "")))
            news = _ensure_str(item.get("news", ""))
            bank_coverage = _ensure_str(item.get("investment_bank_coverage", ""))
            risk_info = _ensure_str(item.get("risk_assessment") or key_risks)
            pe_peg = _ensure_str(item.get("PE_and_PEG") or item.get("forward_pe", ""))
            raw_json = json.dumps(item)

            cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, decision, confidence, reason, institutional_data, macro_data, news,
                    investment_bank_coverage, risk_assessment, pe_and_peg, model_used, raw_json,
                    buy_score, hold_score, sell_score, horizon_days, quant_score,
                    trend_score, sector_score, alpha_score, val_history_score, peer_val_score,
                    data_completeness, entry_price, stop_loss_price, target_price,
                    rsi14, rvol_20d, us_10y_yield, yield_spread_10y2y, fear_greed_score,
                    days_to_earnings, bull_case, bear_case, key_risks, missing_information,
                    reward_risk_ratio, breakeven_win_rate, analyst_target_rr,
                    structural_stop_price, structural_target_price,
                    vol_factor, atr_pct, primary_driver, falsification_bull, falsification_bear,
                    model_confidence, model_quant_score, model_pillar_scores,
                    no_trade_reason, decision_origin, model_data_completeness, deterministic_data_completeness,
                    implied_volatility, expected_move, expected_move_pct, stop_distance, stop_distance_pct,
                    expected_move_stop_ratio, volatility_regime, days_to_next_event, next_event_type, next_event_date,
                    sell_type, short_reward_risk_ratio, structural_short_stop_price, structural_short_target_price,
                    short_stop_distance, short_stop_distance_pct, short_expected_move_stop_ratio,
                    pillar_status, pillar_input_tracking, financial_quality_score, is_value_trap, value_trap_reasons,
                    ml_score, ml_predicted_return, ml_rank, ml_universe_size, ml_signal_label, ml_disposition,
                    run_id, model_status, passed_gates, failed_gates, data_quality_status, position_action, reason_codes, structured_decision,
                    danelfin_score, danelfin_signal, danelfin_details,
                    model_decision, candidate_action, candidate_type, setup_evidence, gate_results, rejection_reasons, final_decision,
                    decision_reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, stock, decision, confidence, reason, inst_data, macro_data, news,
                bank_coverage, risk_info, pe_peg, model_used, raw_json,
                buy_score, hold_score, sell_score, horizon_days, quant_score,
                trend_score, sector_score, alpha_score, val_history_score, peer_val_score,
                data_completeness, entry_price, stop_loss_price, target_price,
                rsi14, rvol_20d, us_10y_yield, yield_spread_10y2y, fear_greed_score,
                days_to_earnings, bull_case, bear_case, key_risks, missing_info,
                reward_risk_ratio, breakeven_win_rate, analyst_target_rr,
                structural_stop_price, structural_target_price,
                vol_factor, atr_pct, primary_driver, falsification_bull, falsification_bear,
                model_confidence, model_quant_score, model_pillar_scores,
                no_trade_reason, decision_origin, model_data_completeness, deterministic_data_completeness,
                implied_volatility, expected_move, expected_move_pct, stop_distance, stop_distance_pct,
                expected_move_stop_ratio, volatility_regime, days_to_next_event, next_event_type, next_event_date,
                sell_type, short_reward_risk_ratio, structural_short_stop_price, structural_short_target_price,
                short_stop_distance, short_stop_distance_pct, short_expected_move_stop_ratio,
                pillar_status, pillar_input_tracking, financial_quality_score, is_value_trap, value_trap_reasons,
                ml_score, ml_predicted_return, ml_rank, ml_universe_size, ml_signal_label, ml_disposition,
                run_id, model_status, passed_gates, failed_gates, data_quality_status, position_action, reason_codes, structured_decision,
                danelfin_score, danelfin_signal, danelfin_details,
                model_decision, candidate_action, candidate_type, setup_evidence, gate_results, rejection_reasons, final_decision,
                decision_reasoning
            ))

            signal_id = cursor.lastrowid
            evidence_records = item.get("evidence_records") or []
            if evidence_records and signal_id:
                _save_evidence_records_with_cursor(cursor, evidence_records, signal_id=signal_id)
        conn.commit()
    print(f"[DB] Successfully saved {len(results)} analysis records to database.")


def _save_evidence_records_with_cursor(cursor, records: list, signal_id: int | None = None):
    """Internal helper to insert evidence records using an active cursor."""
    from evidence_record import EvidenceRecord

    for r in records:
        if isinstance(r, EvidenceRecord):
            rec = r
        elif isinstance(r, dict):
            rec = EvidenceRecord.from_dict(r)
        else:
            continue

        cursor.execute("""
            INSERT INTO rag_evidence_records (
                signal_id, document_id, ticker, source_type, source_url,
                publication_timestamp, available_timestamp, retrieval_timestamp,
                passage_or_excerpt, event_type, event_horizon,
                reliability, importance, hybrid_score, sub_query
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_id,
            rec.document_id,
            rec.ticker,
            rec.source_type,
            rec.source_url,
            rec.publication_timestamp,
            rec.available_timestamp,
            rec.retrieval_timestamp,
            rec.passage_or_excerpt,
            rec.event_type,
            rec.event_horizon,
            rec.reliability,
            rec.importance,
            rec.hybrid_score,
            rec.sub_query,
        ))


def save_evidence_records(records: list, signal_id: int | None = None, db_path=DB_PATH):
    """Save a list of EvidenceRecords or dictionaries to SQLite."""
    if not records:
        return
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        _save_evidence_records_with_cursor(cursor, records, signal_id=signal_id)
        conn.commit()


def get_evidence_records_for_signal(signal_id: int, db_path=DB_PATH) -> list[dict]:
    """Retrieve all auditable evidence records associated with a signal ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM rag_evidence_records
            WHERE signal_id = ?
            ORDER BY hybrid_score DESC, id ASC
        """, (signal_id,))
        return [dict(row) for row in cursor.fetchall()]


def query_evidence_as_of(
    ticker: str,
    as_of_timestamp: str,
    limit: int = 50,
    db_path=DB_PATH,
) -> list[dict]:
    """
    Retrieve auditable evidence records for a ticker with strict point-in-time filtering.

    Enforces zero-lookahead:
        publication_timestamp <= as_of_timestamp AND available_timestamp <= as_of_timestamp
    """
    init_db(db_path)
    from evidence_record import format_iso_timestamp
    as_of_iso = format_iso_timestamp(as_of_timestamp)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM rag_evidence_records
            WHERE ticker = ?
              AND publication_timestamp <= ?
              AND available_timestamp <= ?
            ORDER BY publication_timestamp DESC, hybrid_score DESC
            LIMIT ?
        """, (str(ticker).strip().upper(), as_of_iso, as_of_iso, limit))
        return [dict(row) for row in cursor.fetchall()]


def get_latest_signals(db_path=DB_PATH, max_age_days: int | None = 7, order_by: str = "date_desc") -> list:
    """
    Returns the most recent analysis record for each stock symbol.
    By default, restricts to signals generated within the last `max_age_days` (default: 7 days / 1 week).
    Pass max_age_days=None or max_age_days=0 to disable the recency filter and return all-time latest signals.
    `order_by`: 'date_desc' (newest analysis first) or 'symbol_asc' (alphabetical).
    """
    init_db(db_path)
    order_clause = "ORDER BY s.timestamp DESC, s.id DESC" if order_by == "date_desc" else "ORDER BY s.symbol ASC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if max_age_days and max_age_days > 0:
            cursor.execute(f"""
                SELECT s.*
                FROM signals s
                INNER JOIN (
                    SELECT symbol, MAX(id) as max_id
                    FROM signals
                    WHERE timestamp >= datetime('now', ?)
                    GROUP BY symbol
                ) latest ON s.id = latest.max_id
                WHERE s.timestamp >= datetime('now', ?)
                {order_clause};
            """, (f"-{max_age_days} days", f"-{max_age_days} days"))
        else:
            cursor.execute(f"""
                SELECT s.*
                FROM signals s
                INNER JOIN (
                    SELECT symbol, MAX(id) as max_id
                    FROM signals
                    GROUP BY symbol
                ) latest ON s.id = latest.max_id
                {order_clause};
            """)
        rows = cursor.fetchall()
        return [_enrich_signal_dict(dict(r)) for r in rows]


def _enrich_signal_dict(s: dict) -> dict:
    """Enriches a signal dict with company name and point-in-time ML indicator fields."""
    if not s:
        return s
    sym = s.get("symbol", "")
    if not s.get("company_name"):
        try:
            from ticker_resolver import resolve_symbol
            info = resolve_symbol(sym)
            s["company_name"] = info.get("company_name") or sym
        except Exception:
            pass
    # Enrich ML indicator if missing
    ml_score = s.get("ml_score")
    if ml_score is None:
        raw_j = s.get("raw_json")
        if raw_j:
            try:
                parsed_payload = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
                ml_sig = parsed_payload.get("ml_signal")
                if isinstance(ml_sig, dict):
                    s["ml_score"] = _to_float(ml_sig.get("score_0_100"))
                    s["ml_predicted_return"] = _to_float(ml_sig.get("predicted_20d_excess_return"))
                    s["ml_rank"] = ml_sig.get("rank")
                    s["ml_universe_size"] = ml_sig.get("universe_size")
                    s["ml_signal_label"] = ml_sig.get("signal")
                if parsed_payload.get("ml_signal_assessment"):
                    s["ml_disposition"] = parsed_payload.get("ml_signal_assessment", {}).get("disposition")
            except Exception:
                pass
        if s.get("ml_score") is None:
            try:
                from ml_signal_service import get_ml_context
                ctx = get_ml_context(sym)
                if ctx:
                    s["ml_score"] = ctx.get("score_0_100")
                    s["ml_predicted_return"] = ctx.get("predicted_20d_excess_return")
                    s["ml_rank"] = ctx.get("rank")
                    s["ml_universe_size"] = ctx.get("universe_size")
                    s["ml_signal_label"] = ctx.get("signal")
            except Exception:
                pass
    if s.get("ml_score") is not None and not s.get("ml_signal_label"):
        try:
            from ml_signal_service import classify_ml_signal
            cls = classify_ml_signal(s["ml_score"])
            s["ml_signal_label"] = cls["signal"]
            s["ml_badge"] = cls["badge"]
        except Exception:
            pass

    # Deserialize structured decision columns (Step 6.4)
    for json_col in ("passed_gates", "failed_gates", "reason_codes", "structured_decision"):
        val = s.get(json_col)
        if val and isinstance(val, str):
            try:
                s[json_col] = json.loads(val)
            except Exception:
                pass

    if "position_sizing" not in s or not s["position_sizing"]:
        raw_j = s.get("raw_json")
        if raw_j:
            try:
                parsed_payload = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
                if parsed_payload.get("position_sizing"):
                    s["position_sizing"] = parsed_payload.get("position_sizing")
            except Exception:
                pass

    # Enrich decision_reasoning from explicit field, raw payload, or reason
    dec_reasoning = s.get("decision_reasoning")
    raw_j = s.get("raw_json")
    parsed_payload = None
    if raw_j:
        try:
            parsed_payload = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
        except Exception:
            pass

    if not dec_reasoning and parsed_payload and isinstance(parsed_payload, dict):
        dec_reasoning = (
            parsed_payload.get("decision_reasoning")
            or parsed_payload.get("decision_rationale")
            or (parsed_payload.get("forecast_20d", {}).get("rationale") if isinstance(parsed_payload.get("forecast_20d"), dict) else None)
        )
    if not dec_reasoning:
        dec_reasoning = s.get("reason")
    s["decision_reasoning"] = dec_reasoning or ""
    if dec_reasoning and (not s.get("reason") or s.get("reason") == s.get("bull_case")):
        s["reason"] = dec_reasoning

    return s




def get_signal_history(symbol: str = None, limit: int = 100, db_path=DB_PATH) -> list:
    """
    Returns historical analysis records sorted by most recent first.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if symbol and symbol.strip():
            cursor.execute("""
                SELECT * FROM signals
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?;
            """, (symbol.strip().upper(), limit))
        else:
            cursor.execute("""
                SELECT * FROM signals
                ORDER BY id DESC
                LIMIT ?;
            """, (limit,))
        rows = cursor.fetchall()
        return [_enrich_signal_dict(dict(r)) for r in rows]


def get_summary_stats(db_path=DB_PATH, max_age_days: int | None = 7) -> dict:
    """
    Returns summary metrics (counts, last run date, latest decision breakdown).
    By default, metrics reflect active signals from the last `max_age_days` (default: 7 days / 1 week).
    """
    init_db(db_path)
    latest = get_latest_signals(db_path=db_path, max_age_days=max_age_days)
    buy_count = sum(1 for s in latest if s["decision"] == "BUY")
    sell_count = sum(1 for s in latest if s["decision"] == "SELL")
    hold_count = sum(1 for s in latest if s["decision"] == "HOLD")

    from quant.compliance_gates import is_signal_vetoed
    model_buy_count = sum(
        1 for s in latest
        if str(s.get("model_decision") or s.get("decision") or "").upper() in ("BUY", "DEEP BUY")
    )
    model_sell_count = sum(
        1 for s in latest
        if str(s.get("model_decision") or s.get("decision") or "").upper() in ("SELL", "BOOK PROFIT")
    )
    model_hold_count = sum(
        1 for s in latest
        if str(s.get("model_decision") or s.get("decision") or "").upper() == "HOLD"
    )
    vetoed_count = sum(1 for s in latest if is_signal_vetoed(s))

    hold_origin_counts = {}
    for signal in latest:
        if signal["decision"] != "HOLD":
            continue
        origin = signal.get("decision_origin") or "NOT_RECORDED"
        hold_origin_counts[origin] = hold_origin_counts.get(origin, 0) + 1

    last_run = None
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) as last_run FROM signals;")
        row = cursor.fetchone()
        if row and row["last_run"]:
            last_run = row["last_run"]

    return {
        "total_tracked": len(latest),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "model_buy_count": model_buy_count,
        "model_sell_count": model_sell_count,
        "model_hold_count": model_hold_count,
        "vetoed_count": vetoed_count,
        "hold_origin_counts": hold_origin_counts,
        "last_run": last_run or "N/A"
    }


def delete_signals_older_than(days: int = 7, db_path=DB_PATH) -> int:
    """
    Permanently deletes signals older than `days` from the database.
    Returns the count of deleted records.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM signals WHERE timestamp < datetime('now', ?);", (f"-{days} days",))
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count



def update_signal_outcomes(db_path=DB_PATH) -> int:
    """
    Evaluates past trading signals against forward market price action using yfinance.
    Calculates realized return %, max runup %, max drawdown %, and profit status.
    Returns the number of signals evaluated.
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    init_db(db_path)
    evaluated_count = 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # Find signals with entry_price that don't have an outcome record yet
        cursor.execute("""
            SELECT s.id, s.timestamp, s.symbol, s.decision, s.entry_price,
                   s.stop_loss_price, s.target_price, s.horizon_days
            FROM signals s
            LEFT JOIN signal_outcomes o ON s.id = o.signal_id
            WHERE o.id IS NULL AND s.entry_price IS NOT NULL AND s.entry_price > 0
            ORDER BY s.id ASC;
        """)
        pending = cursor.fetchall()

        if not pending:
            return 0

        now = datetime.utcnow()

        for row in pending:
            sig_id = row["id"]
            sig_time_str = row["timestamp"]
            symbol = row["symbol"]
            decision = str(row["decision"]).upper()
            entry_price = float(row["entry_price"])
            stop_loss = row["stop_loss_price"]
            target_price = row["target_price"]
            horizon_days = int(row["horizon_days"] or 10)

            try:
                sig_dt = datetime.strptime(sig_time_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    sig_dt = datetime.strptime(sig_time_str[:10], "%Y-%m-%d")
                except Exception:
                    continue

            # Need at least 1 day elapsed to evaluate forward price action
            if (now - sig_dt).total_seconds() < 86400:
                continue

            try:
                start_date = sig_dt.strftime("%Y-%m-%d")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, interval="1d")

                if hist.empty or len(hist) < 2:
                    continue

                # Bars strictly after signal date
                eval_bars = hist.iloc[1:horizon_days + 1]
                if eval_bars.empty:
                    continue

                exit_price = round(float(eval_bars["Close"].iloc[-1]), 2)
                realized_return_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)

                max_high = float(eval_bars["High"].max())
                min_low = float(eval_bars["Low"].min())

                max_runup_pct = round(((max_high - entry_price) / entry_price) * 100, 2)
                max_drawdown_pct = round(((min_low - entry_price) / entry_price) * 100, 2)

                hit_target = bool(target_price and max_high >= float(target_price))
                hit_stop = bool(stop_loss and min_low <= float(stop_loss))

                if decision == "BUY":
                    is_profitable = realized_return_pct > 0
                elif decision == "SELL":
                    is_profitable = realized_return_pct < 0
                else:
                    is_profitable = abs(realized_return_pct) <= 3.0  # HOLD is accurate if price remained stable

                cursor.execute("""
                    INSERT INTO signal_outcomes (
                        signal_id, horizon_days, entry_price, exit_price,
                        realized_return_pct, max_runup_pct, max_drawdown_pct,
                        hit_target, hit_stop, is_profitable
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sig_id, len(eval_bars), entry_price, exit_price,
                    realized_return_pct, max_runup_pct, max_drawdown_pct,
                    hit_target, hit_stop, is_profitable
                ))
                evaluated_count += 1

            except Exception as e:
                print(f"[DB Outcome Evaluator] Error evaluating signal {sig_id} for {symbol}: {e}")

        conn.commit()

    if evaluated_count > 0:
        print(f"[DB] Successfully evaluated {evaluated_count} trade outcomes.")
    return evaluated_count


def save_portfolio_review(review: dict, portfolio: dict, model_used: str = "gemini", db_path=DB_PATH) -> int:
    """
    Saves a portfolio risk review result into the portfolio_reviews table.
    Returns the new record id.
    """
    init_db(db_path)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio_reviews (timestamp, model_used, portfolio_json, review_json, raw_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            timestamp,
            str(model_used),
            json.dumps(portfolio or {}),
            json.dumps(review or {}),
            json.dumps({"review": review, "portfolio": portfolio})
        ))
        conn.commit()
        return cursor.lastrowid


def get_latest_portfolio_reviews(limit: int = 20, db_path=DB_PATH) -> list:
    """
    Returns the most recent portfolio risk reviews sorted newest first.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM portfolio_reviews
            ORDER BY id DESC
            LIMIT ?;
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_outcome_performance_stats(db_path=DB_PATH) -> dict:
    """
    Computes win rate, average return, and outcome statistics across all evaluated signals.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT o.*, s.symbol, s.decision, s.confidence, s.quant_score
            FROM signal_outcomes o
            JOIN signals s ON o.signal_id = s.id;
        """)
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return {
                "total_evaluated": 0,
                "win_rate_pct": 0.0,
                "avg_return_pct": 0.0,
                "profitable_trades": 0,
                "losing_trades": 0
            }

        total = len(rows)
        profitable = sum(1 for r in rows if r["is_profitable"])
        win_rate = round((profitable / total) * 100, 1)
        returns = [r["realized_return_pct"] for r in rows if r["realized_return_pct"] is not None]
        avg_ret = round(sum(returns) / len(returns), 2) if returns else 0.0

        return {
            "total_evaluated": total,
            "win_rate_pct": win_rate,
            "avg_return_pct": avg_ret,
            "profitable_trades": profitable,
            "losing_trades": total - profitable
        }


# =============================================================================
# Execution Layer Persistence Helpers (Step 7.3)
# =============================================================================

def save_execution_order(order_dict: dict, db_path=DB_PATH) -> None:
    """
    Inserts or updates an order in the execution_orders table.
    """
    init_db(db_path)
    now_str = _now_str()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO execution_orders (
                order_id, client_order_id, idempotency_key, signal_id, run_id,
                ticker, side, order_type, status, requested_shares, filled_shares,
                remaining_shares, signal_price, limit_price, stop_loss_price,
                target_price, avg_fill_price, total_commission, broker_order_id,
                rejection_reason, cancellation_reason, created_at, updated_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status = excluded.status,
                filled_shares = excluded.filled_shares,
                remaining_shares = excluded.remaining_shares,
                avg_fill_price = excluded.avg_fill_price,
                total_commission = excluded.total_commission,
                broker_order_id = excluded.broker_order_id,
                rejection_reason = excluded.rejection_reason,
                cancellation_reason = excluded.cancellation_reason,
                updated_at = excluded.updated_at,
                raw_json = excluded.raw_json;
        """, (
            str(order_dict.get("order_id")),
            str(order_dict.get("client_order_id")),
            str(order_dict.get("idempotency_key")),
            order_dict.get("signal_id"),
            order_dict.get("run_id"),
            str(order_dict.get("ticker")),
            str(order_dict.get("side")),
            str(order_dict.get("order_type", "LIMIT")),
            str(order_dict.get("status")),
            int(order_dict.get("requested_shares", 0)),
            int(order_dict.get("filled_shares", 0)),
            int(order_dict.get("remaining_shares", 0)),
            _to_float(order_dict.get("signal_price")),
            _to_float(order_dict.get("limit_price")),
            _to_float(order_dict.get("stop_loss_price")),
            _to_float(order_dict.get("target_price")),
            _to_float(order_dict.get("avg_fill_price"), 0.0),
            _to_float(order_dict.get("total_commission"), 0.0),
            order_dict.get("broker_order_id"),
            order_dict.get("rejection_reason"),
            order_dict.get("cancellation_reason"),
            str(order_dict.get("created_at") or now_str),
            str(order_dict.get("updated_at") or now_str),
            json.dumps(order_dict) if not isinstance(order_dict.get("raw_json"), str) else order_dict.get("raw_json"),
        ))
        conn.commit()


def save_execution_fill(fill_dict: dict, db_path=DB_PATH) -> None:
    """
    Records a trade execution fill into the execution_fills table.
    """
    init_db(db_path)
    now_str = _now_str()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO execution_fills (
                fill_id, order_id, ticker, side, shares, price,
                commission, timestamp, execution_id, slippage_bps, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(fill_dict.get("fill_id")),
            str(fill_dict.get("order_id")),
            str(fill_dict.get("ticker")),
            str(fill_dict.get("side")),
            int(fill_dict.get("shares", 0)),
            float(fill_dict.get("price", 0.0)),
            float(fill_dict.get("commission", 0.0)),
            str(fill_dict.get("timestamp") or now_str),
            fill_dict.get("execution_id"),
            float(fill_dict.get("slippage_bps", 0.0)),
            json.dumps(fill_dict),
        ))
        conn.commit()


def save_execution_event(event_dict: dict, db_path=DB_PATH) -> None:
    """
    Records an execution state transition event into the execution_events table.
    """
    init_db(db_path)
    now_str = _now_str()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO execution_events (
                event_id, order_id, event_type, from_status, to_status, details, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(event_dict.get("event_id")),
            str(event_dict.get("order_id")),
            str(event_dict.get("event_type")),
            event_dict.get("from_status"),
            str(event_dict.get("to_status")),
            str(event_dict.get("details", "")),
            str(event_dict.get("timestamp") or now_str),
        ))
        conn.commit()


def save_kill_switch_state(
    is_tripped: bool,
    reason: str = "",
    details: str = "",
    actor: str = "SYSTEM",
    db_path=DB_PATH
) -> int:
    """
    Persists an operational kill switch state change event.
    """
    init_db(db_path)
    now_str = _now_str()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO operational_kill_switch_state (
                is_tripped, reason, details, actor, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            1 if is_tripped else 0,
            str(reason),
            str(details),
            str(actor),
            now_str,
        ))
        conn.commit()
        return cursor.lastrowid


def get_latest_kill_switch_state(db_path=DB_PATH) -> Optional[dict]:
    """
    Returns the most recent operational kill switch state record, or None if never recorded.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM operational_kill_switch_state
            ORDER BY id DESC
            LIMIT 1;
        """)
        row = cursor.fetchone()
        return dict(row) if row else None


def get_execution_orders(
    ticker: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db_path=DB_PATH
) -> list:
    """
    Fetches execution orders matching optional filter criteria, sorted newest first.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM execution_orders WHERE 1=1"
        params = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if status:
            query += " AND status = ?"
            params.append(status.upper())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def get_execution_fills(
    order_id: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 100,
    db_path=DB_PATH
) -> list:
    """
    Fetches execution fills matching optional filter criteria, sorted newest first.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM execution_fills WHERE 1=1"
        params = []
        if order_id:
            query += " AND order_id = ?"
            params.append(str(order_id))
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def get_execution_events(
    order_id: Optional[str] = None,
    limit: int = 100,
    db_path=DB_PATH
) -> list:
    """
    Fetches execution status events, sorted newest first.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM execution_events WHERE 1=1"
        params = []
        if order_id:
            query += " AND order_id = ?"
            params.append(str(order_id))
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


# =============================================================================
# Shadow Trading Persistence Helpers (Step 8)
# =============================================================================

def save_shadow_decision(record_dict: dict, db_path=DB_PATH) -> int:
    """
    Inserts a shadow trading decision record into shadow_decisions table.
    Returns the new record id.
    """
    init_db(db_path)
    now_str = _now_str()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO shadow_decisions (
                session_id, run_id, ticker, timestamp, quote_timestamp,
                decision_latency_seconds, quote_age_seconds, is_fresh,
                decision, position_action, composite_score, signal_strength,
                ml_score, ml_predicted_return, entry_price, suggested_stop_loss,
                suggested_target_price, passed_gates, failed_gates, reason_codes,
                proposed_shares, binding_constraint, is_order_proposed,
                pre_flight_valid, rejection_reasons, simulated_order_id,
                simulated_fill_price, simulated_slippage_bps, simulated_commission,
                simulated_status, input_snapshot_json, component_outputs_json,
                realized_forward_return_pct, realized_net_return_pct,
                outcome_evaluated_at, hit_target, hit_stop, is_profitable, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(record_dict.get("session_id", "DEFAULT_SHADOW_SESSION")),
            str(record_dict.get("run_id", "")),
            str(record_dict.get("ticker", "")).upper(),
            str(record_dict.get("timestamp") or now_str),
            str(record_dict.get("quote_timestamp") or ""),
            _to_float(record_dict.get("decision_latency_seconds")),
            _to_float(record_dict.get("quote_age_seconds")),
            1 if record_dict.get("is_fresh", True) else 0,
            str(record_dict.get("decision", "HOLD")),
            str(record_dict.get("position_action", "NO_ACTION")),
            _to_float(record_dict.get("composite_score")),
            _to_float(record_dict.get("signal_strength")),
            _to_float(record_dict.get("ml_score")),
            _to_float(record_dict.get("ml_predicted_return")),
            _to_float(record_dict.get("entry_price")),
            _to_float(record_dict.get("suggested_stop_loss")),
            _to_float(record_dict.get("suggested_target_price")),
            json.dumps(record_dict.get("passed_gates") or []) if isinstance(record_dict.get("passed_gates"), (list, tuple)) else _ensure_str(record_dict.get("passed_gates"), "[]"),
            json.dumps(record_dict.get("failed_gates") or []) if isinstance(record_dict.get("failed_gates"), (list, tuple)) else _ensure_str(record_dict.get("failed_gates"), "[]"),
            json.dumps(record_dict.get("reason_codes") or []) if isinstance(record_dict.get("reason_codes"), (list, tuple)) else _ensure_str(record_dict.get("reason_codes"), "[]"),
            int(record_dict.get("proposed_shares", 0)),
            _ensure_str(record_dict.get("binding_constraint")),
            1 if record_dict.get("is_order_proposed") else 0,
            1 if record_dict.get("pre_flight_valid") else (0 if record_dict.get("pre_flight_valid") is not None else None),
            json.dumps(record_dict.get("rejection_reasons") or []) if isinstance(record_dict.get("rejection_reasons"), (list, tuple)) else _ensure_str(record_dict.get("rejection_reasons"), "[]"),
            _ensure_str(record_dict.get("simulated_order_id")),
            _to_float(record_dict.get("simulated_fill_price")),
            _to_float(record_dict.get("simulated_slippage_bps")),
            _to_float(record_dict.get("simulated_commission")),
            _ensure_str(record_dict.get("simulated_status")),
            json.dumps(record_dict.get("input_snapshot") or {}, default=str) if isinstance(record_dict.get("input_snapshot"), dict) else _ensure_str(record_dict.get("input_snapshot_json"), "{}"),
            json.dumps(record_dict.get("component_outputs") or {}, default=str) if isinstance(record_dict.get("component_outputs"), dict) else _ensure_str(record_dict.get("component_outputs_json"), "{}"),
            _to_float(record_dict.get("realized_forward_return_pct")),
            _to_float(record_dict.get("realized_net_return_pct")),
            str(record_dict.get("outcome_evaluated_at") or ""),
            1 if record_dict.get("hit_target") else (0 if record_dict.get("hit_target") is not None else None),
            1 if record_dict.get("hit_stop") else (0 if record_dict.get("hit_stop") is not None else None),
            1 if record_dict.get("is_profitable") else (0 if record_dict.get("is_profitable") is not None else None),
            json.dumps(record_dict, default=str) if not isinstance(record_dict.get("raw_json"), str) else record_dict.get("raw_json"),
        ))
        conn.commit()
        return cursor.lastrowid


def update_shadow_decision_outcome(
    decision_id: int,
    outcome_dict: Optional[dict] = None,
    db_path: str = DB_PATH,
    **kwargs
) -> None:
    """
    Updates the realized outcome metrics of a shadow trade record.
    Supports either passing an outcome_dict or keyword arguments.
    """
    init_db(db_path)
    combined = dict(outcome_dict or {})
    combined.update(kwargs)
    now_str = _now_str()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE shadow_decisions SET
                realized_forward_return_pct = ?,
                realized_net_return_pct = ?,
                outcome_evaluated_at = ?,
                hit_target = ?,
                hit_stop = ?,
                is_profitable = ?
            WHERE id = ?;
        """, (
            _to_float(combined.get("realized_forward_return_pct")),
            _to_float(combined.get("realized_net_return_pct")),
            str(combined.get("outcome_evaluated_at") or now_str),
            1 if combined.get("hit_target") else 0,
            1 if combined.get("hit_stop") else 0,
            1 if combined.get("is_profitable") else 0,
            decision_id,
        ))
        conn.commit()



def get_shadow_decisions(
    session_id: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 1000,
    db_path=DB_PATH
) -> list:
    """
    Fetches shadow decisions matching optional filter criteria, sorted newest first.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM shadow_decisions WHERE 1=1"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(str(session_id))
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            for json_col in ("passed_gates", "failed_gates", "reason_codes", "rejection_reasons", "input_snapshot_json", "component_outputs_json"):
                if d.get(json_col) and isinstance(d[json_col], str):
                    try:
                        d[json_col] = json.loads(d[json_col])
                    except Exception:
                        pass
            results.append(d)
        return results


