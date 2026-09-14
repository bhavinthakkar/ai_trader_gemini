import sqlite3
import json
from datetime import datetime

DB_PATH = "trader.db"


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
            "model_data_completeness": "REAL",
            "deterministic_data_completeness": "REAL"
        }

        cursor.execute("PRAGMA table_info(signals);")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type};")

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
            model_data_completeness = _to_float(item.get("model_data_completeness"))
            deterministic_data_completeness = _to_float(item.get("deterministic_data_completeness"))

            # Backward-compatible text summaries
            reason = _ensure_str(item.get("reason") or bull_case)
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
                    no_trade_reason, model_data_completeness, deterministic_data_completeness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                no_trade_reason, model_data_completeness, deterministic_data_completeness
            ))
        conn.commit()
    print(f"[DB] Successfully saved {len(results)} analysis records to database.")


def get_latest_signals(db_path=DB_PATH) -> list:
    """
    Returns the most recent analysis record for each stock symbol.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*
            FROM signals s
            INNER JOIN (
                SELECT symbol, MAX(id) as max_id
                FROM signals
                GROUP BY symbol
            ) latest ON s.id = latest.max_id
            ORDER BY s.symbol ASC;
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


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
        return [dict(r) for r in rows]


def get_summary_stats(db_path=DB_PATH) -> dict:
    """
    Returns summary metrics (counts, last run date, latest decision breakdown).
    """
    init_db(db_path)
    latest = get_latest_signals(db_path)
    buy_count = sum(1 for s in latest if s["decision"] == "BUY")
    sell_count = sum(1 for s in latest if s["decision"] == "SELL")
    hold_count = sum(1 for s in latest if s["decision"] == "HOLD")

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
        "last_run": last_run or "N/A"
    }


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
