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
    Initializes the SQLite database schema if tables do not exist.
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
                news TEXT,
                investment_bank_coverage TEXT,
                risk_assessment TEXT,
                pe_and_peg TEXT,
                model_used TEXT,
                raw_json TEXT
            );
        """)
        conn.commit()


def save_results(results: list, model_used: str = "Gemini 3.6 Flash", db_path=DB_PATH):
    """
    Saves a list of analysis result dicts into the signals table.
    """
    if not results:
        return

    init_db(db_path)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for item in results:
            stock = item.get("stock", item.get("symbol", "N/A"))
            decision = str(item.get("decision", "HOLD")).upper()
            confidence = item.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0

            reason = item.get("reason", "")
            news = item.get("news", "")
            bank_coverage = item.get("investment_bank_coverage", "")
            risk_info = item.get("risk_assessment", "")
            pe_peg = item.get("PE_and_PEG", "")
            raw_json = json.dumps(item)

            cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, decision, confidence, reason, news,
                    investment_bank_coverage, risk_assessment, pe_and_peg, model_used, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, stock, decision, confidence, reason, news,
                bank_coverage, risk_info, pe_peg, model_used, raw_json
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
