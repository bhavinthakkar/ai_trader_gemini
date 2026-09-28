"""
Telegram Digest Formatter and Delivery Utility
=============================================
Handles formatting, markdown sanitization, safe chunking (<= 3800 chars),
and delivery of swing trading digests to Telegram with plain-text fallback.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import requests


def _clean_telegram_markdown(text: str) -> str:
    """Sanitize text to avoid malformed markdown entities in Telegram Markdown v1."""
    if not text:
        return ""
    # Strip backticks, asterisks, and replace underscores to avoid broken formatting pairs
    s = str(text).replace("_", " ").replace("*", "").replace("`", "'").strip()
    while s.startswith(("- ", "* ", "• ")):
        s = s[2:].strip()
    return s


def _truncate_telegram_text(text: str, max_chars: int = 140) -> str:
    """Truncate text cleanly to a maximum character count at a word boundary."""
    cleaned = _clean_telegram_markdown(text)
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars].rsplit(" ", 1)[0]
    if not truncated:
        truncated = cleaned[:max_chars]
    return truncated.rstrip(".,;:- ") + "..."


def split_telegram_text(text: str, max_chars: int = 3800) -> List[str]:
    """
    Split text into chunks that strictly fit within max_chars (Telegram max limit: 4096).
    Prefers splitting between double newlines (paragraphs/stocks), then single newlines (lines),
    and falls back to character slicing if a single line exceeds max_chars.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    blocks = text.split("\n\n")
    current_chunk = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if len(block) > max_chars:
            lines = block.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if len(line) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    for i in range(0, len(line), max_chars):
                        chunks.append(line[i:i + max_chars])
                else:
                    if current_chunk and (len(current_chunk) + len(line) + 1 > max_chars):
                        chunks.append(current_chunk)
                        current_chunk = line
                    else:
                        current_chunk = f"{current_chunk}\n{line}".strip() if current_chunk else line
        else:
            if current_chunk and (len(current_chunk) + len(block) + 2 > max_chars):
                chunks.append(current_chunk)
                current_chunk = block
            else:
                current_chunk = f"{current_chunk}\n\n{block}".strip() if current_chunk else block

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def format_telegram_digest(results: List[Dict[str, Any]], model_label: str = "Nemotron-3 Super 120B") -> str:
    """Formats a concise, truncated digest of model trade signals for Telegram."""
    clean_model_label = _clean_telegram_markdown(model_label)
    lines = [f"⚡ *Gloomberb Multi-Pillar RAG Digest ({clean_model_label})* ⚡\n"]
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

    for item in results:
        stock = item.get("stock", "N/A")
        decision = str(item.get("decision", "HOLD")).upper()
        confidence = item.get("confidence", 0.70)
        buy_score = item.get("buy_score", 0.0)
        hold_score = item.get("hold_score", 0.0)
        sell_score = item.get("sell_score", 0.0)
        horizon = item.get("horizon_days", 10)
        quant_score = item.get("quant_score", 65.0)
        pillars = item.get("pillar_scores", {})
        completeness = item.get("data_completeness", 0.85)
        emoji = emoji_map.get(decision, "⚪")

        lines.append(f"{emoji} *{stock}* | *{decision}* (Conf: {confidence} | {horizon}d Horizon)")
        no_trade_reason = item.get("no_trade_reason")
        if no_trade_reason:
            lines.append(f"• *No-Trade:* `{no_trade_reason}`")
        vol_factor = item.get("vol_factor", 1.0)
        atr_pct = item.get("atr_pct", 0.0)
        coverage_pct = int((completeness or 0.0) * 100)
        lines.append(f"• *Quant Score:* `{quant_score}/100` | *Vol Factor:* `{vol_factor}` (ATR {atr_pct}%) | *Data Coverage:* `{coverage_pct}%`")

        rr = item.get("reward_risk_ratio")
        breakeven = item.get("breakeven_win_rate")
        if rr is not None:
            be_str = f"{breakeven * 100:.0f}%" if isinstance(breakeven, (int, float)) else "N/A"
            lines.append(f"• *Reward:Risk:* `{rr}` (breakeven win rate: `{be_str}`)")
        lines.append(f"• *Pillars:* Trend: {pillars.get('trend', 0)} | Sector: {pillars.get('sector', 0)} | Alpha: {pillars.get('alpha', 0)} | ValHist: {pillars.get('valuation_history', 0)} | PeerVal: {pillars.get('peer_valuation', 0)}")
        lines.append(f"• *Probabilities:* Buy: {buy_score} | Hold: {hold_score} | Sell: {sell_score}")
        driver = item.get("primary_driver")
        if driver and str(driver).strip() and str(driver).strip().upper() != "NONE":
            lines.append(f"• *Primary Driver:* `{driver}`")

        # Truncated key qualitative thesis points
        bull_items = [b for b in item.get("bull_case", []) if str(b).strip() and str(b).strip() != "None"]
        max_bull = 2 if decision == "BUY" else 1
        bull_subset = [_truncate_telegram_text(b, 140) for b in bull_items[:max_bull]]
        bull_subset = [b for b in bull_subset if b]
        if len(bull_subset) == 1:
            lines.append(f"• *Bull Case:* _{bull_subset[0]}_")
        elif len(bull_subset) > 1:
            lines.append("• *Bull Case:*")
            for b in bull_subset:
                lines.append(f"  - _{b}_")

        bear_items = [b for b in item.get("bear_case", []) if str(b).strip() and str(b).strip() != "None"]
        max_bear = 2 if decision == "SELL" else 1
        bear_subset = [_truncate_telegram_text(b, 140) for b in bear_items[:max_bear]]
        bear_subset = [b for b in bear_subset if b]
        if len(bear_subset) == 1:
            lines.append(f"• *Bear Case:* _{bear_subset[0]}_")
        elif len(bear_subset) > 1:
            lines.append("• *Bear Case:*")
            for b in bear_subset:
                lines.append(f"  - _{b}_")

        risk_items = [r for r in item.get("key_risks", []) if str(r).strip() and str(r).strip() != "None"]
        risk_subset = [_truncate_telegram_text(r, 140) for r in risk_items[:1]]
        risk_subset = [r for r in risk_subset if r]
        if risk_subset:
            lines.append(f"• *Key Risk:* _{risk_subset[0]}_")

        missing_items = [m for m in item.get("missing_information", []) if str(m).strip() and str(m).strip() != "None"]
        if missing_items:
            m_text = _truncate_telegram_text(missing_items[0], 100)
            if m_text:
                lines.append(f"• *Missing Info:* _{m_text}_")

        lines.append("")

    return "\n".join(lines)


def send_telegram_digest(token: str, chat_id: str, text: str) -> None:
    """Sends text to Telegram, safely chunking into <= 3800 character parts."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = split_telegram_text(text, max_chars=3800)

    if not chunks:
        print("\nTelegram digest: no content to send.")
        return

    def _post_chunk(chunk_text: str) -> requests.Response:
        res = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": chunk_text,
                "parse_mode": "Markdown"
            },
            timeout=15
        )
        # If Telegram rejects markdown formatting (e.g. entities parsing error), fallback to plain text
        if res.status_code != 200 and "can't parse entities" in res.text:
            res = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk_text
                },
                timeout=15
            )
        return res

    if len(chunks) == 1:
        try:
            res = _post_chunk(chunks[0])
            if res.status_code == 200:
                print("\nTelegram digest message sent successfully!")
            else:
                print(f"\nTelegram error: {res.status_code} {res.text}")
        except Exception as e:
            print(f"\nTelegram connection error: {e}")
    else:
        for idx, chunk in enumerate(chunks):
            try:
                res = _post_chunk(chunk)
                if res.status_code == 200:
                    print(f"Telegram digest part {idx+1}/{len(chunks)} sent.")
                else:
                    print(f"Telegram error on part {idx+1}: {res.status_code} {res.text}")
            except Exception as e:
                print(f"Telegram connection error on part {idx+1}: {e}")
            time.sleep(1)
