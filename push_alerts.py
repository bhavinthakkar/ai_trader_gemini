import requests

TOKEN = "8837106861:AAEHwkeU67Mw8mAJaTyRvC0YGEMuc4s7ecY"
CHAT_ID = "969601315"

message = """
BUY SIGNAL

AAPL
Confidence: 0.84
"""

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

requests.post(
    url,
    json={
        "chat_id": CHAT_ID,
        "text": message
    }
)
