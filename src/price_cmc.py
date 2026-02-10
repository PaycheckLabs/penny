# src/price_cmc.py
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# Cache: (symbol, convert) -> (timestamp, response_json)
_CACHE: Dict[Tuple[str, str], Tuple[float, dict]] = {}
CACHE_TTL_SECONDS = int(os.getenv("CMC_CACHE_TTL_SECONDS", "20"))


class CMCError(Exception):
    pass


def _get_api_key() -> str:
    key = (os.getenv("CMC_API_KEY") or "").strip()
    if not key:
        raise CMCError("CMC_API_KEY is not set")
    return key


def fetch_quote(symbol: str, convert: str = "USD", timeout: float = 10.0) -> dict:
    symbol = (symbol or "").strip().upper()
    convert = (convert or "USD").strip().upper()
    if not symbol:
        raise CMCError("Missing symbol (example: BTC)")

    cache_key = (symbol, convert)
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return data

    params = {"symbol": symbol, "convert": convert}
    url = CMC_BASE_URL + "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("X-CMC_PRO_API_KEY", _get_api_key())

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        raise CMCError(f"CMC request failed: {e}") from e

    if not isinstance(data, dict) or "data" not in data:
        raise CMCError("Unexpected CMC response")

    if symbol not in data["data"]:
        raise CMCError(f"Symbol not found: {symbol}")

    payload = data["data"][symbol]
    if isinstance(payload, list) and payload:
        payload = payload[0]

    _CACHE[cache_key] = (now, payload)
    return payload


def format_quote(payload: dict, convert: str = "USD") -> str:
    convert = (convert or "USD").strip().upper()
    name = payload.get("name") or ""
    sym = payload.get("symbol") or ""
    quote = (payload.get("quote") or {}).get(convert) or {}

    price = quote.get("price")
    pct_24h = quote.get("percent_change_24h")
    mcap = quote.get("market_cap")

    if price is None:
        raise CMCError("No price in CMC response")

    # Pretty formatting
    price_str = f"{float(price):,.6f}".rstrip("0").rstrip(".")
    parts = [f"{name} ({sym})", f"Price: {price_str} {convert}"]

    if pct_24h is not None:
        parts.append(f"24h: {float(pct_24h):+.2f}%")
    if mcap is not None:
        parts.append(f"MC: {float(mcap):,.0f} {convert}")

    return " | ".join(parts)


def get_price_line(symbol: str, convert: str = "USD") -> str:
    payload = fetch_quote(symbol=symbol, convert=convert)
    return format_quote(payload, convert=convert)
