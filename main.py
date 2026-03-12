import os
from pathlib import Path
from time import time
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq

load_dotenv()

COINGECKO_SIMPLE_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
MODEL_NAME = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 10
CACHE_TTL_SECONDS = 20

SYMBOL_MAP = {
    "BTC": {"id": "bitcoin", "name": "Bitcoin"},
    "ETH": {"id": "ethereum", "name": "Ethereum"},
    "SOL": {"id": "solana", "name": "Solana"},
    "BNB": {"id": "binancecoin", "name": "BNB"},
    "XRP": {"id": "ripple", "name": "XRP"},
    "ADA": {"id": "cardano", "name": "Cardano"},
    "DOGE": {"id": "dogecoin", "name": "Dogecoin"},
}
SUPPORTED_SYMBOLS = list(SYMBOL_MAP.keys())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

BASE_DIR = Path(__file__).resolve().parent
MARKET_CACHE: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}

app = FastAPI(title="AI Crypto Market Explainer", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_symbol_details(symbol: str) -> tuple[str, dict[str, str]]:
    cleaned_symbol = symbol.strip().upper()
    details = SYMBOL_MAP.get(cleaned_symbol)
    if not details:
        supported = ", ".join(SUPPORTED_SYMBOLS)
        raise HTTPException(
            status_code=404,
            detail=f"'{cleaned_symbol}' is not supported. Try one of: {supported}",
        )
    return cleaned_symbol, details


def fetch_market_snapshot(coin_ids: list[str]) -> dict[str, Any]:
    cache_key = tuple(coin_ids)
    cached = MARKET_CACHE.get(cache_key)

    if cached and time() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        response = requests.get(
            COINGECKO_SIMPLE_PRICE_URL,
            params={
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        market_data = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="CoinGecko is unavailable right now. Please try again in a moment.",
        ) from exc

    for coin_id in coin_ids:
        if coin_id not in market_data:
            raise HTTPException(
                status_code=502,
                detail="Market data came back incomplete. Please try again.",
            )

    MARKET_CACHE[cache_key] = (time(), market_data)
    return market_data


def get_sentiment(change_24h: float) -> str:
    if change_24h > 2:
        return "bullish"
    if change_24h < -2:
        return "bearish"
    return "neutral"


def build_market_payload(symbol: str, details: dict[str, str], entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": details["name"],
        "price": entry["usd"],
        "volume_24h": entry["usd_24h_vol"],
        "change_24h": entry["usd_24h_change"],
        "sentiment": get_sentiment(entry["usd_24h_change"]),
        "updated_at": int(time()),
    }


def build_fallback_explanation(market: dict[str, Any]) -> str:
    direction = "up" if market["change_24h"] >= 0 else "down"
    return (
        f"{market['name']} is trading near ${market['price']:,.2f} and is {direction} "
        f"{abs(market['change_24h']):.2f}% over the last 24 hours. "
        f"Trading volume is around ${market['volume_24h']:,.0f}, which suggests the market is still active. "
        f"Momentum looks {market['sentiment']} right now based on the short-term move."
    )


def build_fallback_comparison(first: dict[str, Any], second: dict[str, Any]) -> str:
    winner = first if first["change_24h"] >= second["change_24h"] else second
    loser = second if winner is first else first
    return (
        f"{winner['name']} is performing better over the last 24 hours because its move "
        f"({winner['change_24h']:.2f}%) is stronger than {loser['name']} ({loser['change_24h']:.2f}%). "
        f"{winner['name']} is trading near ${winner['price']:,.2f}, while {loser['name']} is near "
        f"${loser['price']:,.2f}. Both assets still show active trading, but {winner['symbol']} has the stronger short-term momentum right now."
    )


def ask_ai(prompt: str, fallback_text: str) -> str:
    if client is None:
        return fallback_text

    try:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain crypto market data clearly, honestly, and without hype. "
                        "Keep answers short, useful, and beginner-friendly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return chat.choices[0].message.content.strip()
    except (APIConnectionError, APIStatusError, APITimeoutError):
        return fallback_text


@app.get("/")
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/crypto")
def get_crypto(symbol: str) -> dict[str, Any]:
    normalized_symbol, details = get_symbol_details(symbol)
    market_data = fetch_market_snapshot([details["id"]])
    return build_market_payload(normalized_symbol, details, market_data[details["id"]])


@app.get("/crypto/explain")
def explain_crypto(symbol: str) -> dict[str, str]:
    normalized_symbol, details = get_symbol_details(symbol)
    market_data = fetch_market_snapshot([details["id"]])
    market = build_market_payload(normalized_symbol, details, market_data[details["id"]])

    prompt = (
        f"Analyze {market['name']} ({market['symbol']}) in 3-4 sentences for a beginner. "
        f"Focus on what the numbers mean and whether short-term momentum looks weak, balanced, or strong.\n\n"
        f"Current price: ${market['price']:,.2f}\n"
        f"24h trading volume: ${market['volume_24h']:,.0f}\n"
        f"24h price change: {market['change_24h']:.2f}%\n"
        f"Current sentiment label: {market['sentiment']}"
    )

    explanation = ask_ai(prompt, build_fallback_explanation(market))
    return {"explanation": explanation}


@app.get("/crypto/compare")
def compare_crypto(symbol1: str, symbol2: str) -> dict[str, Any]:
    normalized_symbol1, details1 = get_symbol_details(symbol1)
    normalized_symbol2, details2 = get_symbol_details(symbol2)

    if normalized_symbol1 == normalized_symbol2:
        raise HTTPException(status_code=400, detail="Choose two different coins to compare.")

    market_data = fetch_market_snapshot([details1["id"], details2["id"]])
    first_market = build_market_payload(normalized_symbol1, details1, market_data[details1["id"]])
    second_market = build_market_payload(normalized_symbol2, details2, market_data[details2["id"]])

    prompt = (
        f"Compare {first_market['name']} ({first_market['symbol']}) and {second_market['name']} ({second_market['symbol']}) "
        f"in 4-5 sentences for a beginner. Explain which asset looks stronger in the last 24 hours and why.\n\n"
        f"{first_market['symbol']}: price ${first_market['price']:,.2f}, change {first_market['change_24h']:.2f}%, volume ${first_market['volume_24h']:,.0f}\n"
        f"{second_market['symbol']}: price ${second_market['price']:,.2f}, change {second_market['change_24h']:.2f}%, volume ${second_market['volume_24h']:,.0f}"
    )

    comparison = ask_ai(prompt, build_fallback_comparison(first_market, second_market))
    return {
        "left": first_market,
        "right": second_market,
        "comparison": comparison,
    }
