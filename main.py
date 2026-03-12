import requests
from groq import Groq
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)   

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")   # ← add this

SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
}

@app.get("/")
def home():
    return {"message": "Crypto AI Analyzer running"}

@app.get("/crypto")
def get_crypto(symbol: str):
    coin = SYMBOL_MAP.get(symbol.upper())
    if not coin:
        return {"error": f"'{symbol.upper()}' is not supported. Try: {list(SYMBOL_MAP.keys())}"}
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
    response = requests.get(url)
    data = response.json()
    return {
        "symbol": symbol.upper(),
        "price": data[coin]["usd"],
        "volume_24h": data[coin]["usd_24h_vol"],
        "change_24h": data[coin]["usd_24h_change"],
    }

@app.get("/crypto/explain")
def explain_crypto(symbol: str):
    coin = SYMBOL_MAP.get(symbol.upper())
    if not coin:
        return {"error": f"'{symbol.upper()}' is not supported. Try: {list(SYMBOL_MAP.keys())}"}
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
    response = requests.get(url)
    data = response.json()

    price = data[coin]["usd"]
    volume = data[coin]["usd_24h_vol"]
    change = data[coin]["usd_24h_change"]

    # Step 2 - build the prompt
    prompt = (
        f"You are a friendly crypto market analyst. "
        f"Analyze this data for {symbol.upper()} and explain it in 3-4 sentences "
        f"in plain language that a beginner can understand. "
        f"Focus on what the numbers MEAN, not just what they are.\n\n"
        f"- Current price: ${price:,.2f}\n"
        f"- 24h trading volume: ${volume:,.0f}\n"
        f"- 24h price change: {change:.2f}%"
    )

    # Step 3 - call Groq AI
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"explanation": chat.choices[0].message.content}
