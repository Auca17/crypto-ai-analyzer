import requests
from groq import Groq          # ← add this line
from fastapi import FastAPI
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)   

app = FastAPI()

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
    coin = SYMBOL_MAP.get(symbol.upper())    # ← add this line
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
    coin = SYMBOL_MAP.get(symbol.upper())    # ← add this line
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
    response = requests.get(url)
    data = response.json()

    price = data[coin]["usd"]
    volume = data[coin]["usd_24h_vol"]
    change = data[coin]["usd_24h_change"]

    # Step 2 - build the prompt
    prompt = f"{symbol} price: ${price}, 24h volume: ${volume:.0f}, 24h change: {change:.2f}%. Explain this in simple terms."

    # Step 3 - call Groq AI
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"explanation": chat.choices[0].message.content}
