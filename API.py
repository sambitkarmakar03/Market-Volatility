import os
import finnhub
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=api_key)

print(finnhub_client.quote("AAPL"))