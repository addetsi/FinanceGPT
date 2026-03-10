from sec_edgar_downloader import Downloader
import os
from dotenv import load_dotenv

load_dotenv()

dl = Downloader("FinanceGPT", os.getenv("SEC_EMAIL"), "data/sec_filings")

companies = {
    "Tech":       ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "ORCL", "CRM", "INTC", "AMD"],
    "Finance":    ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "USB", "PNC"],
    "Healthcare": ["JNJ", "PFE", "UNH", "CVS", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "Energy":     ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY"],
    "Retail":     ["WMT", "AMZN", "HD", "TGT", "COST", "LOW", "EBAY", "ETSY", "KR", "DG"]
}

for sector, tickers in companies.items():
    print(f"\nDownloading {sector} sector...")
    for ticker in tickers:
        try:
            dl.get("10-K", ticker, limit=3)
            print(f"  Done {ticker}")
        except Exception as e:
            print(f"  Failed {ticker}: {e}")