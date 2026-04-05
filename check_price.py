import yfinance as yf
import json

def get_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # yfinance info contains the latest market data
        info = ticker.info
        current_price = info.get("regularMarketPrice")
        if current_price is None:
            # Fallback to history if info is incomplete
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
        
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "currency": info.get("currency"),
            "prev_close": info.get("previousClose"),
            "open": info.get("open"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "timestamp": info.get("regularMarketTime", "Unknown")
        }
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    ticker_id = "4005.T"
    data = get_price(ticker_id)
    print(json.dumps(data, indent=2))
