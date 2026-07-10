import pandas as pd
import numpy as np
import datetime
import math
import yfinance as yf

# Try nsepython for live OI data
try:
    import nsepython as nse
    # Quick connection test
    _test = nse is not None
except Exception:
    nse = None

class OptionsEngine:
    def __init__(self):
        # Try to verify nsepython actually works (not just imported)
        self.nse_live = False
        if nse is not None:
            try:
                # Light test — just check module attributes
                if hasattr(nse, 'option_chain'):
                    self.nse_live = True
            except Exception:
                self.nse_live = False
        # Always available — fallback to yfinance if nse fails
        self.is_available = True

    def fetch_option_chain(self, symbol):
        """Fetches the live option chain from NSE."""
        if not self.nse_live:
            return None
        try:
            chain = nse.option_chain(symbol)
            if not chain or 'filtered' not in chain or 'data' not in chain['filtered']:
                return None
            return chain
        except Exception as e:
            print(f"NSE option chain error for {symbol}: {e}")
            return None

    def get_yf_data(self, symbol):
        """Get underlying price and basic info via yfinance as fallback."""
        try:
            ticker_map = {
                'NIFTY':     '^NSEI',
                'BANKNIFTY': '^NSEBANK',
                'FINNIFTY':  '^CNXFIN',
            }
            ticker = ticker_map.get(symbol, f"{symbol}.NS")
            tk = yf.Ticker(ticker)
            info = tk.fast_info
            price = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
            return float(price) if price else None
        except Exception:
            return None

    def calculate_oi_walls(self, chain):
        """Finds strikes with Highest Call OI and Highest Put OI."""
        if not chain:
            return {"resistance_strike": None, "support_strike": None, "max_ce_oi": 0, "max_pe_oi": 0}
        data = chain['filtered']['data']
        max_ce_oi, max_pe_oi = 0, 0
        res_strike, sup_strike = None, None
        for strike_data in data:
            strike = strike_data.get('strikePrice')
            if 'CE' in strike_data:
                ce_oi = strike_data['CE'].get('openInterest', 0)
                if ce_oi > max_ce_oi:
                    max_ce_oi = ce_oi
                    res_strike = strike
            if 'PE' in strike_data:
                pe_oi = strike_data['PE'].get('openInterest', 0)
                if pe_oi > max_pe_oi:
                    max_pe_oi = pe_oi
                    sup_strike = strike
        return {"resistance_strike": res_strike, "support_strike": sup_strike,
                "max_ce_oi": max_ce_oi, "max_pe_oi": max_pe_oi}

    def calculate_pcr(self, chain):
        """Calculates Put Call Ratio."""
        if not chain:
            return None
        tot_ce_oi = chain.get('filtered', {}).get('CE', {}).get('totOI', 0)
        tot_pe_oi = chain.get('filtered', {}).get('PE', {}).get('totOI', 0)
        if tot_ce_oi == 0:
            return {"pcr": 1.0, "sentiment": "Neutral"}
        pcr = tot_pe_oi / tot_ce_oi
        sentiment = "Neutral"
        if pcr < 0.8:
            sentiment = "Bearish"
        elif pcr > 1.2:
            sentiment = "Bullish"
        return {"pcr": round(pcr, 2), "sentiment": sentiment}

    def calculate_max_pain(self, chain):
        """Calculates Max Pain strike."""
        if not chain:
            return None
        data = chain['filtered']['data']
        strikes = [d['strikePrice'] for d in data]
        pain_values = {}
        for test_strike in strikes:
            total_pain = 0
            for strike_data in data:
                strike = strike_data['strikePrice']
                if 'CE' in strike_data:
                    oi = strike_data['CE'].get('openInterest', 0)
                    if test_strike > strike:
                        total_pain += (test_strike - strike) * oi
                if 'PE' in strike_data:
                    oi = strike_data['PE'].get('openInterest', 0)
                    if test_strike < strike:
                        total_pain += (strike - test_strike) * oi
            pain_values[test_strike] = total_pain
        if not pain_values:
            return None
        return min(pain_values, key=pain_values.get)

    def scan_iv_rank(self, chain):
        """Scans Implied Volatility."""
        if not chain:
            return {"iv": None, "preference": "Neutral"}
        data = chain['filtered']['data']
        underlying = chain.get('records', {}).get('underlyingValue', 0)
        if underlying == 0:
            return {"iv": None, "preference": "Neutral"}
        closest_strike, min_diff = None, float('inf')
        atm_ce_iv, atm_pe_iv = 0, 0
        for strike_data in data:
            strike = strike_data['strikePrice']
            diff = abs(strike - underlying)
            if diff < min_diff:
                min_diff = diff
                closest_strike = strike
                if 'CE' in strike_data:
                    atm_ce_iv = strike_data['CE'].get('impliedVolatility', 0)
                if 'PE' in strike_data:
                    atm_pe_iv = strike_data['PE'].get('impliedVolatility', 0)
        avg_iv = (atm_ce_iv + atm_pe_iv) / 2
        preference = "Neutral"
        if avg_iv > 25:
            preference = "Sell Options"
        elif avg_iv > 0 and avg_iv < 15:
            preference = "Buy Options"
        return {"iv": round(avg_iv, 2), "preference": preference, "atm_strike": closest_strike}

    def detect_oi_buildup(self, chain):
        """Detects OI buildup type."""
        if not chain:
            return None
        data = chain['filtered']['data']
        underlying = chain.get('records', {}).get('underlyingValue', 0)
        if underlying == 0:
            return None
        atm_strike = min([d['strikePrice'] for d in data], key=lambda x: abs(x - underlying))
        atm_data = next((d for d in data if d['strikePrice'] == atm_strike), None)
        if not atm_data or 'CE' not in atm_data:
            return None
        ce_oi_change    = atm_data['CE'].get('changeinOpenInterest', 0)
        ce_price_change = atm_data['CE'].get('pChange', 0)
        if ce_price_change > 0 and ce_oi_change > 0:
            return "🟢 Long Build-up (Bullish)"
        elif ce_price_change < 0 and ce_oi_change > 0:
            return "🔴 Short Build-up (Bearish)"
        elif ce_price_change > 0 and ce_oi_change < 0:
            return "🟢 Short Covering (Strong Bullish)"
        elif ce_price_change < 0 and ce_oi_change < 0:
            return "🔴 Long Unwinding (Weakness)"
        return "Neutral"

    def get_yf_analysis(self, symbol):
        """
        Fallback analysis using yfinance when NSE live data is unavailable.
        Returns estimated levels based on current price.
        """
        price = self.get_yf_data(symbol)
        if not price:
            return None

        # Round to nearest 50 for index options
        step = 50
        atm = round(price / step) * step
        resistance = atm + step * 2
        support    = atm - step * 2

        # Simple RSI-style momentum from recent data
        ticker_map = {'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'FINNIFTY': '^CNXFIN'}
        ticker = ticker_map.get(symbol, f"{symbol}.NS")
        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1h")
            if not hist.empty:
                last_close  = hist['Close'].iloc[-1]
                prev_close  = hist['Close'].iloc[-5] if len(hist) > 5 else hist['Close'].iloc[0]
                pct_change  = ((last_close - prev_close) / prev_close) * 100
                buildup = "🟢 Long Build-up (Bullish)" if pct_change > 0.5 else \
                          "🔴 Short Build-up (Bearish)" if pct_change < -0.5 else "Neutral"
            else:
                buildup = "Neutral"
        except Exception:
            buildup = "Neutral"

        return {
            "symbol":     symbol,
            "underlying": round(price, 2),
            "oi_walls": {
                "resistance_strike": resistance,
                "support_strike":    support,
                "max_ce_oi":         0,
                "max_pe_oi":         0,
            },
            "pcr":      {"pcr": 1.0, "sentiment": "Neutral (Live OI unavailable)"},
            "max_pain": atm,
            "iv":       {"iv": "N/A", "preference": "Neutral"},
            "buildup":  buildup,
            "source":   "yfinance (NSE live data unavailable)",
        }

    def get_full_analysis(self, symbol):
        """Runs full options analysis — uses NSE if available, yfinance as fallback."""
        # Try NSE live first
        if self.nse_live:
            chain = self.fetch_option_chain(symbol)
            if chain:
                return {
                    "symbol":     symbol,
                    "underlying": chain.get('records', {}).get('underlyingValue', 0),
                    "oi_walls":   self.calculate_oi_walls(chain),
                    "pcr":        self.calculate_pcr(chain),
                    "max_pain":   self.calculate_max_pain(chain),
                    "iv":         self.scan_iv_rank(chain),
                    "buildup":    self.detect_oi_buildup(chain),
                    "source":     "NSE Live",
                }
        # Fallback to yfinance
        return self.get_yf_analysis(symbol)


if __name__ == "__main__":
    engine = OptionsEngine()
    print("NSE Live:", engine.nse_live)
    print("Testing NIFTY...")
    res = engine.get_full_analysis("NIFTY")
    print(res)
