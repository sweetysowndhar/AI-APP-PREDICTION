import pandas as pd
import numpy as np
import datetime
import math
try:
    import nsepython as nse
except ImportError:
    nse = None

class OptionsEngine:
    def __init__(self):
        self.is_available = nse is not None

    def fetch_option_chain(self, symbol):
        """Fetches the live option chain from NSE."""
        if not self.is_available: return None
        try:
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                chain = nse.option_chain(symbol)
            else:
                chain = nse.option_chain(symbol)
            
            if not chain or 'filtered' not in chain or 'data' not in chain['filtered']:
                return None
                
            return chain
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return None

    def calculate_oi_walls(self, chain):
        """Finds the strikes with Highest Call OI and Highest Put OI."""
        if not chain: return {"resistance_strike": None, "support_strike": None}
        
        data = chain['filtered']['data']
        max_ce_oi = 0
        max_pe_oi = 0
        res_strike = None
        sup_strike = None
        
        for strike_data in data:
            strike = strike_data.get('strikePrice')
            
            # Call side
            if 'CE' in strike_data:
                ce_oi = strike_data['CE'].get('openInterest', 0)
                if ce_oi > max_ce_oi:
                    max_ce_oi = ce_oi
                    res_strike = strike
                    
            # Put side
            if 'PE' in strike_data:
                pe_oi = strike_data['PE'].get('openInterest', 0)
                if pe_oi > max_pe_oi:
                    max_pe_oi = pe_oi
                    sup_strike = strike
                    
        return {"resistance_strike": res_strike, "support_strike": sup_strike, "max_ce_oi": max_ce_oi, "max_pe_oi": max_pe_oi}

    def calculate_pcr(self, chain):
        """Calculates Put Call Ratio."""
        if not chain: return None
        
        tot_ce_oi = chain.get('filtered', {}).get('CE', {}).get('totOI', 0)
        tot_pe_oi = chain.get('filtered', {}).get('PE', {}).get('totOI', 0)
        
        if tot_ce_oi == 0: return 1.0
        pcr = tot_pe_oi / tot_ce_oi
        
        sentiment = "Neutral"
        if pcr < 0.8: sentiment = "Bearish"
        elif pcr > 1.2: sentiment = "Bullish"
        
        return {"pcr": round(pcr, 2), "sentiment": sentiment}

    def calculate_max_pain(self, chain):
        """Calculates Max Pain strike."""
        if not chain: return None
        data = chain['filtered']['data']
        
        strikes = [d['strikePrice'] for d in data]
        pain_values = {}
        
        for test_strike in strikes:
            total_pain = 0
            for strike_data in data:
                strike = strike_data['strikePrice']
                
                # Call Option Pain (Intrinsic value if expires at test_strike)
                if 'CE' in strike_data:
                    oi = strike_data['CE'].get('openInterest', 0)
                    if test_strike > strike:
                        total_pain += (test_strike - strike) * oi
                        
                # Put Option Pain (Intrinsic value if expires at test_strike)
                if 'PE' in strike_data:
                    oi = strike_data['PE'].get('openInterest', 0)
                    if test_strike < strike:
                        total_pain += (strike - test_strike) * oi
                        
            pain_values[test_strike] = total_pain
            
        if not pain_values: return None
        # Strike with minimum total pain
        max_pain_strike = min(pain_values, key=pain_values.get)
        return max_pain_strike

    def scan_iv_rank(self, chain):
        """Scans Implied Volatility (IV). Simplified version since we don't have 1-year historical IV."""
        if not chain: return {"iv": None, "preference": "Neutral"}
        data = chain['filtered']['data']
        
        # Find ATM strike by getting strike closest to underlying value
        underlying = chain.get('records', {}).get('underlyingValue', 0)
        if underlying == 0: return {"iv": None, "preference": "Neutral"}
        
        closest_strike = None
        min_diff = float('inf')
        atm_ce_iv = 0
        atm_pe_iv = 0
        
        for strike_data in data:
            strike = strike_data['strikePrice']
            diff = abs(strike - underlying)
            if diff < min_diff:
                min_diff = diff
                closest_strike = strike
                if 'CE' in strike_data: atm_ce_iv = strike_data['CE'].get('impliedVolatility', 0)
                if 'PE' in strike_data: atm_pe_iv = strike_data['PE'].get('impliedVolatility', 0)
                
        avg_iv = (atm_ce_iv + atm_pe_iv) / 2
        
        # Simple proxy: IV > 25 is High (Sell), IV < 15 is Low (Buy)
        preference = "Neutral"
        if avg_iv > 25: preference = "Sell Options"
        elif avg_iv > 0 and avg_iv < 15: preference = "Buy Options"
        
        return {"iv": round(avg_iv, 2), "preference": preference, "atm_strike": closest_strike}

    def detect_oi_buildup(self, chain):
        """Detects OI buildup (Long Build-up, Short Build-up, Short Covering, Long Unwinding)."""
        if not chain: return None
        data = chain['filtered']['data']
        
        # Aggregate ATM buildup
        underlying = chain.get('records', {}).get('underlyingValue', 0)
        if underlying == 0: return None
        
        atm_strike = min([d['strikePrice'] for d in data], key=lambda x: abs(x - underlying))
        atm_data = next((d for d in data if d['strikePrice'] == atm_strike), None)
        
        if not atm_data or 'CE' not in atm_data: return None
        
        ce_oi_change = atm_data['CE'].get('changeinOpenInterest', 0)
        ce_price_change = atm_data['CE'].get('pChange', 0)
        
        buildup = "Neutral"
        if ce_price_change > 0 and ce_oi_change > 0:
            buildup = "🟢 Long Build-up (Bullish)"
        elif ce_price_change < 0 and ce_oi_change > 0:
            buildup = "🔴 Short Build-up (Bearish)"
        elif ce_price_change > 0 and ce_oi_change < 0:
            buildup = "🟢 Short Covering (Strong Bullish)"
        elif ce_price_change < 0 and ce_oi_change < 0:
            buildup = "🔴 Long Unwinding (Weakness)"
            
        return buildup

    def get_full_analysis(self, symbol):
        """Runs all options analysis on a symbol."""
        chain = self.fetch_option_chain(symbol)
        if not chain:
            return None
            
        return {
            "symbol": symbol,
            "underlying": chain.get('records', {}).get('underlyingValue', 0),
            "oi_walls": self.calculate_oi_walls(chain),
            "pcr": self.calculate_pcr(chain),
            "max_pain": self.calculate_max_pain(chain),
            "iv": self.scan_iv_rank(chain),
            "buildup": self.detect_oi_buildup(chain)
        }

if __name__ == "__main__":
    engine = OptionsEngine()
    print("Testing NIFTY...")
    res = engine.get_full_analysis("NIFTY")
    print(res)
