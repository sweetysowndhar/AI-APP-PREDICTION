import pandas as pd
import numpy as np
from intraday_execution import IntradayExecutionEngine

def run_test():
    # 1. Generate mock intraday data
    # Let's create a DataFrame with 15-minute intervals starting at 9:15 AM
    times = pd.date_range('2026-06-26 09:15:00', periods=30, freq='15min')
    
    # Simple price movement: price starts at 100, goes up to 105, then down
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0.2, 0.5, size=30))
    highs = prices + 0.5
    lows = prices - 0.5
    opens = prices - 0.1
    volumes = [1000 + np.random.randint(-200, 200) for _ in range(30)]
    # Add a huge volume spike at the end to trigger RVOL > 1.5
    volumes[-1] = 3000
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=times)
    
    # 2. Compute VWAP
    vwap = IntradayExecutionEngine.calculate_vwap(df)
    print("VWAP calculated: last value is", vwap.iloc[-1])
    assert not vwap.empty, "VWAP Series is empty"
    
    # 3. Get ORB
    orb_high, orb_low = IntradayExecutionEngine.get_orb_range(df)
    print(f"ORB Range High: {orb_high}, Low: {orb_low}")
    assert orb_high is not None and orb_low is not None, "ORB range is None"
    
    # 4. Compute RVOL
    rvol = IntradayExecutionEngine.calculate_rvol(df)
    print("RVOL computed:", rvol)
    assert rvol > 1.5, "RVOL should be > 1.5 due to the spike"
    
    # 5. Evaluate Entry
    res = IntradayExecutionEngine.evaluate_entry(
        df=df,
        signal_direction="BUY",
        mtf_sync=True,
        liquidity_sweep=True,
        ob_prox=True,
        news_score=1.0
    )
    print("Execution Evaluation:", res)
    assert res['score'] > 0, "Institutional Entry Score should be greater than 0"
    print("All checks passed successfully!")

if __name__ == "__main__":
    run_test()
