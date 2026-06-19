import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_prediction_app import AIEngine

def test():
    df = pd.DataFrame({
        'Open': np.random.rand(100),
        'High': np.random.rand(100) + 1,
        'Low': np.random.rand(100),
        'Close': np.random.rand(100) + 0.5,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=pd.date_range('2023-01-01', periods=100))
    engine = AIEngine()
    try:
        engine.predict('RELIANCE', df['Close'].values, df['Volume'].values, df=df)
        print("Success")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test()
