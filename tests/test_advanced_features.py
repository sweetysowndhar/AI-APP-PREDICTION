import sys
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import unittest

# Mock Streamlit and prediction tracker to prevent streamlit and file-loading errors on import
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

mock_tracker = MagicMock()
mock_tracker.save_prediction = MagicMock()
mock_tracker.load_history = MagicMock(return_value=[])
mock_tracker.load_advanced_stats = MagicMock(return_value={})
sys.modules['prediction_tracker'] = mock_tracker

# Import AIEngine from ai_prediction_app
from ai_prediction_app import AIEngine

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.engine = AIEngine()

    def test_detect_liquidity_sweeps_bullish(self):
        """Verify that a bullish liquidity sweep is identified correctly."""
        # Create a series with a swing low at index 5
        lows = [100.0] * 20
        highs = [105.0] * 20
        closes = [102.0] * 20
        
        # Set swing low at index 5
        lows[5] = 95.0
        # A swing low requires window=2, so it's lower than indices 3, 4, 6, 7
        
        # Now create a bullish sweep at index 18:
        # Low at index 18 drops below the swing low (95.0), say to 94.0, but Close at index 18 is 96.0 (back above 95.0)
        lows[18] = 94.0
        closes[18] = 96.0
        
        df = pd.DataFrame({
            'Open': [102.0] * 20,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': [1000] * 20
        })
        
        sweeps = self.engine.detect_liquidity_sweeps(df)
        self.assertTrue(sweeps['bullish'])
        self.assertFalse(sweeps['bearish'])

    def test_detect_liquidity_sweeps_bearish(self):
        """Verify that a bearish liquidity sweep is identified correctly."""
        # Create a series with a swing high at index 5
        lows = [95.0] * 20
        highs = [100.0] * 20
        closes = [98.0] * 20
        
        # Set swing high at index 5
        highs[5] = 105.0
        
        # Now create a bearish sweep at index 18:
        # High at index 18 goes above the swing high (105.0), say to 106.0, but Close at index 18 is 104.0 (back below 105.0)
        highs[18] = 106.0
        closes[18] = 104.0
        
        df = pd.DataFrame({
            'Open': [98.0] * 20,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': [1000] * 20
        })
        
        sweeps = self.engine.detect_liquidity_sweeps(df)
        self.assertFalse(sweeps['bullish'])
        self.assertTrue(sweeps['bearish'])

    def test_triple_timeframe_alignment(self):
        """Verify triple timeframe confirmation logic in prediction results."""
        # Create dummy dataframes for 15m, 1h, and 1d
        # We need their trends to align.
        df_bull = pd.DataFrame({
            'Open': [100.0] * 25,
            'High': [105.0] * 25,
            'Low': [95.0] * 25,
            'Close': [100.0] * 24 + [110.0],
            'Volume': [1000] * 25
        })
        
        status = self.engine.get_timeframe_status(df_bull)
        self.assertEqual(status['trend'], 'Bullish')
        
        # Now set up the predict call mock and check triple_confirm
        self.engine.models['TEST_SYM'] = {
            'd1': {'rf': MagicMock(), 'gb': MagicMock(), 'acc': 0.8},
            'd2': {'rf': MagicMock(), 'gb': MagicMock(), 'acc': 0.8},
            'd3': {'rf': MagicMock(), 'gb': MagicMock(), 'acc': 0.8},
            'd4': {'rf': MagicMock(), 'gb': MagicMock(), 'acc': 0.8}
        }
        
        for key in ['d1', 'd2', 'd3', 'd4']:
            self.engine.models['TEST_SYM'][key]['rf'].classes_ = [1, -1]
            self.engine.models['TEST_SYM'][key]['gb'].classes_ = [1, -1]
            self.engine.models['TEST_SYM'][key]['rf'].predict_proba = MagicMock(return_value=[[0.6, 0.4]])
            self.engine.models['TEST_SYM'][key]['gb'].predict_proba = MagicMock(return_value=[[0.6, 0.4]])
            
        sc_mock = MagicMock()
        sc_mock.transform = MagicMock(return_value=[[0.0]*14])
        self.engine.scalers['TEST_SYM'] = sc_mock
        
        prices = [100.0] * 30
        volumes = [1000] * 30
        
        # Run predict with identical Bullish status on 15m, 1h, 1d
        res = self.engine.predict(
            'TEST_SYM', prices, volumes,
            df=df_bull, df_1h=df_bull, df_1d=df_bull, df_15m=df_bull
        )
        
        self.assertIsNotNone(res)
        self.assertEqual(res['triple_confirm'], 'Bullish')

if __name__ == '__main__':
    unittest.main()
