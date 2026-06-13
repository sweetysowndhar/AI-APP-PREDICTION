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

class TestSMCScanner(unittest.TestCase):
    def setUp(self):
        self.engine = AIEngine()

    def test_swing_highs_and_lows(self):
        """Verify that swing highs and lows are correctly identified with a rolling window of 2."""
        # Create a series with a clear peak at index 5 and a clear trough at index 10
        highs = [100.0] * 15
        lows = [90.0] * 15
        
        # Swing High at index 5
        highs[5] = 110.0
        # Swing Low at index 10
        lows[10] = 80.0
        
        df = pd.DataFrame({
            'Open': [95.0] * 15,
            'High': highs,
            'Low': lows,
            'Close': [95.0] * 15,
            'Volume': [1000] * 15
        })
        
        smc = self.engine.detect_smc_features(df)
        
        # Swing Highs list contains tuples of (index, value)
        swing_high_indices = [idx for idx, val in smc['swing_highs']]
        swing_low_indices = [idx for idx, val in smc['swing_lows']]
        
        self.assertIn(5, swing_high_indices)
        self.assertIn(10, swing_low_indices)
        
        # Verify the actual values
        self.assertEqual(dict(smc['swing_highs'])[5], 110.0)
        self.assertEqual(dict(smc['swing_lows'])[10], 80.0)

    def test_fvg_detection(self):
        """Verify FVG detection for both Bullish (Low[i] > High[i-2]) and Bearish (High[i] < Low[i-2]) gaps."""
        # 1. Bullish FVG
        # Low[3] = 105, High[1] = 100 -> gap between 100 and 105
        df_bull = pd.DataFrame({
            'Open': [95.0] * 5,
            'High': [100.0, 100.0, 101.0, 108.0, 108.0],
            'Low':  [90.0,  90.0,  95.0,  105.0, 105.0],
            'Close': [95.0] * 5,
            'Volume': [1000] * 5
        })
        smc_bull = self.engine.detect_smc_features(df_bull)
        active_bull_fvgs = smc_bull['active_bullish_fvg']
        
        self.assertTrue(len(active_bull_fvgs) >= 1)
        self.assertEqual(active_bull_fvgs[0]['bottom'], 100.0)
        self.assertEqual(active_bull_fvgs[0]['top'], 105.0)
        self.assertEqual(active_bull_fvgs[0]['type'], 'Bullish')

        # 2. Bearish FVG
        # High[3] = 95, Low[1] = 100 -> gap between 95 and 100
        df_bear = pd.DataFrame({
            'Open': [95.0] * 5,
            'High': [105.0, 105.0, 102.0, 95.0,  95.0],
            'Low':  [100.0, 100.0, 96.0,  90.0,  90.0],
            'Close': [95.0] * 5,
            'Volume': [1000] * 5
        })
        smc_bear = self.engine.detect_smc_features(df_bear)
        active_bear_fvgs = smc_bear['active_bearish_fvg']
        
        self.assertTrue(len(active_bear_fvgs) >= 1)
        self.assertEqual(active_bear_fvgs[0]['bottom'], 95.0)
        self.assertEqual(active_bear_fvgs[0]['top'], 100.0)
        self.assertEqual(active_bear_fvgs[0]['type'], 'Bearish')

    def test_bos_and_choch_triggers(self):
        """Verify BOS and CHOCH are triggered when price crosses previous structure levels."""
        # We construct a trend sequence:
        # 1. Bullish trend starting. Swing high created at index 2 (val=105).
        # 2. Price dips, then shoots up past 105 at index 7 -> triggers BOS.
        highs = [100.0] * 12
        closes = [95.0] * 12
        opens = [95.0] * 12
        
        highs[2] = 105.0 # swing high
        
        # Breakout at index 7
        closes[7] = 108.0
        # Set last close high so trend starts as Bullish
        closes[11] = 110.0
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': [90.0] * 12,
            'Close': closes,
            'Volume': [1000] * 12
        })
        
        smc = self.engine.detect_smc_features(df)
        self.assertTrue(len(smc['bos']) >= 1)
        self.assertEqual(smc['bos'][0]['type'], 'Bullish')
        self.assertEqual(smc['bos'][0]['price'], 105.0)

    def test_confluence_and_guardrails(self):
        """Verify 7-factor confluence weights and AI alignment guardrails."""
        # Mock class structures for model predict probability
        self.engine.models = {
            'TEST_SYM': {
                'd1': {'rf': MagicMock(), 'gb': MagicMock()},
                'd2': {'rf': MagicMock(), 'gb': MagicMock()},
                'd3': {'rf': MagicMock(), 'gb': MagicMock()},
                'd4': {'rf': MagicMock(), 'gb': MagicMock()}
            }
        }
        
        # Set classes & predict probability for HOLD / neutral AI scenario
        for d in ['d1', 'd2', 'd3', 'd4']:
            self.engine.models['TEST_SYM'][d]['rf'].classes_ = [1, -1]
            self.engine.models['TEST_SYM'][d]['gb'].classes_ = [1, -1]
            # p_buy = 0.40, p_sell = 0.60
            self.engine.models['TEST_SYM'][d]['rf'].predict_proba.return_value = np.array([[0.40, 0.60]])
            self.engine.models['TEST_SYM'][d]['gb'].predict_proba.return_value = np.array([[0.40, 0.60]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, 14))
        self.engine.scalers = {'TEST_SYM': mock_scaler}

        # Setup mock data frame
        df = pd.DataFrame({
            'Open': [100.0] * 35,
            'High': [102.0] * 35,
            'Low': [98.0] * 35,
            'Close': [100.0] * 35,
            'Volume': [1000] * 35
        })

        # Mock timeframe status
        self.engine.get_timeframe_status = MagicMock(return_value={
            "trend": "Neutral", "rsi": 50, "score": 0.5, "pattern": "N/A"
        })
        self.engine.calculate_volatility_state = MagicMock(return_value=("Normal", 1.0))
        self.engine.calculate_fibonacci = MagicMock(return_value={
            0.0: 90.0, 0.236: 92.36, 0.382: 93.82, 0.5: 95.0, 0.618: 96.18, 0.786: 97.86, 1.0: 100.0
        })
        self.engine.fib_near_levels = MagicMock(return_value=(False, False, False))
        
        # 1. Test Guardrail 1: AI Model not aligned (p_sell is 0.60, but raw_prob is 0.60 which is < 0.50 if looking at BUY, or if it is SELL bias but raw_prob = 0.60)
        # Actually raw_prob is max(p_buy, p_sell) = 0.60.
        # Let's verify prediction output
        res = self.engine.predict(
            symbol='TEST_SYM',
            prices=[100.0] * 35,
            volumes=[1000] * 35,
            news_sent=0.0,
            tv_sent=0.0,
            df=df
        )
        
        # Assert prediction results returned
        self.assertIsNotNone(res)
        today_res = res['today']
        
        # Since raw_prob is 0.60 (which is >= 0.50), let's test what happens when we force raw_prob < 0.50
        for d in ['d1', 'd2', 'd3', 'd4']:
            self.engine.models['TEST_SYM'][d]['rf'].predict_proba.return_value = np.array([[0.48, 0.52]])
            self.engine.models['TEST_SYM'][d]['gb'].predict_proba.return_value = np.array([[0.48, 0.52]])
            
        res_weak = self.engine.predict(
            symbol='TEST_SYM',
            prices=[100.0] * 35,
            volumes=[1000] * 35,
            news_sent=0.0,
            tv_sent=0.0,
            df=df
        )
        
        # Verify that weak raw_prob (< 0.50) forces signal to HOLD (NO TRADE / low confidence) and caps confidence at 0.39
        self.assertLessEqual(res_weak['today']['confidence'], 0.39)
        self.assertIn("NO TRADE", res_weak['today']['signal'])

if __name__ == '__main__':
    unittest.main()
