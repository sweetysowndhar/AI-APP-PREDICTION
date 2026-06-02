# AI-Based Stock Market Prediction System - Complete Guide

## 🏗️ System Architecture

```
PIPELINE FLOW:
┌──────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION LAYER                      │
│        Yahoo Finance API / Alpha Vantage API Integration         │
│           Fetch OHLCV data + Volume information                  │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│              2. DATA PREPROCESSING & CLEANING                    │
│         (preprocess.py - TechnicalIndicators class)              │
│  • Normalize price data                                           │
│  • Calculate moving averages (EMA, SMA)                          │
│  • Remove outliers and missing values                            │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│         3. TECHNICAL FEATURE EXTRACTION LAYER                    │
│          (ai_engine.py - FeatureExtractor class)                 │
│  • RSI, MACD, ATR calculations                                   │
│  • Support & Resistance detection                                │
│  • Volume analysis                                               │
│  • Pattern recognition                                           │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│            4. AI PREDICTION ENGINE                               │
│      (ai_engine.py - AIPredictor class)                          │
│  • Ensemble model (RandomForest + GradientBoosting + Logistic)  │
│  • Binary classification: BUY (1) vs SELL (0)                   │
│  • Returns prediction confidence (0-1)                           │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│         5. CONFIDENCE CALCULATION & ANALYSIS                     │
│   (ai_engine.py - PredictionAnalyzer class)                      │
│  • Technical Score (0-100)                                       │
│  • Sentiment Score (0-100)                                       │
│  • Combined Confidence (weighted average)                        │
│  • Signal strength assessment                                    │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│        6. TRADE EXECUTION CALCULATOR                             │
│   (ai_engine.py - TradeExecution class)                          │
│  • Entry price calculation                                       │
│  • Stop Loss placement                                           │
│  • Target price calculation                                      │
│  • Position sizing                                               │
│  • Risk/Reward ratio                                             │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│          7. VISUALIZATION LAYER                                  │
│      (visualization.py - Advanced Charting)                      │
│  • Candlestick charts with OHLC data                            │
│  • Support/Resistance levels overlay                             │
│  • Technical indicators (RSI, MACD, Volume)                      │
│  • AI prediction levels (Entry, Target, SL)                      │
│  • Performance analytics                                         │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│          8. STREAMLIT UI & DEPLOYMENT                            │
│    (ai_prediction_app.py - Interactive Dashboard)                │
│  • Real-time predictions                                         │
│  • Trade history tracking                                        │
│  • News sentiment analysis                                       │
│  • Performance metrics                                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📦 Key Modules

### 1. **preprocess.py** - Technical Indicators
```python
from preprocess import TechnicalIndicators

# Calculate technical indicators
rsi = TechnicalIndicators.calculate_rsi(prices, period=14)
macd, signal, histogram = TechnicalIndicators.calculate_macd(prices)
ema = TechnicalIndicators.calculate_ema(prices, period=9)
atr = TechnicalIndicators.calculate_atr(df, period=14)
```

### 2. **ai_engine.py** - AI Model & Analysis
```python
from ai_engine import FeatureExtractor, AIPredictor, PredictionAnalyzer, TradeExecution

# Extract features
features = FeatureExtractor.extract_features(df)

# Train model
predictor = AIPredictor(model_type='ensemble')
predictor.train(X_features, y_targets)

# Make prediction
result = predictor.predict(latest_features)
# Returns: {'signal': 'BUY', 'confidence': 0.78, 'buy_probability': 0.78, ...}

# Calculate confidence
confidence = PredictionAnalyzer.calculate_combined_confidence(
    ai_confidence=0.78,
    technical_score=75,
    sentiment_score=68
)

# Execute trade levels
levels = TradeExecution.calculate_levels(
    current_price=1250,
    atr=25,
    confidence=0.78,
    signal='BUY'
)
# Returns: {'entry': 1250, 'target': 1300, 'sl': 1220, 'pos_size': 10, ...}
```

### 3. **visualization.py** - Advanced Charts
```python
from visualization import (
    create_advanced_candlestick,
    detect_support_resistance,
    create_prediction_comparison_chart
)

# Create candlestick chart with all overlays
fig = create_advanced_candlestick(
    df=stock_data,
    symbol='RELIANCE',
    title='Technical Analysis',
    prediction={'entry': 2500, 'target': 2600, 'sl': 2450},
    show_rsi=True,
    show_macd=True,
    show_volume=True
)

# Detect support & resistance
sr = detect_support_resistance(df, window=20)
# Returns: {'support_levels': [...], 'resistance_levels': [...], 'pivot': X}
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
import yfinance as yf
import pandas as pd
from ai_engine import FeatureExtractor, AIPredictor, TradeExecution
from preprocess import TechnicalIndicators
from visualization import create_advanced_candlestick

# 1. Fetch data
df = yf.download('RELIANCE.NS', period='6mo', interval='1d')

# 2. Extract features
fe = FeatureExtractor()
features = fe.extract_features(df)

# 3. Load or train model
predictor = AIPredictor(model_type='ensemble', model_path='my_model.pkl')
if not predictor.model:  # If model not trained yet
    target = fe.create_target_variable(df, target_period=5)
    predictor.train(features[:-5], target[:-5])
    predictor.save_model()

# 4. Make prediction
latest_features = features.iloc[-1:, :]
prediction = predictor.predict(latest_features)
print(f"Signal: {prediction['signal']}")
print(f"Confidence: {prediction['confidence']:.2%}")

# 5. Calculate trade levels
levels = TradeExecution.calculate_levels(
    current_price=df['Close'].iloc[-1],
    atr=TechnicalIndicators.calculate_atr(df, 14).iloc[-1],
    confidence=prediction['confidence'],
    signal=prediction['signal']
)

# 6. Visualize
fig = create_advanced_candlestick(
    df=df.tail(100),
    symbol='RELIANCE',
    prediction=levels,
    show_rsi=True,
    show_macd=True,
    show_volume=True
)
fig.show()
```

---

## 📊 Output Format

### Prediction Result
```python
{
    'signal': 'BUY',                    # Buy or Sell
    'confidence': 0.78,                # 0-1 confidence level
    'buy_probability': 0.78,           # ML model's buy probability
    'sell_probability': 0.22,          # ML model's sell probability
    'all_predictions': array([...]),   # Historical predictions
    'all_probabilities': array([...])  # Historical probabilities
}
```

### Trade Levels
```python
{
    'entry': 1250.50,                  # Entry price
    'target': 1315.25,                 # Target profit level
    'sl': 1220.00,                     # Stop loss level
    'pos_size': 10,                    # Number of shares to buy
    'risk_amt': 305.00,                # Maximum loss in rupees
    'profit_amt': 645.00,              # Profit at target
    'risk_reward': '1:2.1',            # Risk/Reward ratio
    'stop_loss_pct': 2.43,             # SL distance %
    'target_pct': 5.20                 # Target distance %
}
```

### Support & Resistance
```python
{
    'support_levels': [1200, 1180, 1150],
    'resistance_levels': [1300, 1350, 1400],
    'pivot': 1280,
    'current_price': 1250
}
```

---

## 🎯 Model Features (70+ Total)

### Price Features
- Returns, Log Returns, Volatility
- High-Low Ratio, Close-Open Ratio

### Moving Averages & Crossovers
- SMA/EMA (5, 10, 20, 50 periods)
- SMA Crossover Signals

### Momentum Indicators
- RSI (7, 14 periods)
- MACD, Signal Line, Histogram
- MACD Crossover

### Volume Analysis
- Volume Moving Average
- Volume Ratio, OBV (On Balance Volume)

### Volatility
- ATR (Average True Range)
- ATR as % of price

### Support & Resistance
- Distance to Support
- Distance to Resistance

### Pattern Recognition
- Consecutive Green/Red candles
- Price position vs MA200

### Rate of Change
- ROC (5, 10, 20 periods)

---

## 🔧 Configuration & Tuning

### Adjust Risk/Reward
```python
# In TradeExecution.calculate_levels()
risk_reward_ratio = 3 + (confidence * 2)  # Range: 3-5x
# Increase multiplier for aggressive trading
```

### Model Selection
```python
# Choose model type
predictor = AIPredictor(model_type='ensemble')  # Best balanced
# OR
predictor = AIPredictor(model_type='gb')        # Fast, accurate
# OR
predictor = AIPredictor(model_type='rf')        # Robust
```

### Feature Selection
Modify `FeatureExtractor.extract_features()` to include/exclude features based on:
- Market conditions
- Stock volatility
- Sector characteristics

---

## 📈 Performance Tracking

### Track Predictions
```python
from prediction_tracker import save_prediction, load_history

# Save prediction
save_prediction({
    'symbol': 'RELIANCE',
    'signal': 'BUY',
    'confidence': 0.78,
    'price': 1250,
    'entry': 1250,
    'target': 1315,
    'sl': 1220,
    'timestamp': datetime.now()
})

# Load history
history = load_history()
# Analyze accuracy, win rate, etc.
```

### Analyze Performance
```python
from visualization import create_prediction_comparison_chart

# View accuracy over time
fig = create_prediction_comparison_chart(prediction_history)
```

---

## ⚠️ Risk Management

### Position Sizing
- Based on portfolio size and risk tolerance
- Calculated from ATR and stop loss distance

### Stop Loss Placement
```python
stop_loss = entry - (2 * ATR)  # For BUY
stop_loss = entry + (2 * ATR)  # For SELL
```

### Risk/Reward Ratio
- Minimum 1:2 ratio recommended
- Higher confidence → Higher R/R ratio

### Daily Loss Limit
- Set max drawdown per day
- Stop trading if reached

---

## 🌐 Integration with Streamlit App

The complete system is integrated into `ai_prediction_app.py`:

```bash
streamlit run ai_prediction_app.py
```

Features:
- Real-time stock predictions
- Interactive candlestick charts
- Trade tracking dashboard
- News sentiment analysis
- Market calendar integration
- Prediction history & analytics

---

## 📋 Dataset & Training

### Minimum Data Requirements
- 1 year of historical data (250 trading days)
- OHLCV (Open, High, Low, Close, Volume) data
- 5+ years recommended for robust models

### Model Training
```python
# Download data
df = yf.download('RELIANCE.NS', period='5y', interval='1d')

# Create features & targets
features = FeatureExtractor.extract_features(df)
target = FeatureExtractor.create_target_variable(df, target_period=5)

# Train model
predictor = AIPredictor(model_type='ensemble')
stats = predictor.train(features[:-5], target[:-5], test_size=0.2)

# Evaluate
print(f"Train Accuracy: {stats['train_accuracy']:.2%}")
print(f"Test Accuracy: {stats['test_accuracy']:.2%}")
```

---

## 🔐 Important Notes

1. **Backtesting Required**: Always backtest strategies before live trading
2. **Market Conditions Change**: Retrain models periodically (monthly/quarterly)
3. **Diversification**: Use multiple models and indicators
4. **Risk Management**: Never risk more than 2-5% per trade
5. **News Events**: Major events can override technical signals
6. **Liquidity Check**: Ensure stocks have sufficient volume

---

## 📞 Support & Troubleshooting

### Model Not Improving
- Add more diverse data
- Engineer new features
- Adjust hyperparameters
- Check for overfitting

### Predictions Are Inconsistent
- Validate feature extraction
- Check data quality
- Verify market conditions
- Retrain model with fresh data

### Visualization Issues
- Update Plotly: `pip install --upgrade plotly`
- Check data format (datetime index)
- Verify OHLCV columns exist

---

## 🚀 Future Enhancements

- [ ] LSTM/Neural Network models
- [ ] Sentiment analysis from news/social media
- [ ] Options strategy integration
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization
- [ ] Real-time WebSocket data
- [ ] Options Greeks calculation
- [ ] Statistical arbitrage signals

---

## 📚 References

- Technical Analysis: [TradingView Indicators](https://www.tradingview.com/pine-script-docs/)
- ML Models: [Scikit-learn Documentation](https://scikit-learn.org/)
- Visualization: [Plotly Documentation](https://plotly.com/python/)
- Stock Data: [yfinance Documentation](https://yfinance.readthedocs.io/)

---

**Last Updated**: May 2026
**Version**: 3.0
**Status**: Production Ready ✅
