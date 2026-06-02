"""
Complete AI Stock Prediction System - Example Usage
Demonstrates all components from data collection to visualization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

from preprocess import TechnicalIndicators
from ai_engine import FeatureExtractor, AIPredictor, PredictionAnalyzer, TradeExecution
from visualization import create_advanced_candlestick, detect_support_resistance
from prediction_tracker import save_prediction


def example_complete_pipeline():
    """
    Complete pipeline example:
    1. Fetch data
    2. Extract features
    3. Train model
    4. Make prediction
    5. Visualize results
    6. Track prediction
    """
    
    print("=" * 80)
    print("AI STOCK MARKET PREDICTION SYSTEM - COMPLETE PIPELINE EXAMPLE")
    print("=" * 80)
    
    # ─── STEP 1: FETCH STOCK DATA ─────────────────────────────────────
    print("\n[STEP 1] Fetching stock data...")
    symbol = 'RELIANCE.NS'
    
    df = yf.download(
        symbol,
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        interval='1d',
        progress=False
    )
    
    print(f"✓ Downloaded {len(df)} days of data for {symbol}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Latest close: ₹{df['Close'].iloc[-1]:.2f}")
    print(f"  52-week high: ₹{df['High'].max():.2f}")
    print(f"  52-week low: ₹{df['Low'].min():.2f}")
    
    # ─── STEP 2: EXTRACT TECHNICAL FEATURES ────────────────────────────
    print("\n[STEP 2] Extracting technical features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(df)
    
    print(f"✓ Extracted {len(features.columns)} features")
    print(f"  Features include:")
    print(f"  - Moving Averages (SMA/EMA 5, 10, 20, 50)")
    print(f"  - Momentum (RSI, MACD, Histogram)")
    print(f"  - Volume Analysis (OBV, Volume Ratio)")
    print(f"  - Volatility (ATR, Standard Deviation)")
    print(f"  - Support/Resistance levels")
    print(f"  - Pattern Recognition signals")
    
    # ─── STEP 3: CREATE TARGET VARIABLE ────────────────────────────────
    print("\n[STEP 3] Creating target variable for training...")
    target = feature_extractor.create_target_variable(df, target_period=5, threshold=0.01)
    
    buy_signals = (target == 1).sum()
    sell_signals = (target == 0).sum()
    print(f"✓ Target distribution:")
    print(f"  Buy signals (1): {buy_signals} ({buy_signals/len(target)*100:.1f}%)")
    print(f"  Sell signals (0): {sell_signals} ({sell_signals/len(target)*100:.1f}%)")
    
    # ─── STEP 4: TRAIN AI MODEL ───────────────────────────────────────
    print("\n[STEP 4] Training AI model...")
    predictor = AIPredictor(model_type='ensemble', model_path='ai_model.pkl')
    
    # Use data up to 5 days ago (exclude recent dates for training)
    X_train = features[:-5]
    y_train = target[:-5]
    
    # Remove any NaN values
    valid_indices = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]
    
    stats = predictor.train(X_train, y_train, test_size=0.2)
    
    print(f"✓ Model training complete:")
    print(f"  Training Accuracy: {stats['train_accuracy']:.2%}")
    print(f"  Testing Accuracy: {stats['test_accuracy']:.2%}")
    print(f"  Training samples: {stats['train_samples']}")
    print(f"  Testing samples: {stats['test_samples']}")
    
    # Save model for future use
    predictor.save_model()
    print(f"✓ Model saved to ai_model.pkl")
    
    # ─── STEP 5: MAKE PREDICTION ──────────────────────────────────────
    print("\n[STEP 5] Making AI prediction...")
    latest_features = features.iloc[-1:, :]
    
    # Remove NaN values if any
    latest_features = latest_features.fillna(0)
    
    prediction = predictor.predict(latest_features)
    
    print(f"✓ Prediction result:")
    print(f"  Signal: {prediction['signal']}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
    print(f"  Buy Probability: {prediction['buy_probability']:.2%}")
    print(f"  Sell Probability: {prediction['sell_probability']:.2%}")
    
    # ─── STEP 6: CALCULATE CONFIDENCE SCORES ──────────────────────────
    print("\n[STEP 6] Calculating multi-factor confidence scores...")
    
    # Technical analysis score
    tech_score = PredictionAnalyzer.calculate_technical_score(df, features)
    print(f"  Technical Score: {tech_score:.1f}/100")
    
    # Sentiment score (simulated)
    sentiment_score = PredictionAnalyzer.calculate_sentiment_score(
        headlines_sentiment='positive',
        market_bias='bullish'
    )
    print(f"  Sentiment Score: {sentiment_score:.1f}/100")
    
    # Combined confidence
    final_confidence = PredictionAnalyzer.calculate_combined_confidence(
        ai_confidence=prediction['confidence'],
        technical_score=tech_score,
        sentiment_score=sentiment_score,
        pattern_score=65,
        volume_strength=72
    )
    print(f"  Combined Confidence: {final_confidence:.2%}")
    
    # ─── STEP 7: CALCULATE TRADE EXECUTION LEVELS ─────────────────────
    print("\n[STEP 7] Calculating trade execution levels...")
    
    current_price = df['Close'].iloc[-1]
    atr = TechnicalIndicators.calculate_atr(df, period=14).iloc[-1]
    
    levels = TradeExecution.calculate_levels(
        current_price=current_price,
        atr=atr,
        confidence=final_confidence,
        signal=prediction['signal'],
        risk_percent=2,
        portfolio_value=100000
    )
    
    print(f"✓ Trade execution plan:")
    print(f"  Entry Price: ₹{levels['entry']:.2f}")
    print(f"  Target: ₹{levels['target']:.2f} (+{levels['target_pct']:.2f}%)")
    print(f"  Stop Loss: ₹{levels['sl']:.2f} (-{levels['stop_loss_pct']:.2f}%)")
    print(f"  Position Size: {levels['pos_size']} shares")
    print(f"  Risk Amount: ₹{levels['risk_amt']:.2f}")
    print(f"  Profit at Target: ₹{levels['profit_amt']:.2f}")
    print(f"  Risk/Reward Ratio: {levels['risk_reward']}")
    
    # ─── STEP 8: DETECT SUPPORT & RESISTANCE ──────────────────────────
    print("\n[STEP 8] Detecting support & resistance levels...")
    
    sr_levels = detect_support_resistance(df, window=20)
    
    print(f"✓ Support & Resistance levels:")
    print(f"  Current Price: ₹{sr_levels['current_price']:.2f}")
    print(f"  Pivot Point: ₹{sr_levels['pivot']:.2f}")
    print(f"  Resistance Levels: {[f'₹{r:.0f}' for r in sr_levels['resistance_levels'][-3:]]}")
    print(f"  Support Levels: {[f'₹{s:.0f}' for s in sr_levels['support_levels'][-3:]]}")
    
    # ─── STEP 9: CREATE VISUALIZATION ────────────────────────────────
    print("\n[STEP 9] Creating advanced candlestick chart...")
    
    try:
        fig = create_advanced_candlestick(
            df=df.tail(100),  # Last 100 days
            symbol=symbol,
            title=f"AI Prediction: {prediction['signal']} (Confidence: {final_confidence:.1%})",
            prediction=levels,
            show_rsi=True,
            show_macd=True,
            show_volume=True
        )
        
        # Save chart
        fig.write_html("candlestick_chart.html")
        print(f"✓ Chart saved to candlestick_chart.html")
        
    except Exception as e:
        print(f"⚠ Error creating chart: {e}")
    
    # ─── STEP 10: SAVE PREDICTION ────────────────────────────────────
    print("\n[STEP 10] Saving prediction to history...")
    
    prediction_data = {
        'symbol': symbol,
        'signal': prediction['signal'],
        'confidence': final_confidence,
        'price': current_price,
        'entry': levels['entry'],
        'target': levels['target'],
        'sl': levels['sl'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'catalyst': 'Technical Analysis - AI Ensemble Model',
        'technical_score': tech_score,
        'sentiment_score': sentiment_score,
        'atr': atr,
        'position_size': levels['pos_size']
    }
    
    save_prediction(prediction_data)
    print(f"✓ Prediction saved to prediction_history.json")
    
    # ─── SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY & TRADING RECOMMENDATION")
    print("=" * 80)
    
    signal_emoji = "📈" if prediction['signal'] == 'BUY' else "📉"
    confidence_emoji = "🟢" if final_confidence >= 0.7 else "🟡" if final_confidence >= 0.5 else "🔴"
    
    print(f"\n{signal_emoji} Signal: {prediction['signal']}")
    print(f"{confidence_emoji} Confidence: {final_confidence:.1%}")
    print(f"\nExecution Plan:")
    print(f"  • BUY at: ₹{levels['entry']:.2f}")
    print(f"  • TARGET: ₹{levels['target']:.2f}")
    print(f"  • STOP LOSS: ₹{levels['sl']:.2f}")
    print(f"  • Qty: {levels['pos_size']} shares")
    print(f"\nRisk Management:")
    print(f"  • Max Loss: ₹{levels['risk_amt']:.2f}")
    print(f"  • Profit Target: ₹{levels['profit_amt']:.2f}")
    print(f"  • Risk/Reward: {levels['risk_reward']}")
    
    if final_confidence >= 0.75:
        print(f"\n✅ STRONG {prediction['signal']} signal - High confidence")
    elif final_confidence >= 0.65:
        print(f"\n⚠️  MODERATE {prediction['signal']} signal - Consider your risk tolerance")
    else:
        print(f"\n❌ WEAK signal - Wait for better setup")
    
    print("\n" + "=" * 80)
    print("End of pipeline execution")
    print("=" * 80 + "\n")
    
    return {
        'df': df,
        'features': features,
        'prediction': prediction,
        'levels': levels,
        'confidence': final_confidence,
        'sr_levels': sr_levels,
        'predictor': predictor
    }


def example_multi_stock_analysis():
    """
    Example: Analyze multiple stocks simultaneously
    """
    
    print("\n" + "=" * 80)
    print("MULTI-STOCK ANALYSIS EXAMPLE")
    print("=" * 80 + "\n")
    
    stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HDFCBANK.NS']
    predictor = AIPredictor(model_type='ensemble', model_path='ai_model.pkl')
    
    results = []
    
    for stock in stocks:
        print(f"\nAnalyzing {stock}...", end=" ")
        
        try:
            # Fetch data
            df = yf.download(stock, period='1y', interval='1d', progress=False)
            
            # Extract features
            fe = FeatureExtractor()
            features = fe.extract_features(df)
            latest_features = features.iloc[-1:, :].fillna(0)
            
            # Predict
            pred = predictor.predict(latest_features)
            
            # Calculate levels
            current_price = df['Close'].iloc[-1]
            atr = TechnicalIndicators.calculate_atr(df, 14).iloc[-1]
            levels = TradeExecution.calculate_levels(current_price, atr, pred['confidence'], pred['signal'])
            
            # Technical score
            tech_score = PredictionAnalyzer.calculate_technical_score(df, features)
            
            results.append({
                'Symbol': stock,
                'Signal': pred['signal'],
                'Confidence': f"{pred['confidence']:.1%}",
                'Price': f"₹{current_price:.2f}",
                'Target': f"₹{levels['target']:.2f}",
                'SL': f"₹{levels['sl']:.2f}",
                'Tech_Score': f"{tech_score:.0f}/100"
            })
            
            print("✓")
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Display results
    print("\n" + "-" * 80)
    print("ANALYSIS RESULTS")
    print("-" * 80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


def example_backtest():
    """
    Example: Simple backtest of predictions
    """
    
    print("\n" + "=" * 80)
    print("BACKTEST EXAMPLE")
    print("=" * 80 + "\n")
    
    # Fetch data
    df = yf.download('RELIANCE.NS', period='2y', interval='1d', progress=False)
    
    # Extract features
    fe = FeatureExtractor()
    features = fe.extract_features(df)
    
    # Load trained model
    predictor = AIPredictor(model_type='ensemble', model_path='ai_model.pkl')
    
    # Backtest
    trades = []
    
    for i in range(200, len(df) - 5):
        # Get features and predict
        current_features = features.iloc[i:i+1, :].fillna(0)
        prediction = predictor.predict(current_features)
        
        # Simulate trade
        entry_price = df['Close'].iloc[i]
        exit_price = df['Close'].iloc[i+5]
        
        if prediction['signal'] == 'BUY':
            profit = (exit_price - entry_price) / entry_price * 100
            correct = profit > 0
        else:
            profit = (entry_price - exit_price) / entry_price * 100
            correct = profit > 0
        
        trades.append({
            'Date': df.index[i],
            'Signal': prediction['signal'],
            'Confidence': prediction['confidence'],
            'Entry': entry_price,
            'Exit': exit_price,
            'Profit%': profit,
            'Correct': correct
        })
    
    trades_df = pd.DataFrame(trades)
    
    # Statistics
    total_trades = len(trades_df)
    winning_trades = (trades_df['Correct']).sum()
    win_rate = winning_trades / total_trades * 100
    avg_profit = trades_df['Profit%'].mean()
    
    print(f"Backtest Results (Last {total_trades} signals):")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Average Profit/Loss: {avg_profit:.2f}%")
    print(f"  Total Return: {trades_df['Profit%'].sum():.2f}%")
    
    return trades_df


if __name__ == "__main__":
    # Run complete pipeline example
    pipeline_result = example_complete_pipeline()
    
    # Uncomment to run other examples
    # multi_results = example_multi_stock_analysis()
    # backtest_results = example_backtest()
