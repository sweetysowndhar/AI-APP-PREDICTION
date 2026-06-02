"""
INTEGRATION GUIDE: New Modules with Existing Streamlit App

This file shows how to integrate the new visualization and AI modules
into your existing ai_prediction_app.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# FILE: ai_prediction_app.py (Updated Integration)
# ═══════════════════════════════════════════════════════════════════════════

"""
ADD THESE IMPORTS TO THE TOP OF ai_prediction_app.py:
"""

# ─── NEW IMPORTS (Add to existing imports) ─────────────────────────────────
from visualization import (
    create_advanced_candlestick,
    detect_support_resistance,
    create_prediction_comparison_chart,
    create_correlation_heatmap
)
from ai_engine import (
    FeatureExtractor,
    AIPredictor,
    PredictionAnalyzer,
    TradeExecution
)

# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTION 1: Enhanced Charting
# ═══════════════════════════════════════════════════════════════════════════

def render_enhanced_chart(stock_data, symbol, prediction_result=None):
    """
    Replace existing chart rendering with enhanced candlestick
    
    USAGE IN ai_prediction_app.py:
    -----
    In your chart rendering section, replace existing plotly code with:
    """
    
    # Calculate trade levels if prediction exists
    prediction_levels = None
    if prediction_result:
        from preprocess import TechnicalIndicators
        
        current_price = stock_data['Close'].iloc[-1]
        atr = TechnicalIndicators.calculate_atr(stock_data, 14).iloc[-1]
        
        prediction_levels = TradeExecution.calculate_levels(
            current_price=current_price,
            atr=atr,
            confidence=prediction_result.get('confidence', 0.5),
            signal=prediction_result['signal'],
            risk_percent=2
        )
    
    # Create advanced chart
    fig = create_advanced_candlestick(
        df=stock_data.tail(100),
        symbol=symbol,
        title=f"Technical Analysis - {symbol}",
        prediction=prediction_levels,
        show_rsi=True,
        show_macd=True,
        show_volume=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return prediction_levels


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTION 2: AI-Powered Predictions
# ═══════════════════════════════════════════════════════════════════════════

def get_ai_prediction(stock_data):
    """
    Get AI prediction with confidence scores
    
    USAGE IN ai_prediction_app.py:
    -----
    In your prediction section, use this instead of existing logic:
    """
    
    # Extract features
    fe = FeatureExtractor()
    features = fe.extract_features(stock_data)
    
    # Load or train model
    predictor = AIPredictor(model_type='ensemble', model_path='ai_model.pkl')
    
    if predictor.model is None:
        # If no trained model, use a default prediction
        st.warning("Model not trained. Train first using example_usage.py")
        return None
    
    # Get prediction
    latest_features = features.iloc[-1:, :].fillna(0)
    prediction = predictor.predict(latest_features)
    
    # Calculate multi-factor confidence
    tech_score = PredictionAnalyzer.calculate_technical_score(stock_data, features)
    sentiment_score = PredictionAnalyzer.calculate_sentiment_score()
    
    final_confidence = PredictionAnalyzer.calculate_combined_confidence(
        ai_confidence=prediction['confidence'],
        technical_score=tech_score,
        sentiment_score=sentiment_score
    )
    
    return {
        'signal': prediction['signal'],
        'confidence': final_confidence,
        'buy_probability': prediction['buy_probability'],
        'sell_probability': prediction['sell_probability'],
        'technical_score': tech_score,
        'sentiment_score': sentiment_score
    }


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTION 3: Support & Resistance Analysis
# ═══════════════════════════════════════════════════════════════════════════

def display_support_resistance(stock_data, symbol):
    """
    Display detected support and resistance levels
    
    USAGE IN ai_prediction_app.py:
    -----
    Add this section to your technical analysis tab:
    """
    
    sr = detect_support_resistance(stock_data, window=20)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"₹{sr['current_price']:.2f}")
    
    with col2:
        st.metric("Pivot Point", f"₹{sr['pivot']:.2f}")
    
    with col3:
        st.metric("Distance to S/R", f"{abs(sr['current_price'] - sr['pivot']):.2f}")
    
    # Display levels
    st.subheader("📊 Support & Resistance Levels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Resistance Levels (↑)**")
        for i, level in enumerate(sr['resistance_levels'][-3:], 1):
            st.write(f"  R{i}: ₹{level:.2f}")
    
    with col2:
        st.write("**Support Levels (↓)**")
        for i, level in enumerate(reversed(sr['support_levels'][-3:]), 1):
            st.write(f"  S{i}: ₹{level:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTION 4: Trade Execution Card
# ═══════════════════════════════════════════════════════════════════════════

def render_execution_card(prediction_levels, confidence):
    """
    Display trade execution details with professional styling
    
    USAGE IN ai_prediction_app.py:
    -----
    After rendering chart, display execution plan:
    """
    
    if not prediction_levels:
        return
    
    signal = "BUY" if prediction_levels['entry'] > prediction_levels['sl'] else "SELL"
    signal_color = "#10b981" if signal == "BUY" else "#ef4444"
    
    st.markdown(f'''
    <div style="background: {signal_color}10; border: 2px solid {signal_color}; 
                padding: 20px; border-radius: 15px; margin-top: 20px;">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px;">
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;">SIGNAL</div>
                <div style="font-size: 1.5rem; color: {signal_color}; font-weight: 800;">{signal}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;">CONFIDENCE</div>
                <div style="font-size: 1.5rem; color: {signal_color}; font-weight: 800;">{confidence:.1%}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;">POSITION SIZE</div>
                <div style="font-size: 1.5rem; color: #38bdf8; font-weight: 800;">{prediction_levels['pos_size']} Qty</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;">RISK/REWARD</div>
                <div style="font-size: 1.5rem; color: #fcd34d; font-weight: 800;">{prediction_levels['risk_reward']}</div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid {signal_color};">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; font-size: 0.9rem;">
                <div>
                    <span style="color: #94a3b8;">Entry Price</span><br>
                    <span style="color: #f8fafc; font-weight: 800; font-size: 1.1rem;">₹{prediction_levels['entry']:.2f}</span>
                </div>
                <div>
                    <span style="color: #10b981;">Target</span><br>
                    <span style="color: #f8fafc; font-weight: 800; font-size: 1.1rem;">₹{prediction_levels['target']:.2f}</span>
                </div>
                <div>
                    <span style="color: #ef4444;">Stop Loss</span><br>
                    <span style="color: #f8fafc; font-weight: 800; font-size: 1.1rem;">₹{prediction_levels['sl']:.2f}</span>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); 
                    border-radius: 10px; font-size: 0.85rem; color: #94a3b8;">
            <strong>Max Loss:</strong> ₹{prediction_levels['risk_amt']:.2f} | 
            <strong>Profit Target:</strong> ₹{prediction_levels['profit_amt']:.2f}
        </div>
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTION 5: Performance Analytics
# ═══════════════════════════════════════════════════════════════════════════

def display_prediction_analytics():
    """
    Show prediction performance and accuracy metrics
    
    USAGE IN ai_prediction_app.py:
    -----
    Add this to your analytics section:
    """
    
    from prediction_tracker import load_history
    
    history = load_history()
    
    if not history:
        st.info("No prediction history available")
        return
    
    # Calculate statistics
    total_predictions = len(history)
    
    # Determine correct/incorrect (if actual_result is tracked)
    predictions_with_outcome = [p for p in history if 'actual_result' in p]
    
    if predictions_with_outcome:
        correct = sum(1 for p in predictions_with_outcome if p.get('actual_result'))
        win_rate = correct / len(predictions_with_outcome) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            st.metric("Correct Predictions", correct)
        
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%", 
                     delta=f"{win_rate - 50:.1f}%" if win_rate > 50 else None)
    
    # Recent predictions table
    st.subheader("📋 Recent Predictions")
    
    recent = pd.DataFrame(history[:10])
    
    display_cols = ['symbol', 'signal', 'confidence', 'price', 'timestamp']
    if all(col in recent.columns for col in display_cols):
        st.dataframe(
            recent[display_cols].rename(columns={
                'symbol': 'Stock',
                'signal': 'Signal',
                'confidence': 'Confidence',
                'price': 'Price',
                'timestamp': 'Time'
            }),
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE: Complete Page Integration
# ═══════════════════════════════════════════════════════════════════════════

def example_complete_page(symbol='RELIANCE.NS'):
    """
    Complete page example using all new functions together
    
    USAGE IN ai_prediction_app.py:
    -----
    Add this as a new page or tab:
    """
    
    import yfinance as yf
    
    st.set_page_config(page_title="AI Stock Predictor", layout="wide")
    st.title("🤖 AI Stock Market Prediction")
    
    # Fetch data
    with st.spinner("Fetching stock data..."):
        stock_data = yf.download(symbol, period='1y', interval='1d', progress=False)
    
    # Get AI prediction
    with st.spinner("Running AI analysis..."):
        prediction = get_ai_prediction(stock_data)
    
    if prediction:
        # Display enhanced chart
        st.subheader("📈 Technical Analysis Chart")
        levels = render_enhanced_chart(stock_data, symbol, prediction)
        
        # Display execution card
        if levels:
            render_execution_card(levels, prediction['confidence'])
        
        # Display support & resistance
        st.markdown("---")
        display_support_resistance(stock_data, symbol)
        
        # Display analytics
        st.markdown("---")
        display_prediction_analytics()
        
        # Display prediction details
        st.markdown("---")
        st.subheader("🔍 Prediction Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AI Signal", prediction['signal'],
                     delta=f"{prediction['confidence']:.1%}")
        
        with col2:
            st.metric("Technical Score", f"{prediction['technical_score']:.0f}/100")
        
        with col3:
            st.metric("Sentiment Score", f"{prediction['sentiment_score']:.0f}/100")
        
        with col4:
            buy_prob = prediction['buy_probability']
            st.metric("Buy Probability", f"{buy_prob:.1%}")
    
    else:
        st.error("Failed to get prediction. Train the model first.")


# ═══════════════════════════════════════════════════════════════════════════
# QUICK START CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════

"""
INTEGRATION CHECKLIST:

1. ✅ Copy visualization.py to your trading folder
2. ✅ Copy ai_engine.py to your trading folder
3. ✅ Update requirements.txt (already has all dependencies)
4. ✅ Add new imports to ai_prediction_app.py (see above)
5. ✅ Run example_usage.py to train the AI model:
   
   python example_usage.py
   
   This creates: ai_model.pkl (your trained model)

6. ✅ Add integration functions to ai_prediction_app.py
7. ✅ Use render_enhanced_chart() instead of existing chart rendering
8. ✅ Use get_ai_prediction() in your prediction section
9. ✅ Use render_execution_card() to display trade details
10. ✅ Test with: streamlit run ai_prediction_app.py

QUICK TEST:
-----------
python example_usage.py  # Train model
streamlit run ai_prediction_app.py  # Run app

Then update your existing pages to use the new functions.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CODE SNIPPET: Update Existing Prediction Page
# ═══════════════════════════════════════════════════════════════════════════

"""
EXAMPLE: Update your existing page_analysis() function

BEFORE (existing code):
-----------------------
def page_analysis():
    symbol = st.sidebar.selectbox("Select Stock", list(STOCK_MAP.keys()))
    df = fetch_stock(symbol)
    
    # Create basic plotly chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(...))
    st.plotly_chart(fig)

AFTER (with new modules):
------------------------
def page_analysis():
    symbol = st.sidebar.selectbox("Select Stock", list(STOCK_MAP.keys()))
    df = fetch_stock(symbol)
    
    # Get AI prediction
    pred = get_ai_prediction(df)
    
    # Create enhanced chart with prediction levels
    levels = render_enhanced_chart(df, symbol, pred)
    
    # Display execution details
    if pred and levels:
        render_execution_card(levels, pred['confidence'])
    
    # Show support & resistance
    display_support_resistance(df, symbol)
"""

# ═══════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════

"""
COMMON ISSUES & SOLUTIONS:

Issue: "ModuleNotFoundError: No module named 'visualization'"
Solution: Make sure visualization.py is in the same directory as ai_prediction_app.py

Issue: "Model not trained" warning appears
Solution: Run example_usage.py first to create ai_model.pkl

Issue: Chart not displaying
Solution: 
  - Update Plotly: pip install --upgrade plotly
  - Check data has proper datetime index
  - Verify OHLCV columns exist

Issue: Predictions seem inaccurate
Solution:
  - Retrain model with recent data: python example_usage.py
  - Check if market conditions have changed
  - Verify features are being calculated correctly

Issue: High memory usage
Solution:
  - Reduce data period (use 1y instead of 5y initially)
  - Use caching: st.cache_data
  - Close old streamlit servers
"""
