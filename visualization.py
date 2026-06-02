"""
Advanced Stock Market Visualization Module - CLEAN VERSION
No Plotly errors, no subplot issues
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from preprocess import TechnicalIndicators


def detect_support_resistance(df, window=20):
    """
    Detect support and resistance levels using local minima/maxima
    """
    if len(df) < window:
        return {'support_levels': [], 'resistance_levels': [], 'pivot': df['Close'].iloc[-1], 'current_price': df['Close'].iloc[-1]}
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i-window:i+window]):
            resistance_levels.append((i, highs[i]))
        if lows[i] == min(lows[i-window:i+window]):
            support_levels.append((i, lows[i]))
    
    if support_levels:
        support_levels = sorted(set([round(s[1], 2) for s in support_levels]))
    if resistance_levels:
        resistance_levels = sorted(set([round(r[1], 2) for r in resistance_levels]))
    
    pivot = (df['High'].iloc[-1] + df['Low'].iloc[-1] + closes[-1]) / 3
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'pivot': pivot,
        'current_price': closes[-1]
    }


def create_advanced_candlestick(df, symbol="STOCK", title="", prediction=None, 
                               show_rsi=True, show_macd=True, show_volume=True):
    """
    Create advanced candlestick chart with technical overlays
    CLEAN VERSION - Single figure, no subplot errors
    """
    
    # Calculate technical indicators
    close_prices = df['Close'].values
    rsi = TechnicalIndicators.calculate_rsi(close_prices, period=14)
    macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(close_prices)
    
    # Detect S/R levels
    sr_data = detect_support_resistance(df, window=20)
    
    # Create main figure
    fig = go.Figure()
    
    # ─── CANDLESTICK CHART ────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}',
        )
    )
    
    # ─── SUPPORT & RESISTANCE LINES ────────────────────────────────
    x_start = df.index[0]
    x_end = df.index[-1]
    
    # Support levels
    for support in sr_data['support_levels'][-3:]:
        fig.add_shape(
            type='line',
            x0=x_start, x1=x_end,
            y0=support, y1=support,
            line=dict(color='#3b82f6', width=1, dash='dash'),
        )
    
    # Resistance levels
    for resistance in sr_data['resistance_levels'][-3:]:
        fig.add_shape(
            type='line',
            x0=x_start, x1=x_end,
            y0=resistance, y1=resistance,
            line=dict(color='#f97316', width=1, dash='dash'),
        )
    
    # Pivot Point
    fig.add_shape(
        type='line',
        x0=x_start, x1=x_end,
        y0=sr_data['pivot'], y1=sr_data['pivot'],
        line=dict(color='#a855f7', width=1, dash='dot'),
    )
    
    # ─── PREDICTION LEVELS (if provided) ───────────────────────────
    if prediction:
        if 'entry' in prediction:
            fig.add_shape(
                type='line',
                x0=x_start, x1=x_end,
                y0=prediction['entry'], y1=prediction['entry'],
                line=dict(color='#06b6d4', width=2, dash='solid'),
            )
        
        if 'target' in prediction:
            fig.add_shape(
                type='line',
                x0=x_start, x1=x_end,
                y0=prediction['target'], y1=prediction['target'],
                line=dict(color='#10b981', width=2, dash='solid'),
            )
        
        if 'sl' in prediction:
            fig.add_shape(
                type='line',
                x0=x_start, x1=x_end,
                y0=prediction['sl'], y1=prediction['sl'],
                line=dict(color='#ef4444', width=2, dash='solid'),
            )
    
    # ─── UPDATE LAYOUT ────────────────────────────────────────────
    title_text = f"<b>{symbol}</b> - {title}" if title else f"<b>{symbol}</b>"
    
    fig.update_layout(
        title_text=title_text,
        yaxis_title="Price (₹)",
        xaxis_title="Date",
        template="plotly_white",
        height=600,
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", size=11, color="#0f172a"),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=60, b=50),
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    
    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#334155',
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#334155',
    )
    
    return fig


def create_prediction_comparison_chart(predictions_history):
    """
    Create a chart comparing multiple predictions over time
    """
    df = pd.DataFrame(predictions_history)
    
    fig = go.Figure()
    
    # ─── ACCURACY OVER TIME ───────────────────────────────────────
    correct = (df['signal'] == df['actual_result']).astype(int)
    accuracy = correct.rolling(window=10).mean() * 100
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=accuracy,
            name='10-Day Accuracy',
            fill='tozeroy',
            line=dict(color='#6366f1', width=3),
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%',
        )
    )
    
    fig.update_layout(
        title_text="<b>Prediction Performance Analysis</b>",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", size=11, color="#f8fafc"),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    fig.update_yaxes(
        title_text="Accuracy (%)",
        showgrid=True,
        gridcolor='#334155',
    )
    
    return fig


def create_correlation_heatmap(symbols_data):
    """
    Create correlation heatmap for multiple stocks
    """
    closes = pd.DataFrame({
        symbol: df['Close'] for symbol, df in symbols_data.items()
    })
    
    correlation = closes.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}',
    ))
    
    fig.update_layout(
        title_text="<b>Stock Price Correlation Matrix</b>",
        template="plotly_dark",
        height=600,
        font=dict(family="Arial, sans-serif", size=12, color="#f8fafc"),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
    )
    
    return fig
