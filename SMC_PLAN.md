# SMC Institutional Order Block & Confluence Scanner Plan

This document details the Smart Money Concepts (SMC) scanner and 7-Factor Confluence framework implemented in **AI Market Predictor Pro**.

---

## 🏛️ Smart Money Concepts (SMC) Engine

The scanner identifies institutional liquidity levels, trend alignment, and order blocks directly from price action:

### 1. Swing Highs & Swing Lows (Market Structure)
- **Swing High**: Identified when a bar's High is higher than the High of the 2 preceding and 2 succeeding bars:
  $$\text{High}[i] > \max(\text{High}[i-1], \text{High}[i-2], \text{High}[i+1], \text{High}[i+2])$$
- **Swing Low**: Identified when a bar's Low is lower than the Low of the 2 preceding and 2 succeeding bars:
  $$\text{Low}[i] < \min(\text{Low}[i-1], \text{Low}[i-2], \text{Low}[i+1], \text{Low}[i+2])$$
*Note: Due to the 2-bar lookahead requirement, swing points are confirmed on candle $i+2$.*

### 2. Break of Structure (BOS) & Change of Character (CHOCH)
- **BOS (Break of Structure)**: Triggered when the price closes past the recent swing high/low in the direction of the current trend (trend continuation).
- **CHOCH (Change of Character)**: Triggered when the price closes past the opposite structure peak/trough (reversal from bullish to bearish or vice versa).

### 3. Active (Unmitigated) Order Blocks (OB)
- **Bullish OB**: The body (Open to Close range) of the last bearish candle prior to a bullish BOS or CHOCH. It remains active until price dips below the low of that bearish candle:
  $$\text{Mitigation Condition: } \text{Low}[k] < \text{Low}_{\text{OB}}$$
- **Bearish OB**: The body of the last bullish candle prior to a bearish BOS or CHOCH. It remains active until price rises above the high of that bullish candle:
  $$\text{Mitigation Condition: } \text{High}[k] > \text{High}_{\text{OB}}$$

### 4. Freshness Score
Fresh order blocks are more reliable. We apply a linear decay of $2\%$ per candle age to the OB score:
$$\text{Freshness} = \max(0, 100 - \text{Age} \times 2)\%$$
where $\text{Age}$ is the number of candles elapsed since the OB was created.

### 5. Fair Value Gaps (FVG)
Represent institutional liquidity imbalances:
- **Bullish FVG**: Gaps where $\text{Low}[i] > \text{High}[i-2]$. Remains active until price trades back below the gap bottom.
- **Bearish FVG**: Gaps where $\text{High}[i] < \text{Low}[i-2]$. Remains active until price trades back above the gap top.

---

## 📊 7-Factor Confluence Model

To keep the AI model as the primary decision-maker, we replaced the old confidence logic with this weighted confluence model:

| Factor | Weight | Description |
| :--- | :---: | :--- |
| **AI Prediction (ML)** | **35%** | Raw probability score from RF + GB ensemble models |
| **Technical Indicators** | **20%** | Technical strength index (RSI, EMA crossovers) |
| **Volume** | **15%** | Volume spike strength compared to 20-period average |
| **Market Structure** | **10%** | Trend alignment based on latest BOS/CHOCH structure |
| **Order Block Proximity** | **10%** | Proximity to closest active OB, scaled by Freshness |
| **Fibonacci Support** | **5%** | Retest/proximity to 61.8%, 50%, or 38.2% levels |
| **FVG Proximity** | **5%** | Proximity to active liquidity imbalance gaps |

$$\text{Confluence Score} = (\text{ML} \times 0.35) + (\text{Tech} \times 0.20) + (\text{Vol} \times 0.15) + (\text{Struct} \times 0.10) + (\text{OB} \times 0.10) + (\text{Fib} \times 0.05) + (\text{FVG} \times 0.05)$$

---

## 🛡️ Strict Risk Guardrails

To prevent false signals or elevated conviction from single indicators, the following guardrails are hardcoded:

1. **AI Model Alignment Guardrail**:
   An Order Block, FVG, or Fibonacci level can *never* trigger a trade by itself. If the AI classification model's prediction probability is neutral or opposite (i.e. $\text{Probability} < 50\%$), the final score is capped below $40\%$, forcing a `NO TRADE (Low Confidence)` signal.

2. **No Elevation of HOLD Signals**:
   If the baseline conviction score (ML + Tech + Volume + Structure) is less than $40\%$ (HOLD criteria), the final score is capped at $39\%$ regardless of OB, FVG, or Fibonacci alignment.

3. **No Elevation of BUY/SELL to POWER BUY/SELL**:
   If the baseline conviction is moderate ($< 65\%$), the final score is capped at $64\%$ to ensure that auxiliary indicators cannot elevate a standard setup to a "Power Setup".

---

## 📈 Chart Visualizations

- **Order Blocks**: Displayed as semi-transparent green (Bullish) or red (Bearish) shaded rectangles extending from creation to the right edge of the chart.
- **FVG Zones**: Displayed as semi-transparent light blue shaded rectangles with dotted borders.
- **BOS/CHOCH Markers**: Plotted at the breakout index with short dotted trendlines indicating the broken structure price.
