"""
AI Prediction Engine
Combines technical features, market sentiment, and ML models for stock predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from preprocess import TechnicalIndicators


class FeatureExtractor:
    """Extract advanced technical features for ML models"""
    
    @staticmethod
    def extract_features(df, window_periods=[5, 10, 20, 50]):
        """
        Extract comprehensive technical features from OHLCV data
        
        Parameters:
        -----------
        df : DataFrame with OHLCV columns
        window_periods : list of periods for moving averages
        
        Returns:
        --------
        DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # ─── PRICE FEATURES ──────────────────────────────────────────
        features['price'] = df['Close']
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        features['volatility'] = df['Close'].rolling(window=20).std()
        features['high_low_ratio'] = df['High'] / df['Low']
        features['close_open_ratio'] = df['Close'] / df['Open']
        
        # ─── MOVING AVERAGES & CROSSOVERS ──────────────────────────
        for period in window_periods:
            features[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            features[f'ema_{period}'] = TechnicalIndicators.calculate_ema(df['Close'], period)
        
        # SMA crossover signals
        features['sma_cross_signal'] = (
            (features['sma_5'] > features['sma_20']).astype(int) - 
            (features['sma_5'] < features['sma_20']).astype(int)
        )
        
        # ─── MOMENTUM INDICATORS ─────────────────────────────────────
        features['rsi_14'] = TechnicalIndicators.calculate_rsi(df['Close'], period=14)
        features['rsi_7'] = TechnicalIndicators.calculate_rsi(df['Close'], period=7)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.calculate_macd(df['Close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_crossover'] = (
            (features['macd'] > features['macd_signal']).astype(int) - 
            (features['macd'] < features['macd_signal']).astype(int)
        )
        
        # ─── VOLUME ANALYSIS ─────────────────────────────────────────
        features['volume'] = df['Volume']
        features['volume_ma'] = df['Volume'].rolling(window=20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_ma']
        features['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # ─── VOLATILITY INDICATORS ───────────────────────────────────
        if 'High' in df.columns and 'Low' in df.columns:
            features['atr'] = TechnicalIndicators.calculate_atr(df, period=14)
            features['atr_percent'] = features['atr'] / df['Close'] * 100
        
        # ─── SUPPORT & RESISTANCE ────────────────────────────────────
        features['distance_to_support'] = (
            df['Close'] - df['Low'].rolling(window=20).min()
        ) / (df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())
        
        features['distance_to_resistance'] = (
            df['High'].rolling(window=20).max() - df['Close']
        ) / (df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())
        
        # ─── PATTERN RECOGNITION ─────────────────────────────────────
        features['consecutive_green'] = (df['Close'] > df['Open']).rolling(window=5).sum()
        features['consecutive_red'] = (df['Open'] > df['Close']).rolling(window=5).sum()
        features['price_above_ma50'] = (df['Close'] > features['sma_50']).astype(int)
        features['price_above_ma200'] = (
            df['Close'] > df['Close'].rolling(window=200).mean()
        ).astype(int)
        
        # ─── RATE OF CHANGE ──────────────────────────────────────────
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = df['Close'].pct_change(periods=period)
        
        return features.fillna(0)
    
    @staticmethod
    def create_target_variable(df, target_period=5, threshold=0.01):
        """
        Create binary target: 1 if price goes up, 0 if down
        
        Parameters:
        -----------
        df : DataFrame with Close prices
        target_period : days ahead to predict
        threshold : minimum % change to classify as positive
        
        Returns:
        --------
        Series with binary targets (1=BUY, 0=SELL)
        """
        future_returns = df['Close'].shift(-target_period) / df['Close'] - 1
        target = (future_returns > threshold).astype(int)
        return target


class AIPredictor:
    """AI Model for stock price predictions"""
    
    def __init__(self, model_type='ensemble', model_path='ai_model.pkl'):
        """
        Initialize predictor
        
        Parameters:
        -----------
        model_type : str, 'rf' (RandomForest), 'gb' (GradientBoosting), 'ensemble'
        model_path : str, path to save/load model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if os.path.exists(model_path):
            self.load_model()
    
    def build_ensemble_model(self):
        """Create ensemble model combining multiple classifiers"""
        from sklearn.ensemble import VotingClassifier
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
    
    def train(self, X, y, test_size=0.2):
        """
        Train the AI model
        
        Parameters:
        -----------
        X : DataFrame with features
        y : Series with target values
        test_size : float, proportion for test set
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        if self.model_type == 'ensemble':
            self.build_ensemble_model()
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'gb':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict(self, X, return_probability=True):
        """
        Make predictions
        
        Parameters:
        -----------
        X : DataFrame with features
        return_probability : bool, return confidence scores
        
        Returns:
        --------
        dict with 'signal', 'confidence', 'probabilities'
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get latest prediction
        latest_pred = predictions[-1]
        latest_prob = probabilities[-1]
        
        # Signal interpretation
        signal = 'BUY' if latest_pred == 1 else 'SELL'
        confidence = max(latest_prob)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_probability': latest_prob[1],
            'sell_probability': latest_prob[0],
            'all_predictions': predictions,
            'all_probabilities': probabilities
        }
    
    def save_model(self):
        """Save model to disk"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.feature_names,
                'type': self.model_type
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['features']
            self.model_type = data['type']
            print(f"Model loaded from {self.model_path}")


class PredictionAnalyzer:
    """Analyze and score predictions with multiple factors"""
    
    @staticmethod
    def calculate_technical_score(df, features):
        """
        Score prediction based on technical indicators
        
        Returns: 0-100 score
        """
        score = 50  # Neutral starting point
        
        # RSI signals
        rsi = features['rsi_14'].iloc[-1]
        if rsi > 70:
            score -= 15  # Overbought
        elif rsi < 30:
            score += 15  # Oversold
        elif 40 < rsi < 60:
            score += 5   # Neutral momentum
        
        # MACD signals
        macd_cross = features['macd_crossover'].iloc[-1]
        if macd_cross == 1:  # Bullish crossover
            score += 15
        elif macd_cross == -1:  # Bearish crossover
            score -= 15
        
        # Volume signals
        vol_ratio = features['volume_ratio'].iloc[-1]
        if vol_ratio > 1.2:  # Above average volume
            score += 10
        
        # Moving average alignment
        price = df['Close'].iloc[-1]
        sma_50 = features['sma_50'].iloc[-1]
        sma_200 = features['sma_200'].iloc[-1]
        
        if sma_50 > sma_200:  # Uptrend
            score += 10
        elif sma_50 < sma_200:  # Downtrend
            score -= 10
        
        if price > sma_50:  # Price above short-term MA
            score += 5
        
        return max(0, min(100, score))
    
    @staticmethod
    def calculate_sentiment_score(headlines_sentiment=None, market_bias=None):
        """
        Score prediction based on sentiment analysis
        
        Returns: 0-100 score
        """
        score = 50  # Neutral
        
        if headlines_sentiment:
            if headlines_sentiment == 'positive':
                score += 20
            elif headlines_sentiment == 'negative':
                score -= 20
        
        if market_bias:
            if market_bias == 'bullish':
                score += 15
            elif market_bias == 'bearish':
                score -= 15
        
        return max(0, min(100, score))
    
    @staticmethod
    def calculate_combined_confidence(ai_confidence, technical_score, sentiment_score, 
                                     pattern_score=50, volume_strength=50):
        """
        Combine all confidence factors into final score
        """
        weights = {
            'ai': 0.35,
            'technical': 0.25,
            'sentiment': 0.15,
            'pattern': 0.15,
            'volume': 0.10
        }
        
        combined = (
            ai_confidence * weights['ai'] +
            (technical_score / 100) * weights['technical'] +
            (sentiment_score / 100) * weights['sentiment'] +
            (pattern_score / 100) * weights['pattern'] +
            (volume_strength / 100) * weights['volume']
        )
        
        return round(combined, 4)


class TradeExecution:
    """Calculate trade entry, targets, and stop losses"""
    
    @staticmethod
    def calculate_levels(current_price, atr=None, confidence=0.5, signal='BUY',
                        risk_percent=2, portfolio_value=100000):
        """
        Calculate trade execution levels
        
        Parameters:
        -----------
        current_price : float, current stock price
        atr : float, average true range
        confidence : float, 0-1 confidence in prediction
        signal : str, 'BUY' or 'SELL'
        risk_percent : float, % of portfolio to risk
        portfolio_value : float, total portfolio value
        
        Returns:
        --------
        dict with entry, target, stop loss, position size
        """
        if atr is None:
            atr = current_price * 0.02  # Default 2% of price
        
        # Risk amount
        risk_amount = portfolio_value * (risk_percent / 100)
        
        if signal == 'BUY':
            # Entry at current price
            entry = current_price
            
            # Stop loss: 2 ATR below entry
            stop_loss = entry - (2 * atr)
            
            # Target: 3-5x risk-reward based on confidence
            risk_reward_ratio = 3 + (confidence * 2)  # 3-5x depending on confidence
            target = entry + (atr * risk_reward_ratio)
            
        else:  # SELL
            entry = current_price
            stop_loss = entry + (2 * atr)
            risk_reward_ratio = 3 + (confidence * 2)
            target = entry - (atr * risk_reward_ratio)
        
        # Position sizing (how many shares)
        stop_risk = abs(entry - stop_loss)
        position_size = int(risk_amount / stop_risk)
        
        # Profit calculation
        profit_per_share = abs(target - entry)
        profit_amount = position_size * profit_per_share
        
        return {
            'entry': entry,
            'target': target,
            'sl': stop_loss,
            'pos_size': position_size,
            'risk_amt': risk_amount,
            'profit_amt': profit_amount,
            'risk_reward': f"1:{risk_reward_ratio:.1f}",
            'stop_loss_pct': abs((stop_loss - entry) / entry * 100),
            'target_pct': abs((target - entry) / entry * 100)
        }
    
    @staticmethod
    def calculate_sl_options(current_price, recent_low, recent_high, signal='BUY'):
        """
        Provide multiple stop loss options
        """
        atr_equiv = (recent_high - recent_low) / 2
        
        options = {
            'aggressive': (recent_low - atr_equiv * 0.5) if signal == 'BUY' else (recent_high + atr_equiv * 0.5),
            'moderate': (recent_low - atr_equiv) if signal == 'BUY' else (recent_high + atr_equiv),
            'conservative': (recent_low - atr_equiv * 1.5) if signal == 'BUY' else (recent_high + atr_equiv * 1.5),
        }
        
        return {k: max(0, v) for k, v in options.items()}
