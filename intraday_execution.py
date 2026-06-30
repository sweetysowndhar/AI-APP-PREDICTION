import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 – Institutional Intraday Execution Engine (v2)
# Improvements:
#   1. Tiered Institutional Score ranking (A+, Strong Buy, Buy, Watchlist, Ignore)
#   2. Dynamic RVOL threshold by market cap (Large / Mid / Small)
#   3. Time filter: 11:00-13:00 → "Low Priority" (not blocked)
#   4. Configurable ORB window (5 / 15 / 30 min)
#   5. Anchored VWAP support (future-ready placeholder)
# ─────────────────────────────────────────────────────────────────────────────

class IntradayExecutionEngine:

    # ── Scoring Tiers ─────────────────────────────────────────────────────────
    SCORE_TIERS = [
        (95, 100, "A+ Institutional", "#10b981", "🏛️"),
        (90,  94, "Strong Buy",       "#00b386", "🚀"),
        (80,  89, "Buy",              "#22d3ee", "✅"),
        (70,  79, "Watchlist",        "#fbbf24", "👀"),
        ( 0,  69, "Ignore",           "#ef4444", "❌"),
    ]

    @classmethod
    def get_score_tier(cls, score: int) -> dict:
        """Return tier metadata dict for a given score."""
        for lo, hi, label, color, emoji in cls.SCORE_TIERS:
            if lo <= score <= hi:
                return {"label": label, "color": color, "emoji": emoji, "score": score}
        return {"label": "Ignore", "color": "#ef4444", "emoji": "❌", "score": score}

    # ── Dynamic RVOL Threshold ─────────────────────────────────────────────────
    @staticmethod
    def rvol_threshold(market_cap: str = "mid") -> float:
        """
        Dynamic RVOL thresholds by market cap.
        market_cap: "large" → 1.3,  "mid" → 1.5,  "small" → 2.0
        """
        thresholds = {"large": 1.3, "mid": 1.5, "small": 2.0}
        return thresholds.get(market_cap.lower(), 1.5)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_vwap(df):
        """
        Cumulative intraday VWAP = sum(Typical Price × Volume) / sum(Volume)
        grouped by date (resets each session).
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)

        df = df.copy()
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['pv'] = typical_price * df['Volume']

        dates = df.index.date
        cum_pv  = df.groupby(dates)['pv'].cumsum()
        cum_vol = df.groupby(dates)['Volume'].cumsum()

        vwap = cum_pv / cum_vol
        return vwap.fillna(typical_price)

    @staticmethod
    def calculate_anchored_vwap(df, anchor_date=None):
        """
        Anchored VWAP — calculated from a specific anchor date (e.g. earnings,
        swing low, major support).  Defaults to the first bar of the DataFrame.
        Returns a Series aligned to df.index.
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)

        if anchor_date is not None:
            anchor_df = df[df.index.date >= anchor_date]
        else:
            anchor_df = df  # anchor = first bar

        if anchor_df.empty:
            anchor_df = df

        df2 = anchor_df.copy()
        typical_price = (df2['High'] + df2['Low'] + df2['Close']) / 3.0
        df2['pv'] = typical_price * df2['Volume']
        cum_pv  = df2['pv'].cumsum()
        cum_vol = df2['Volume'].cumsum()
        avwap = (cum_pv / cum_vol).fillna(typical_price)
        # Reindex to match full df
        return avwap.reindex(df.index)

    # ── ORB (configurable window) ──────────────────────────────────────────────
    @staticmethod
    def get_orb_range(df, window_minutes: int = 30):
        """
        Identifies the high/low of the opening range.
        window_minutes: 5 / 15 / 30 (default 30)
        Range = 9:15 AM  →  9:15 + window_minutes
        """
        if df is None or df.empty:
            return None, None

        latest_date = df.index[-1].date()
        day_df = df[df.index.date == latest_date]

        start_time = "09:15"
        end_minutes = 15 + window_minutes
        end_h, end_m = divmod(end_minutes, 60)
        end_time = f"{9 + end_h:02d}:{end_m:02d}"

        orb_df = day_df.between_time(start_time, end_time)
        if orb_df.empty:
            orb_df = day_df.head(max(1, window_minutes // 5))

        if orb_df.empty:
            return None, None

        return float(orb_df['High'].max()), float(orb_df['Low'].min())

    # ── RVOL ──────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_rvol(df, window: int = 20) -> float:
        """RVOL = current volume / N-period average volume."""
        if df is None or len(df) < window:
            return 1.0

        curr_vol = float(df['Volume'].iloc[-1])
        avg_vol  = float(df['Volume'].iloc[-window - 1:-1].mean())

        if avg_vol == 0:
            return 1.0

        return curr_vol / avg_vol

    # ── Time Priority ──────────────────────────────────────────────────────────
    @staticmethod
    def get_time_priority(timestamp):
        """
        Returns (zone_name, time_multiplier).

        High Priority  09:20 – 10:30  → 1.20
        Good           13:45 – 15:00  → 1.00
        Low Priority   11:00 – 13:00  → 0.70  ← changed from block to Low Priority
        Neutral        (all other)    → 0.90
        """
        if timestamp is None:
            return 'Neutral', 0.9

        t = timestamp.time()
        if pd.Timestamp('09:20').time() <= t <= pd.Timestamp('10:30').time():
            return 'High Priority', 1.2
        elif pd.Timestamp('13:45').time() <= t <= pd.Timestamp('15:00').time():
            return 'Good', 1.0
        elif pd.Timestamp('11:00').time() <= t <= pd.Timestamp('13:00').time():
            return 'Low Priority', 0.7          # Not blocked — reduced weight
        else:
            return 'Neutral', 0.9

    # ── Main Evaluation ───────────────────────────────────────────────────────
    @classmethod
    def evaluate_entry(
        cls,
        df,
        signal_direction: str,
        mtf_sync: bool = False,
        liquidity_sweep: bool = False,
        ob_prox: bool = False,
        news_score: float = 0.0,
        market_cap: str = "mid",
        orb_window: int = 30,
    ) -> dict:
        """
        Aggregates all Phase-6 conditions into an Institutional Entry Score (0-100)
        and returns a rich result dict.

        Weights
        -------
        VWAP              30 pts
        ORB               25 pts
        RVOL              20 pts
        Time Priority     15 pts  (multiplier applied post-sum)
        SMC components    10 pts  (MTF + Liquidity Sweep + OB proximity)
        """
        if df is None or df.empty:
            return cls._empty_result()

        current_price = float(df['Close'].iloc[-1])
        latest_time   = df.index[-1]

        # 1. VWAP
        vwap_series   = cls.calculate_vwap(df)
        current_vwap  = float(vwap_series.iloc[-1])
        vwap_ok = (current_price > current_vwap) if signal_direction == "BUY" else (current_price < current_vwap)

        # 2. ORB (configurable window)
        orb_high, orb_low = cls.get_orb_range(df, window_minutes=orb_window)
        orb_ok = False
        if orb_high is not None and orb_low is not None:
            orb_ok = (current_price > orb_high) if signal_direction == "BUY" else (current_price < orb_low)

        # 3. RVOL (dynamic threshold)
        rvol          = cls.calculate_rvol(df)
        rvol_thresh   = cls.rvol_threshold(market_cap)
        rvol_ok       = rvol >= rvol_thresh

        # 4. Time Priority
        time_zone, time_mult = cls.get_time_priority(latest_time)

        # 5. SMC sub-components (10 pts total)
        smc_points = 0
        if mtf_sync:        smc_points += 4
        if liquidity_sweep: smc_points += 3
        if ob_prox:         smc_points += 3

        # Raw score
        vwap_pts = 30 if vwap_ok else 0
        orb_pts  = 25 if orb_ok  else 0
        rvol_pts = 20 if rvol_ok else 0

        time_pts = 0
        if time_zone == 'High Priority': time_pts = 15
        elif time_zone == 'Good':        time_pts = 12
        elif time_zone == 'Neutral':     time_pts = 8
        elif time_zone == 'Low Priority': time_pts = 5

        raw_score = vwap_pts + orb_pts + rvol_pts + time_pts + smc_points
        # Apply time multiplier (no hard block)
        score = min(100, int(raw_score * time_mult))

        tier = cls.get_score_tier(score)

        return {
            'score':         score,
            'tier_label':    tier['label'],
            'tier_color':    tier['color'],
            'tier_emoji':    tier['emoji'],
            'vwap_ok':       vwap_ok,
            'orb_ok':        orb_ok,
            'rvol':          round(rvol, 2),
            'rvol_ok':       rvol_ok,
            'rvol_thresh':   rvol_thresh,
            'time_zone':     time_zone,
            'current_vwap':  round(current_vwap, 2),
            'orb_high':      round(orb_high, 2) if orb_high else None,
            'orb_low':       round(orb_low,  2) if orb_low  else None,
            'orb_window':    orb_window,
            'market_cap':    market_cap,
        }

    @classmethod
    def _empty_result(cls):
        tier = cls.get_score_tier(0)
        return {
            'score': 0, 'tier_label': 'Ignore', 'tier_color': '#ef4444',
            'tier_emoji': '❌', 'vwap_ok': False, 'orb_ok': False,
            'rvol': 1.0, 'rvol_ok': False, 'rvol_thresh': 1.5,
            'time_zone': 'Neutral', 'current_vwap': 0.0,
            'orb_high': None, 'orb_low': None,
            'orb_window': 30, 'market_cap': 'mid',
        }
