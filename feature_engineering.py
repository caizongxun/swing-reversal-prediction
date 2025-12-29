"""
Phase 2: Feature Engineering for Swing Reversal Prediction
Compute technical indicators and features from labeled ground truth data.

Usage:
    python feature_engineering.py --input labeled_data.csv --output features_data.csv
"""

import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ReversalFeatureEngineer:
    """
    Compute reversal-specific technical indicators and features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: timestamp, open, high, low, close, volume, confirmed_label
        """
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
    
    # ========== MOMENTUM INDICATORS ==========
    
    def compute_rsi(self, period: int = 14) -> np.ndarray:
        """
        Compute Relative Strength Index (RSI).
        
        Parameters:
        -----------
        period : int
            Lookback period (typically 6, 14)
            
        Returns:
        --------
        np.ndarray
            RSI values (0-100)
        """
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def detect_rsi_divergence(self, rsi_period: int = 14, lookback: int = 10) -> np.ndarray:
        """
        Detect RSI divergence (price makes higher high but RSI makes lower high = bullish divergence).
        
        Parameters:
        -----------
        rsi_period : int
            RSI computation period
        lookback : int
            Lookback window for divergence detection
            
        Returns:
        --------
        np.ndarray
            1 = bullish divergence (bottom), -1 = bearish divergence (top), 0 = no divergence
        """
        rsi = self.compute_rsi(rsi_period)
        close = self.df['close'].values
        
        divergence = np.zeros(len(self.df))
        
        for i in range(lookback, len(self.df)):
            window_close = close[i-lookback:i+1]
            window_rsi = rsi[i-lookback:i+1]
            
            # Find local extrema
            close_min_idx = np.argmin(window_close[-5:])
            close_max_idx = np.argmax(window_close[-5:])
            rsi_min_idx = np.argmin(window_rsi[-5:])
            rsi_max_idx = np.argmax(window_rsi[-5:])
            
            # Bullish divergence: lower low in price, higher low in RSI
            if close_min_idx < rsi_min_idx and window_rsi[-1] > 30:
                divergence[i] = 1
            # Bearish divergence: higher high in price, lower high in RSI
            elif close_max_idx < rsi_max_idx and window_rsi[-1] < 70:
                divergence[i] = -1
        
        return divergence
    
    def compute_roc(self, period: int = 12) -> np.ndarray:
        """
        Compute Rate of Change.
        
        Parameters:
        -----------
        period : int
            Lookback period
            
        Returns:
        --------
        np.ndarray
            ROC values (%)
        """
        roc = ((self.df['close'] - self.df['close'].shift(period)) / 
               self.df['close'].shift(period) * 100)
        return roc.values
    
    # ========== VOLATILITY INDICATORS ==========
    
    def compute_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Bollinger Bands (upper, middle, lower).
        
        Returns:
        --------
        Tuple of (upper_band, middle_band, lower_band)
        """
        middle = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper.values, middle.values, lower.values
    
    def compute_bb_percent_b(self, period: int = 20, std_dev: int = 2) -> np.ndarray:
        """
        Compute Bollinger Bands %B (position between bands).
        
        %B = (Close - Lower Band) / (Upper Band - Lower Band)
        - %B < 0: Below lower band (oversold, potential reversal)
        - %B > 1: Above upper band (overbought, potential reversal)
        - %B = 0.5: At middle band
        
        Returns:
        --------
        np.ndarray
            %B values (typically 0-1, can exceed bounds)
        """
        upper, middle, lower = self.compute_bollinger_bands(period, std_dev)
        
        close = self.df['close'].values
        bb_percent_b = np.zeros(len(self.df))
        
        for i in range(len(self.df)):
            if upper[i] - lower[i] != 0:
                bb_percent_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
            else:
                bb_percent_b[i] = 0.5
        
        return bb_percent_b
    
    def compute_bb_bandwidth(self, period: int = 20, std_dev: int = 2) -> np.ndarray:
        """
        Compute Bollinger Bands Bandwidth (indicator of volatility squeeze/expansion).
        
        Bandwidth = (Upper Band - Lower Band) / Middle Band
        - Low bandwidth = squeeze (potential breakout/reversal coming)
        - High bandwidth = expansion (volatility spike, often at reversals)
        
        Returns:
        --------
        np.ndarray
            Bandwidth values
        """
        upper, middle, lower = self.compute_bollinger_bands(period, std_dev)
        
        bandwidth = np.zeros(len(self.df))
        for i in range(len(self.df)):
            if middle[i] != 0:
                bandwidth[i] = (upper[i] - lower[i]) / middle[i]
            else:
                bandwidth[i] = 0
        
        return bandwidth
    
    def compute_atr(self, period: int = 14) -> np.ndarray:
        """
        Compute Average True Range (volatility indicator).
        
        Returns:
        --------
        np.ndarray
            ATR values
        """
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        
        tr = np.zeros(len(self.df))
        for i in range(1, len(self.df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    # ========== CANDLESTICK PATTERNS ==========
    
    def detect_hammer(self, lookback: int = 1) -> np.ndarray:
        """
        Detect Hammer pattern (bullish reversal at bottom).
        Criteria:
        - Small body (close near open)
        - Long lower shadow (2-3x body size)
        - Short/no upper shadow
        - Appears after downtrend
        
        Returns:
        --------
        np.ndarray
            1 if hammer detected, 0 otherwise
        """
        open_p = self.df['open'].values
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        
        hammer = np.zeros(len(self.df))
        
        for i in range(lookback, len(self.df)):
            body_size = abs(close[i] - open_p[i])
            lower_shadow = min(open_p[i], close[i]) - low[i]
            upper_shadow = high[i] - max(open_p[i], close[i])
            total_range = high[i] - low[i]
            
            # Criteria for hammer
            if (body_size > 0 and 
                lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5 and
                lower_shadow > total_range * 0.5):
                hammer[i] = 1
        
        return hammer
    
    def detect_shooting_star(self, lookback: int = 1) -> np.ndarray:
        """
        Detect Shooting Star pattern (bearish reversal at top).
        Criteria:
        - Small body (close near open)
        - Long upper shadow (2-3x body size)
        - Short/no lower shadow
        - Appears after uptrend
        
        Returns:
        --------
        np.ndarray
            1 if shooting star detected, 0 otherwise
        """
        open_p = self.df['open'].values
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        
        shooting_star = np.zeros(len(self.df))
        
        for i in range(lookback, len(self.df)):
            body_size = abs(close[i] - open_p[i])
            upper_shadow = high[i] - max(open_p[i], close[i])
            lower_shadow = min(open_p[i], close[i]) - low[i]
            total_range = high[i] - low[i]
            
            # Criteria for shooting star
            if (body_size > 0 and 
                upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5 and
                upper_shadow > total_range * 0.5):
                shooting_star[i] = 1
        
        return shooting_star
    
    def detect_engulfing(self, lookback: int = 1) -> np.ndarray:
        """
        Detect Engulfing pattern (reversal pattern).
        Criteria:
        - Current bar completely contains previous bar
        - Opposite direction from previous bar
        
        Returns:
        --------
        np.ndarray
            1 if bullish engulfing, -1 if bearish engulfing, 0 otherwise
        """
        open_p = self.df['open'].values
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        
        engulfing = np.zeros(len(self.df))
        
        for i in range(1, len(self.df)):
            # Bullish engulfing
            if (close[i] > open_p[i] and  # Current bar is bullish
                close[i-1] < open_p[i-1] and  # Previous bar is bearish
                low[i] < low[i-1] and  # Current low is lower
                high[i] > high[i-1]):  # Current high is higher
                engulfing[i] = 1
            
            # Bearish engulfing
            elif (close[i] < open_p[i] and  # Current bar is bearish
                  close[i-1] > open_p[i-1] and  # Previous bar is bullish
                  low[i] < low[i-1] and  # Current low is lower
                  high[i] > high[i-1]):  # Current high is higher
                engulfing[i] = -1
        
        return engulfing
    
    # ========== VOLUME FEATURES ==========
    
    def compute_volume_oscillator(self, short_period: int = 5, long_period: int = 35) -> np.ndarray:
        """
        Compute Volume Oscillator.
        
        VO = (Short EMA - Long EMA) / Long EMA * 100
        
        Returns:
        --------
        np.ndarray
            VO values (%)
        """
        short_ema = self.df['volume'].ewm(span=short_period).mean()
        long_ema = self.df['volume'].ewm(span=long_period).mean()
        
        vo = ((short_ema - long_ema) / long_ema * 100)
        return vo.values
    
    def compute_volume_spike(self, lookback: int = 20) -> np.ndarray:
        """
        Compute Volume Spike Ratio (current volume / average volume).
        
        High spike (>1.5) indicates climax volume, often at reversals.
        
        Returns:
        --------
        np.ndarray
            Volume spike ratio
        """
        avg_volume = self.df['volume'].rolling(window=lookback).mean()
        spike_ratio = self.df['volume'] / (avg_volume + 1e-10)
        return spike_ratio.values
    
    def compute_volume_trend(self, period: int = 5) -> np.ndarray:
        """
        Compute volume trend (whether volume is increasing or decreasing).
        
        Returns:
        --------
        np.ndarray
            1 if volume increasing, -1 if decreasing, 0 if neutral
        """
        volume = self.df['volume'].values
        trend = np.zeros(len(self.df))
        
        for i in range(period, len(self.df)):
            vol_sma = np.mean(volume[i-period:i])
            if volume[i] > vol_sma * 1.1:
                trend[i] = 1
            elif volume[i] < vol_sma * 0.9:
                trend[i] = -1
        
        return trend
    
    # ========== PRICE ACTION FEATURES ==========
    
    def compute_price_momentum(self, period: int = 5) -> np.ndarray:
        """
        Compute price momentum over recent period.
        
        Returns:
        --------
        np.ndarray
            Momentum in percentage
        """
        momentum = ((self.df['close'] - self.df['close'].shift(period)) / 
                   self.df['close'].shift(period) * 100)
        return momentum.values
    
    def detect_gap(self) -> np.ndarray:
        """
        Detect price gaps (gap up/down between consecutive candles).
        
        Returns:
        --------
        np.ndarray
            Gap size in percentage
        """
        gap = np.zeros(len(self.df))
        
        for i in range(1, len(self.df)):
            open_current = self.df['open'].iloc[i]
            close_previous = self.df['close'].iloc[i-1]
            gap[i] = ((open_current - close_previous) / close_previous * 100) if close_previous != 0 else 0
        
        return gap
    
    def detect_higher_high_lower_low(self, lookback: int = 5) -> np.ndarray:
        """
        Detect if current high/low is higher/lower than previous lookback.
        
        Returns:
        --------
        np.ndarray
            1 = higher high (bullish), -1 = lower low (bearish), 0 = neutral
        """
        high = self.df['high'].values
        low = self.df['low'].values
        
        pattern = np.zeros(len(self.df))
        
        for i in range(lookback, len(self.df)):
            prev_high = np.max(high[i-lookback:i])
            prev_low = np.min(low[i-lookback:i])
            
            if high[i] > prev_high:
                pattern[i] = 1
            elif low[i] < prev_low:
                pattern[i] = -1
        
        return pattern
    
    # ========== MAIN PIPELINE ==========
    
    def compute_all_features(self) -> pd.DataFrame:
        """
        Compute all features and return augmented DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Original data + all computed features
        """
        print("Computing momentum indicators...")
        self.df['rsi_6'] = self.compute_rsi(6)
        self.df['rsi_14'] = self.compute_rsi(14)
        self.df['rsi_divergence'] = self.detect_rsi_divergence()
        self.df['roc_12'] = self.compute_roc(12)
        
        print("Computing volatility indicators...")
        self.df['bb_percent_b'] = self.compute_bb_percent_b()
        self.df['bb_bandwidth'] = self.compute_bb_bandwidth()
        self.df['atr_14'] = self.compute_atr(14)
        
        print("Detecting candlestick patterns...")
        self.df['hammer'] = self.detect_hammer()
        self.df['shooting_star'] = self.detect_shooting_star()
        self.df['engulfing'] = self.detect_engulfing()
        
        print("Computing volume features...")
        self.df['volume_oscillator'] = self.compute_volume_oscillator()
        self.df['volume_spike'] = self.compute_volume_spike()
        self.df['volume_trend'] = self.compute_volume_trend()
        
        print("Computing price action features...")
        self.df['price_momentum'] = self.compute_price_momentum()
        self.df['gap'] = self.detect_gap()
        self.df['higher_high_lower_low'] = self.detect_higher_high_lower_low()
        
        return self.df


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Feature Engineering for Reversal Classification'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file from Phase 1 (labeled_data.csv)')
    parser.add_argument('--output', type=str, default='features_data.csv',
                       help='Output CSV file with engineered features')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    
    # Initialize feature engineer
    engineer = ReversalFeatureEngineer(df)
    
    # Compute all features
    print("\nComputing features...")
    features_df = engineer.compute_all_features()
    
    # Display results
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    
    print("\nFeatures created:")
    feature_cols = [col for col in features_df.columns if col not in df.columns]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Total columns: {len(features_df.columns)}")
    
    # Show sample data
    print("\nSample data (first 10 rows with features):")
    print(features_df[['timestamp', 'close', 'volume', 'confirmed_label'] + feature_cols[:5]].head(10))
    
    # Check for NaN values
    print("\nData quality check:")
    nan_counts = features_df[feature_cols].isna().sum()
    print(f"  Total rows: {len(features_df)}")
    print(f"  Rows with NaN in features: {nan_counts.max()}")
    
    # Export
    print(f"\nExporting to {args.output}...")
    features_df.to_csv(args.output, index=False)
    print(f"Export complete: {len(features_df)} rows, {len(features_df.columns)} columns")
    
    # Statistics for confirmed reversals
    confirmed = features_df[features_df['confirmed_label'] == 1]
    print(f"\nConfirmed reversals: {len(confirmed)} ({len(confirmed)/len(features_df)*100:.1f}%)")
    
    if len(confirmed) > 0:
        print("\nFeature statistics for CONFIRMED reversals:")
        print(confirmed[feature_cols[:10]].describe())


if __name__ == '__main__':
    main()
