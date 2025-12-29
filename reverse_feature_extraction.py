"""
Reverse Feature Extraction Module
Extracts mathematical formulas from confirmed reversals.
Supports arithmetic operations (+, -, *, /) and indicator combinations.

Key Innovation: AI learns formulas by analyzing patterns in confirmed reversals,
not just indicator combinations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import itertools
from functools import lru_cache


class TechnicalIndicators:
    """
    Compute technical indicators for reverse feature extraction.
    """
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        delta = np.diff(data)
        seed = delta[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = np.zeros_like(data)
        rs[:period] = np.nan
        
        for i in range(period, len(data)):
            delta_val = data[i] - data[i-1]
            if delta_val > 0:
                up = (up * (period - 1) + delta_val) / period
                down = down * (period - 1) / period
            else:
                down = (down * (period - 1) - delta_val) / period
                up = up * (period - 1) / period
            
            rs[i] = 100 - 100 / (1 + (up / (down + 1e-10)))
        
        return rs
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26) -> Tuple[np.ndarray, np.ndarray]:
        """MACD Line and Signal"""
        exp1 = pd.Series(data).ewm(span=fast).mean().values
        exp2 = pd.Series(data).ewm(span=slow).mean().values
        macd_line = exp1 - exp2
        signal = pd.Series(macd_line).ewm(span=9).mean().values
        return macd_line, signal
    
    @staticmethod
    def bbands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = pd.Series(data).rolling(window=period).mean().values
        std = pd.Series(data).rolling(window=period).std().values
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    @staticmethod
    def volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Volume SMA"""
        return pd.Series(volume).rolling(window=period).mean().values


class ReverseFeatureExtractor:
    """
    Extract mathematical formulas from confirmed reversals.
    Supports numerical operations and indicator combinations.
    """
    
    def __init__(self, lookback: int = 20):
        """
        Parameters:
        - lookback: Number of candles to look back for feature extraction
        """
        self.lookback = lookback
        self.indicators = TechnicalIndicators()
    
    def extract_features_around_reversal(self, df: pd.DataFrame, reversal_idx: int) -> Dict:
        """
        Extract all relevant features around a reversal point.
        Returns features as numerical values for pattern analysis.
        """
        start_idx = max(0, reversal_idx - self.lookback)
        end_idx = reversal_idx + 1
        
        window = df.iloc[start_idx:end_idx].copy()
        
        # Basic price features
        features = {
            'close_current': window['close'].iloc[-1],
            'close_change': (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0],
            'high_range': (window['high'].max() - window['low'].min()) / window['close'].iloc[-1],
            'volatility': window['close'].std() / window['close'].mean(),
        }
        
        # Volume features
        features['volume_current'] = window['volume'].iloc[-1]
        features['volume_sma_ratio'] = window['volume'].iloc[-1] / (window['volume'].mean() + 1e-10)
        features['volume_trend'] = (window['volume'].iloc[-1] - window['volume'].iloc[0]) / (window['volume'].iloc[0] + 1e-10)
        
        # Price relationship features
        features['close_vs_high'] = (window['close'].iloc[-1] - window['high'].min()) / (window['high'].max() - window['low'].min() + 1e-10)
        features['close_vs_low'] = (window['close'].iloc[-1] - window['low'].min()) / (window['high'].max() - window['low'].min() + 1e-10)
        
        # Trend features
        close_vals = window['close'].values
        features['momentum'] = (close_vals[-1] - close_vals[0]) / (abs(close_vals[0]) + 1e-10)
        features['avg_body'] = (window['close'] - window['open']).abs().mean() / window['close'].mean()
        
        # RSI
        if len(window) > 14:
            rsi = self.indicators.rsi(close_vals, period=14)
            features['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
        else:
            features['rsi'] = 50
        
        # MACD
        if len(window) > 26:
            macd_line, signal = self.indicators.macd(close_vals)
            features['macd'] = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
            features['macd_signal'] = signal[-1] if not np.isnan(signal[-1]) else 0
        else:
            features['macd'] = 0
            features['macd_signal'] = 0
        
        # Bollinger Bands
        if len(window) > 20:
            upper, mid, lower = self.indicators.bbands(close_vals, period=20)
            current_price = close_vals[-1]
            bb_width = upper[-1] - lower[-1]
            features['bb_position'] = (current_price - lower[-1]) / (bb_width + 1e-10)
            features['bb_squeeze'] = bb_width / (mid[-1] + 1e-10)
        else:
            features['bb_position'] = 0.5
            features['bb_squeeze'] = 0
        
        # ATR
        if len(window) > 14:
            atr = self.indicators.atr(window['high'].values, window['low'].values, close_vals, period=14)
            features['atr_ratio'] = atr[-1] / (close_vals[-1] + 1e-10) if close_vals[-1] != 0 else 0
        else:
            features['atr_ratio'] = 0
        
        return features
    
    def extract_all_reversals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for all confirmed reversals.
        Returns DataFrame with features and reversal labels.
        """
        confirmed_reversals = df[df['confirmed_label'] == 1]
        
        feature_list = []
        
        for idx, row in confirmed_reversals.iterrows():
            features = self.extract_features_around_reversal(df, idx)
            features['reversal_type'] = row['swing_type']
            features['future_move'] = row['future_move_pct']
            features['confirmed'] = 1
            feature_list.append(features)
        
        # Also extract from false signals for comparison
        false_signals = df[(df['raw_label'] == 1) & (df['confirmed_label'] == 0)]
        
        for idx, row in false_signals.head(len(confirmed_reversals)).iterrows():
            features = self.extract_features_around_reversal(df, idx)
            features['reversal_type'] = row['swing_type']
            features['future_move'] = row['future_move_pct']
            features['confirmed'] = 0
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def identify_key_patterns(self, features_df: pd.DataFrame) -> Dict:
        """
        Identify key patterns that differentiate confirmed vs false signals.
        """
        confirmed = features_df[features_df['confirmed'] == 1]
        false = features_df[features_df['confirmed'] == 0]
        
        patterns = {}
        
        # Calculate correlation with confirmation
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['future_move', 'confirmed']:
                # Mean difference
                confirmed_mean = confirmed[col].mean()
                false_mean = false[col].mean()
                
                if abs(false_mean) > 1e-10:
                    diff_ratio = abs(confirmed_mean - false_mean) / abs(false_mean)
                else:
                    diff_ratio = abs(confirmed_mean - false_mean) * 100
                
                patterns[col] = {
                    'confirmed_mean': confirmed_mean,
                    'false_mean': false_mean,
                    'difference_ratio': diff_ratio,
                    'better_for_confirmation': confirmed_mean > false_mean if col not in ['close_change'] else True
                }
        
        # Sort by difference ratio
        sorted_patterns = sorted(patterns.items(), 
                                key=lambda x: x[1]['difference_ratio'], 
                                reverse=True)
        
        return dict(sorted_patterns)
    
    def generate_formula_candidates(self, patterns: Dict, top_n: int = 5) -> List[str]:
        """
        Generate formula candidates based on identified patterns.
        Creates mathematical combinations of key features.
        """
        top_features = [name for name, _ in list(patterns.items())[:top_n]]
        
        formulas = []
        
        # Single features (normalized)
        for feature in top_features:
            formulas.append(f"NORM({feature})")
        
        # Feature combinations (ratios, sums, differences)
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                feat1, feat2 = top_features[i], top_features[j]
                formulas.append(f"({feat1} + {feat2}) / 2")
                formulas.append(f"({feat1} - {feat2})")
                formulas.append(f"({feat1} * {feat2})")
                if feat2 not in ['volume_trend', 'momentum']:
                    formulas.append(f"({feat1} / ({feat2} + 0.01))")
        
        # Common trading indicators combinations
        formulas.extend([
            "RSI > 70 AND volume_sma_ratio > 1.2",
            "RSI < 30 AND volume_sma_ratio > 1.2",
            "MACD > macd_signal AND volume_trend > 0",
            "bb_position < 0.2 AND atr_ratio > 0.01",
            "volatility > 0.01 AND volume_sma_ratio > 1.0",
        ])
        
        return formulas


def main():
    """
    Example usage of reverse feature extraction.
    """
    # This would be called from colab_reverse_feature_discovery.py
    pass


if __name__ == '__main__':
    main()
