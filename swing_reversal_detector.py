"""
Swing Reversal Detector - Phase 1
Core module for identifying and filtering swing highs/lows based on future price movement.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


class SwingReversalDetector:
    """
    Detects swing reversal points with intelligent filtering.
    
    A True Reversal is identified when:
    1. Local High/Low detected (using Window parameter)
    2. Future N candles show significant price movement (> threshold%)
    """
    
    def __init__(self, window: int = 5, future_candles: int = 12, 
                 move_threshold: float = 1.0):
        """
        Parameters:
        -----------
        window : int
            Number of candles to look left/right for identifying local extremes
        future_candles : int
            Number of candles ahead to check for reversal confirmation
        move_threshold : float
            Minimum price movement percentage required for reversal confirmation
        """
        self.window = window
        self.future_candles = future_candles
        self.move_threshold = move_threshold
    
    def detect_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all swing highs and swing lows using local extrema logic.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: ['open', 'high', 'low', 'close']
            
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with added columns:
            - swing_type: 'high' / 'low' / None
            - raw_label: 1 (swing point) / 0 (not a swing point)
        """
        df = df.copy()
        df['swing_type'] = None
        df['raw_label'] = 0
        
        for i in range(self.window, len(df) - self.window):
            # Check for Swing High
            current_high = df.loc[i, 'high']
            if current_high == df.loc[i - self.window:i + self.window, 'high'].max():
                if current_high > df.loc[i - self.window:i - 1, 'high'].max():
                    df.loc[i, 'swing_type'] = 'high'
                    df.loc[i, 'raw_label'] = 1
            
            # Check for Swing Low
            current_low = df.loc[i, 'low']
            if current_low == df.loc[i - self.window:i + self.window, 'low'].min():
                if current_low < df.loc[i - self.window:i - 1, 'low'].min():
                    df.loc[i, 'swing_type'] = 'low'
                    df.loc[i, 'raw_label'] = 1
        
        return df
    
    def validate_reversals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter swing points based on future price movement confirmation.
        
        For Swing High: Check if price drops > threshold% within next N candles
        For Swing Low: Check if price rises > threshold% within next N candles
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with swing_type and raw_label columns
            
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with added columns:
            - is_confirmed_reversal: True/False (intelligent filtering result)
            - confirmed_label: 1 (true reversal) / 0 (false signal)
            - future_move_pct: Actual price movement in next N candles (%)
        """
        df = df.copy()
        df['is_confirmed_reversal'] = False
        df['confirmed_label'] = 0
        df['future_move_pct'] = 0.0
        
        for i in range(len(df) - self.future_candles):
            if df.loc[i, 'raw_label'] == 0:
                continue
            
            swing_type = df.loc[i, 'swing_type']
            reference_price = df.loc[i, 'close']
            
            # Get min/max price in next N candles
            future_slice = df.loc[i + 1:i + self.future_candles]
            future_high = future_slice['high'].max()
            future_low = future_slice['low'].min()
            
            if swing_type == 'high':
                # For swing high, check downward movement
                move_pct = ((reference_price - future_low) / reference_price) * 100
                df.loc[i, 'future_move_pct'] = move_pct
                
                if move_pct >= self.move_threshold:
                    df.loc[i, 'is_confirmed_reversal'] = True
                    df.loc[i, 'confirmed_label'] = 1
            
            elif swing_type == 'low':
                # For swing low, check upward movement
                move_pct = ((future_high - reference_price) / reference_price) * 100
                df.loc[i, 'future_move_pct'] = move_pct
                
                if move_pct >= self.move_threshold:
                    df.loc[i, 'is_confirmed_reversal'] = True
                    df.loc[i, 'confirmed_label'] = 1
        
        return df
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics on swing point detection and filtering.
        
        Returns:
        --------
        Dict
            Statistics including total swings, confirmed reversals, and accuracy metrics
        """
        total_raw_swings = df['raw_label'].sum()
        total_confirmed = df['confirmed_label'].sum()
        
        high_swings = (df['swing_type'] == 'high').sum()
        low_swings = (df['swing_type'] == 'low').sum()
        
        confirmed_highs = ((df['swing_type'] == 'high') & 
                          (df['confirmed_label'] == 1)).sum()
        confirmed_lows = ((df['swing_type'] == 'low') & 
                         (df['confirmed_label'] == 1)).sum()
        
        return {
            'total_raw_swings': total_raw_swings,
            'total_confirmed_reversals': total_confirmed,
            'filtering_ratio': total_confirmed / max(total_raw_swings, 1),
            'swing_highs_detected': high_swings,
            'swing_lows_detected': low_swings,
            'confirmed_highs': confirmed_highs,
            'confirmed_lows': confirmed_lows,
            'high_confirmation_rate': confirmed_highs / max(high_swings, 1),
            'low_confirmation_rate': confirmed_lows / max(low_swings, 1),
        }
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pipeline: detect swings and validate with future movement.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLC data
            
        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            Processed DataFrame and summary statistics
        """
        df = self.detect_swing_points(df)
        df = self.validate_reversals(df)
        stats = self.get_summary_statistics(df)
        
        return df, stats
