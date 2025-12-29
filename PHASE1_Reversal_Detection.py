#!/usr/bin/env python3
"""
PHASE1: Swing Reversal Point Detection & Labeling

Features:
1. Identify all valid reversal points from downloaded K-line data
2. Classify reversal types (range support, range resistance, trend start, trend end)
3. Output labeled dataset for model training

Reversal definitions:
- Range reversals: bounce at support or pullback at resistance
- Trend reversals: trend direction change with momentum confirmation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ReversalDetector:
    """Reversal point detection engine"""
    
    def __init__(self, df, lookback=20, reversal_window=5):
        """
        Initialize reversal detector
        
        Args:
            df: DataFrame with timestamp, open, high, low, close, volume
            lookback: lookback window for support/resistance calculation
            reversal_window: confirmation window (N bars before/after)
        """
        self.df = df.copy()
        self.lookback = lookback
        self.reversal_window = reversal_window
        self.reversals = []
        
    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        self.df['SMA20'] = self.df['close'].rolling(20).mean()
        self.df['SMA50'] = self.df['close'].rolling(50).mean()
        self.df['SMA200'] = self.df['close'].rolling(200).mean()
        
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['Signal']
        
        sma = self.df['close'].rolling(20).mean()
        std = self.df['close'].rolling(20).std()
        self.df['BB_Upper'] = sma + 2 * std
        self.df['BB_Lower'] = sma - 2 * std
        self.df['BB_Pct'] = (self.df['close'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        
        self.df['Support'] = self.df['low'].rolling(self.lookback).min()
        self.df['Resistance'] = self.df['high'].rolling(self.lookback).max()
        
        vol_sma = self.df['volume'].rolling(20).mean()
        vol_std = self.df['volume'].rolling(20).std()
        self.df['Volume_Z'] = (self.df['volume'] - vol_sma) / vol_std
        
        self.df['ROC'] = (self.df['close'] - self.df['close'].shift(5)) / self.df['close'].shift(5) * 100
        
    def detect_regime(self):
        """Detect market regime (trend vs range)"""
        self.df['Above_SMA200'] = self.df['close'] > self.df['SMA200']
        
        tr = pd.concat([
            self.df['high'] - self.df['low'],
            abs(self.df['high'] - self.df['close'].shift()),
            abs(self.df['low'] - self.df['close'].shift())
        ], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(14).mean()
        
        self.df['Volatility_Ratio'] = self.df['ATR'] / self.df['SMA20']
        
        sma20_slope = (self.df['SMA20'] - self.df['SMA20'].shift(5)) / self.df['SMA20'].shift(5) * 100
        self.df['Trend_Slope'] = sma20_slope
        
        self.df['Regime'] = 'Range'
        self.df.loc[self.df['Trend_Slope'] > 1.0, 'Regime'] = 'Uptrend'
        self.df.loc[self.df['Trend_Slope'] < -1.0, 'Regime'] = 'Downtrend'
        
    def detect_range_reversals(self):
        """Detect range bound reversal points"""
        range_reversals = []
        
        for i in range(self.reversal_window, len(self.df) - self.reversal_window):
            current = self.df.iloc[i]
            
            if pd.isna(current['BB_Lower']) or pd.isna(current['RSI']):
                continue
            
            if (current['BB_Pct'] < 0.2 and current['RSI'] < 40):
                future_high = self.df.iloc[i:i+self.reversal_window+1]['high'].max()
                if future_high > current['close'] * 1.002:
                    range_reversals.append({
                        'index': i,
                        'timestamp': current['timestamp'],
                        'price': current['close'],
                        'type': 'Support',
                        'regime': 'Range',
                        'rsi': current['RSI'],
                        'bb_pct': current['BB_Pct'],
                        'confirmation': 'Bounce'
                    })
            
            if (current['BB_Pct'] > 0.8 and current['RSI'] > 60):
                future_low = self.df.iloc[i:i+self.reversal_window+1]['low'].min()
                if future_low < current['close'] * 0.998:
                    range_reversals.append({
                        'index': i,
                        'timestamp': current['timestamp'],
                        'price': current['close'],
                        'type': 'Resistance',
                        'regime': 'Range',
                        'rsi': current['RSI'],
                        'bb_pct': current['BB_Pct'],
                        'confirmation': 'Pullback'
                    })
        
        return range_reversals
    
    def detect_trend_reversals(self):
        """Detect trend reversal points"""
        trend_reversals = []
        
        for i in range(self.reversal_window * 2, len(self.df) - self.reversal_window):
            current = self.df.iloc[i]
            
            if pd.isna(current['MACD_Hist']) or pd.isna(current['Volume_Z']):
                continue
            
            if i > 0:
                prev_regime = self.df.iloc[i-5:i]['Regime'].value_counts().index[0]
                if prev_regime in ['Downtrend', 'Range']:
                    if (current['MACD_Hist'] > 0 and 
                        self.df.iloc[i-1]['MACD_Hist'] <= 0 and
                        current['Volume_Z'] > 1.5):
                        trend_reversals.append({
                            'index': i,
                            'timestamp': current['timestamp'],
                            'price': current['close'],
                            'type': 'Trend_Start_Up',
                            'regime': 'Trend',
                            'signal': 'MACD_Crossover + Volume',
                            'macd_hist': current['MACD_Hist'],
                            'volume_z': current['Volume_Z']
                        })
                    
            if i > 0:
                curr_regime = self.df.iloc[i-5:i]['Regime'].value_counts().index[0]
                if curr_regime == 'Uptrend':
                    if (current['MACD_Hist'] < 0 and 
                        self.df.iloc[i-1]['MACD_Hist'] >= 0 and
                        current['RSI'] > 50):
                        trend_reversals.append({
                            'index': i,
                            'timestamp': current['timestamp'],
                            'price': current['close'],
                            'type': 'Trend_End_Down',
                            'regime': 'Trend',
                            'signal': 'MACD_Divergence',
                            'macd_hist': current['MACD_Hist'],
                            'rsi': current['RSI']
                        })
        
        return trend_reversals
    
    def run_detection(self):
        """Execute complete reversal detection pipeline"""
        print("[PHASE1] Starting reversal point detection...")
        print(f"Data size: {len(self.df)} K-lines")
        
        print("\n[1/4] Calculating technical indicators...")
        self.calculate_technical_indicators()
        
        print("[2/4] Detecting market regime (trend/range)...")
        self.detect_regime()
        regime_counts = self.df['Regime'].value_counts()
        print(f"    Trend bars: {regime_counts.get('Uptrend', 0) + regime_counts.get('Downtrend', 0)}")
        print(f"    Range bars: {regime_counts.get('Range', 0)}")
        
        print("\n[3/4] Detecting range bound reversals...")
        range_reversals = self.detect_range_reversals()
        print(f"    Found {len(range_reversals)} range reversals")
        if len(range_reversals) > 0:
            print(f"      - Support bounces: {len([r for r in range_reversals if r['type']=='Support'])}")
            print(f"      - Resistance pullbacks: {len([r for r in range_reversals if r['type']=='Resistance'])}")
        
        print("\n[4/4] Detecting trend reversals...")
        trend_reversals = self.detect_trend_reversals()
        print(f"    Found {len(trend_reversals)} trend reversals")
        if len(trend_reversals) > 0:
            trend_up = len([r for r in trend_reversals if 'Up' in r['type']])
            trend_down = len([r for r in trend_reversals if 'Down' in r['type']])
            print(f"      - Trend starts (up): {trend_up}")
            print(f"      - Trend ends (down): {trend_down}")
        
        all_reversals = range_reversals + trend_reversals
        self.reversals = pd.DataFrame(all_reversals).sort_values('index').reset_index(drop=True)
        
        return self.reversals
    
    def label_dataset(self):
        """Label the entire dataset with reversal points"""
        self.df['Reversal_Label'] = 0
        self.df['Reversal_Type'] = 'None'
        self.df['Reversal_Strength'] = 0.0
        
        for idx, row in self.reversals.iterrows():
            i = row['index']
            if i < len(self.df):
                self.df.at[i, 'Reversal_Label'] = 1
                self.df.at[i, 'Reversal_Type'] = row['type']
                
                if 'rsi' in row:
                    strength = abs(row['rsi'] - 50) / 50
                    self.df.at[i, 'Reversal_Strength'] = min(strength, 1.0)
        
        return self.df
    
    def save_results(self, output_path='labeled_klines_phase1.csv'):
        """Save results to file"""
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'SMA20', 'SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_Hist',
                'BB_Upper', 'BB_Lower', 'BB_Pct', 'Support', 'Resistance',
                'Volume_Z', 'Regime', 'Reversal_Label', 'Reversal_Type', 'Reversal_Strength']
        
        existing_cols = [c for c in cols if c in self.df.columns]
        self.df[existing_cols].to_csv(output_path, index=False)
        print(f"\nSaved labeled dataset: {output_path}")
        
        return output_path


def main():
    """Main function"""
    import os
    
    print("="*70)
    print("PHASE1: Cryptocurrency Reversal Point Detection & Labeling")
    print("="*70)
    
    csv_file = None
    if os.path.exists('downloaded_klines/BTCUSDT_15m_10000.csv'):
        csv_file = 'downloaded_klines/BTCUSDT_15m_10000.csv'
    elif os.path.exists('BTCUSDT_15m_10000.csv'):
        csv_file = 'BTCUSDT_15m_10000.csv'
    
    if csv_file is None:
        print("Error: CSV file not found")
        print("Please run: python colab_hf_working.py first")
        return
    
    print(f"\nLoading data: {csv_file}")
    df = pd.read_csv(csv_file)
    
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    detector = ReversalDetector(df, lookback=20, reversal_window=3)
    
    reversals = detector.run_detection()
    
    print(f"\nLabeling dataset...")
    labeled_df = detector.label_dataset()
    
    output_file = detector.save_results('labeled_klines_phase1.csv')
    
    print("\n")
    print("="*70)
    print("Detection Results Summary")
    print("="*70)
    print(f"Total reversal points: {detector.reversals.shape[0]}")
    print(f"\nType distribution:")
    print(detector.reversals['type'].value_counts())
    
    print(f"\nTop 10 reversal points:")
    print(detector.reversals[['index', 'timestamp', 'type', 'price']].head(10))
    
    print(f"\nSaved to: {output_file}")
    return labeled_df, reversals


if __name__ == '__main__':
    main()
