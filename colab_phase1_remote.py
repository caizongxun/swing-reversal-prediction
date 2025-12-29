#!/usr/bin/env python3
"""
COLAB PHASE1 Remote Execution Script

Execution method (one-liner in Colab):
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_remote.py | python3

Features:
1. Auto download K-line data (HuggingFace)
2. Standardize column names
3. Execute PHASE1 reversal detection
4. Output labeled dataset
5. Generate summary statistics
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE1: Reversal Point Detection - Remote Execution")
print("="*70)

print("\n[1/5] Installing dependencies...")
packages = ["datasets", "huggingface-hub", "pandas", "numpy", "scikit-learn"]
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
print("Dependencies installed")

print("\n[2/5] Downloading K-line data from HuggingFace...")

from huggingface_hub import hf_hub_download

try:
    csv_file = hf_hub_download(
        repo_id="zongowo111/cpb-models",
        filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
        repo_type="dataset"
    )
    print(f"Successfully downloaded: {csv_file}")
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Download failed: {str(e)[:100]}")
    sys.exit(1)

print("\n[3/5] Calculating technical indicators...")

df['close'] = pd.to_numeric(df['close'])
df['volume'] = pd.to_numeric(df['volume'])
df['open'] = pd.to_numeric(df['open'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])

if 'open_time' in df.columns:
    df['timestamp'] = df['open_time']

print(f"Data shape: {df.shape}")
print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

df['SMA20'] = df['close'].rolling(20).mean()
df['SMA50'] = df['close'].rolling(50).mean()
df['SMA200'] = df['close'].rolling(200).mean()

delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['Signal']

sma = df['close'].rolling(20).mean()
std = df['close'].rolling(20).std()
df['BB_Upper'] = sma + 2 * std
df['BB_Lower'] = sma - 2 * std
df['BB_Pct'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

df['Support'] = df['low'].rolling(20).min()
df['Resistance'] = df['high'].rolling(20).max()

vol_sma = df['volume'].rolling(20).mean()
vol_std = df['volume'].rolling(20).std()
df['Volume_Z'] = (df['volume'] - vol_sma) / vol_std

sma20_slope = (df['SMA20'] - df['SMA20'].shift(5)) / df['SMA20'].shift(5) * 100
df['Trend_Slope'] = sma20_slope
df['Regime'] = 'Range'
df.loc[df['Trend_Slope'] > 1.0, 'Regime'] = 'Uptrend'
df.loc[df['Trend_Slope'] < -1.0, 'Regime'] = 'Downtrend'

print("Technical indicators calculated")

print("\n[4/5] Executing PHASE1 reversal detection...")

reversals = []
reversal_window = 3
lookback = 20

for i in range(reversal_window, len(df) - reversal_window):
    current = df.iloc[i]
    
    if pd.isna(current['BB_Lower']) or pd.isna(current['RSI']):
        continue
    
    if (current['BB_Pct'] < 0.2 and current['RSI'] < 40):
        future_high = df.iloc[i:i+reversal_window+1]['high'].max()
        if future_high > current['close'] * 1.002:
            reversals.append({
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
        future_low = df.iloc[i:i+reversal_window+1]['low'].min()
        if future_low < current['close'] * 0.998:
            reversals.append({
                'index': i,
                'timestamp': current['timestamp'],
                'price': current['close'],
                'type': 'Resistance',
                'regime': 'Range',
                'rsi': current['RSI'],
                'bb_pct': current['BB_Pct'],
                'confirmation': 'Pullback'
            })

for i in range(reversal_window * 2, len(df) - reversal_window):
    current = df.iloc[i]
    
    if pd.isna(current['MACD_Hist']) or pd.isna(current['Volume_Z']):
        continue
    
    if i > 0:
        prev_regime = df.iloc[i-5:i]['Regime'].value_counts().index[0]
        if prev_regime in ['Downtrend', 'Range']:
            if (current['MACD_Hist'] > 0 and 
                df.iloc[i-1]['MACD_Hist'] <= 0 and
                current['Volume_Z'] > 1.5):
                reversals.append({
                    'index': i,
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'type': 'Trend_Start_Up',
                    'regime': 'Trend',
                    'signal': 'MACD_Crossover + Volume',
                    'macd_hist': current['MACD_Hist'],
                    'volume_z': current['Volume_Z']
                })
        
        curr_regime = df.iloc[i-5:i]['Regime'].value_counts().index[0]
        if curr_regime == 'Uptrend':
            if (current['MACD_Hist'] < 0 and 
                df.iloc[i-1]['MACD_Hist'] >= 0 and
                current['RSI'] > 50):
                reversals.append({
                    'index': i,
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'type': 'Trend_End_Down',
                    'regime': 'Trend',
                    'signal': 'MACD_Divergence',
                    'macd_hist': current['MACD_Hist'],
                    'rsi': current['RSI']
                })

reversals_df = pd.DataFrame(reversals).sort_values('index').reset_index(drop=True)
print(f"Detected {len(reversals_df)} reversal points")

print("\n[5/5] Generating labeled dataset...")

df['Reversal_Label'] = 0
df['Reversal_Type'] = 'None'
df['Reversal_Strength'] = 0.0

for idx, row in reversals_df.iterrows():
    i = row['index']
    if i < len(df):
        df.at[i, 'Reversal_Label'] = 1
        df.at[i, 'Reversal_Type'] = row['type']
        
        if 'rsi' in row and not pd.isna(row['rsi']):
            strength = abs(row['rsi'] - 50) / 50
            df.at[i, 'Reversal_Strength'] = min(strength, 1.0)

output_file = 'labeled_klines_phase1.csv'
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
        'SMA20', 'SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Pct', 'Support', 'Resistance',
        'Volume_Z', 'Regime', 'Reversal_Label', 'Reversal_Type', 'Reversal_Strength']

existing_cols = [c for c in cols if c in df.columns]
df[existing_cols].to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")

print("\n" + "="*70)
print("PHASE1 Detection Results Summary")
print("="*70)
print(f"\nTotal reversal points: {len(reversals_df)}")
print(f"\nType distribution:")
if len(reversals_df) > 0:
    print(reversals_df['type'].value_counts())
else:
    print("No reversal points detected")

print(f"\nTop 10 reversal points:")
if len(reversals_df) > 0:
    print(reversals_df[['index', 'timestamp', 'type', 'price']].head(10))

print(f"\nRegime distribution:")
print(df['Regime'].value_counts())

print(f"\nPHASE1 completed! Next: PHASE2 Feature extraction")
print("="*70)
