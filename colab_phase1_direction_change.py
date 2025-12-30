#!/usr/bin/env python3
"""
PHASE1 Direction Change Detection - Colab Version

Usage in Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_direction_change.py | python3

Core Logic:
- Detect direction changes in close prices
- From uptrend to downtrend = High reversal point
- From downtrend to uptrend = Low reversal point
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE1: Direction Change Detection")
print("="*70)

print("\nLoading data...")

try:
    from huggingface_hub import hf_hub_download
    csv_file = hf_hub_download(
        repo_id="zongowo111/cpb-models",
        filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(csv_file)
    print("Successfully downloaded from HuggingFace")
except Exception as e:
    print(f"Error: {str(e)[:100]}")
    print("Using local file...")
    import os
    if os.path.exists('labeled_klines_phase1.csv'):
        df = pd.read_csv('labeled_klines_phase1.csv')
        print("Loaded from labeled_klines_phase1.csv")
    else:
        print("ERROR: No data file found")
        exit(1)

df['close'] = pd.to_numeric(df['close'])
df['open'] = pd.to_numeric(df['open'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

if 'timestamp' not in df.columns:
    if 'open_time' in df.columns:
        df['timestamp'] = df['open_time']
    else:
        df['timestamp'] = range(len(df))

print(f"Data shape: {df.shape}")
print(f"Date range: {str(df['timestamp'].iloc[0])[:19]} to {str(df['timestamp'].iloc[-1])[:19]}")

lookback = 3
lookforward = 3
confirm_local = True

print(f"\nDetecting reversals (lookback={lookback}, confirm_local={confirm_local})...")

reversals = []

def calculate_direction(idx, n):
    if idx < n:
        return 0
    start_close = df.iloc[idx-n]['close']
    end_close = df.iloc[idx]['close']
    return end_close - start_close

def is_local_high(idx):
    if idx <= 0 or idx >= len(df) - 1:
        return False
    current = df.iloc[idx]['close']
    prev = df.iloc[idx-1]['close']
    next_val = df.iloc[idx+1]['close']
    return current >= prev and current >= next_val

def is_local_low(idx):
    if idx <= 0 or idx >= len(df) - 1:
        return False
    current = df.iloc[idx]['close']
    prev = df.iloc[idx-1]['close']
    next_val = df.iloc[idx+1]['close']
    return current <= prev and current <= next_val

print("\n[1/2] Detecting direction changes...")

for i in range(lookback + 1, len(df) - lookforward):
    prev_direction = calculate_direction(i-1, lookback)
    current_direction = calculate_direction(i, 1)
    
    if prev_direction > 0 and current_direction < 0:
        if confirm_local and not is_local_high(i-1):
            continue
        reversals.append({
            'index': i-1,
            'timestamp': df.iloc[i-1]['timestamp'],
            'close': df.iloc[i-1]['close'],
            'type': 'Local_High'
        })
    
    elif prev_direction < 0 and current_direction > 0:
        if confirm_local and not is_local_low(i-1):
            continue
        reversals.append({
            'index': i-1,
            'timestamp': df.iloc[i-1]['timestamp'],
            'close': df.iloc[i-1]['close'],
            'type': 'Local_Low'
        })

print(f"Found {len(reversals)} reversal points")

print("\n[2/2] Labeling dataset...")

df['Reversal_Label'] = 0
df['Reversal_Type'] = 'None'
df['Reversal_Price'] = 0.0

for reversal in reversals:
    i = int(reversal['index'])
    if i < len(df):
        df.at[i, 'Reversal_Label'] = 1
        df.at[i, 'Reversal_Type'] = reversal['type']
        df.at[i, 'Reversal_Price'] = reversal['close']

output_file = 'labeled_klines_phase1_direction_change.csv'
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
        'Reversal_Label', 'Reversal_Type', 'Reversal_Price']
existing_cols = [c for c in cols if c in df.columns]
df[existing_cols].to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")

print(f"\n{'='*70}")
print("PHASE1 Detection Results")
print(f"{'='*70}")

reversals_df = pd.DataFrame(reversals) if reversals else pd.DataFrame()
total = len(reversals)
local_lows = len(reversals_df[reversals_df['type'] == 'Local_Low'])
local_highs = len(reversals_df[reversals_df['type'] == 'Local_High'])

print(f"\nTotal reversal points: {total}")
print(f"  - Local Lows (Buy Points): {local_lows}")
print(f"  - Local Highs (Sell Points): {local_highs}")
print(f"\nReversal ratio: {total / len(df) * 100:.2f}%")
print(f"Average spacing: {len(df) // (total + 1):.0f} bars")

if len(reversals_df) > 0:
    print(f"\nFirst 15 reversals:")
    print(reversals_df[['index', 'timestamp', 'type', 'close']].head(15).to_string())
    
    print(f"\nLast 15 reversals:")
    print(reversals_df[['index', 'timestamp', 'type', 'close']].tail(15).to_string())
else:
    print(f"\nNo reversals detected")

print(f"\n{'='*70}")
print("PHASE1 Complete!")
print(f"Output file: {output_file}")
print(f"{'='*70}")
