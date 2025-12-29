#!/usr/bin/env python3
"""
PHASE1 Simple Reversal Detection - Colab Version

Usage in Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_simple.py | python3

Core Logic:
- Find local lows: close is lowest in past N bars AND future N bars
- Find local highs: close is highest in past N bars AND future N bars
- No indicators needed - just pure price action
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE1: Simple Reversal Detection (God's View)")
print("="*70)

print("\nLoading data from HuggingFace...")
from huggingface_hub import hf_hub_download

try:
    csv_file = hf_hub_download(
        repo_id="zongowo111/cpb-models",
        filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(csv_file)
    print(f"Successfully downloaded")
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

if 'timestamp' not in df.columns:
    if 'open_time' in df.columns:
        df['timestamp'] = df['open_time']
    else:
        df['timestamp'] = range(len(df))

print(f"Data shape: {df.shape}")
print(f"Date range: {str(df['timestamp'].iloc[0])[:19]} to {str(df['timestamp'].iloc[-1])[:19]}")

lookback = 5
lookforward = 10

print(f"\nDetecting reversals (lookback={lookback}, lookforward={lookforward})...")

reversals = []

# Find local lows
print(f"\n[1/2] Finding local lows...")
local_low_count = 0
for i in range(lookback, len(df) - lookforward):
    current_close = df.iloc[i]['close']
    
    lookback_low = df.iloc[i-lookback:i]['close'].min()
    lookforward_low = df.iloc[i+1:i+lookforward+1]['close'].min()
    
    if current_close <= lookback_low and current_close <= lookforward_low:
        reversals.append({
            'index': i,
            'timestamp': df.iloc[i]['timestamp'],
            'close': current_close,
            'type': 'Local_Low'
        })
        local_low_count += 1

print(f"Found {local_low_count} local lows")

# Find local highs
print(f"\n[2/2] Finding local highs...")
local_high_count = 0
for i in range(lookback, len(df) - lookforward):
    current_close = df.iloc[i]['close']
    
    lookback_high = df.iloc[i-lookback:i]['close'].max()
    lookforward_high = df.iloc[i+1:i+lookforward+1]['close'].max()
    
    if current_close >= lookback_high and current_close >= lookforward_high:
        reversals.append({
            'index': i,
            'timestamp': df.iloc[i]['timestamp'],
            'close': current_close,
            'type': 'Local_High'
        })
        local_high_count += 1

print(f"Found {local_high_count} local highs")

reversals_df = pd.DataFrame(reversals).sort_values('index').reset_index(drop=True) if reversals else pd.DataFrame()

print(f"\nLabeling dataset...")
df['Reversal_Label'] = 0
df['Reversal_Type'] = 'None'
df['Reversal_Price'] = 0.0

for idx, row in reversals_df.iterrows():
    i = int(row['index'])
    if i < len(df):
        df.at[i, 'Reversal_Label'] = 1
        df.at[i, 'Reversal_Type'] = row['type']
        df.at[i, 'Reversal_Price'] = row['close']

output_file = 'labeled_klines_phase1_simple.csv'
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
        'Reversal_Label', 'Reversal_Type', 'Reversal_Price']
existing_cols = [c for c in cols if c in df.columns]
df[existing_cols].to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")

print(f"\n{'='*70}")
print("PHASE1 Detection Results Summary")
print(f"{'='*70}")

total = len(reversals_df)
local_lows = len(reversals_df[reversals_df['type'] == 'Local_Low'])
local_highs = len(reversals_df[reversals_df['type'] == 'Local_High'])

print(f"\nTotal reversal points: {total}")
print(f"  - Local Lows (Buy Points): {local_lows}")
print(f"  - Local Highs (Sell Points): {local_highs}")
print(f"\nReversal ratio: {total / len(df) * 100:.2f}%")
print(f"Average spacing: {len(df) // (total + 1):.0f} bars")

if len(reversals_df) > 0:
    print(f"\nFirst 10 reversals:")
    print(reversals_df[['index', 'timestamp', 'type', 'close']].head(10).to_string())
    
    print(f"\nLast 10 reversals:")
    print(reversals_df[['index', 'timestamp', 'type', 'close']].tail(10).to_string())
else:
    print(f"\nNo reversals detected")

print(f"\n{'='*70}")
print("PHASE1 Complete!")
print(f"Output file: {output_file}")
print(f"{'='*70}")
